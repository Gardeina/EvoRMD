import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef, accuracy_score

from utils import gather_targets, gather_probs, gather_sequences, unwrap_module
from embedding import get_rnafm_with_hier_species_before_attention


def train(classifier, attention_layer, train_dataloader, criterion, optimizer,
          rnafm_model, batch_converter, device, world_size, 
          hier_encoder, species_encoder, trainable=False):
    classifier.train(); attention_layer.train()
    epoch_train_loss = 0
    train_acc = 0
    train_loss = 0
    train_preds, train_targets, train_probs,train_embs = [], [], [], []
    train_seqs, train_attention = [], []

    for batch in tqdm(train_dataloader, desc="Training", leave=False):
        seqs, labels, organs, cells, subcells, species = batch
        labels = labels.to(device)
        train_seqs.extend(list(seqs))

        token_concat, mil_embeddings, attention_weights = get_rnafm_with_hier_species_before_attention(
            sequences=list(seqs), organs=list(organs), cells=list(cells), subcells=list(subcells), species=list(species),
            rnafm_model=rnafm_model, batch_converter=batch_converter, attention_layer=attention_layer,
            hier_encoder=hier_encoder, species_encoder=species_encoder, device=device,trainable=bool(trainable)
        )
        outputs = classifier(mil_embeddings)

        loss = criterion(outputs, labels)
        reduced_loss = loss.detach().clone()
        dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
        reduced_loss /= world_size
        epoch_train_loss += reduced_loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        probs = F.softmax(outputs, dim=1)
        preds = torch.max(probs, 1)[1]

        train_preds.append(preds)
        train_targets.append(labels)
        train_probs.append(probs)
        train_embs.append(mil_embeddings.detach().cpu())
        train_attention.append(attention_weights.detach().cpu())

    train_preds = torch.cat(train_preds, dim=0)
    train_targets = torch.cat(train_targets, dim=0)
    train_probs = torch.cat(train_probs, dim=0)
    train_embs = torch.cat(train_embs, dim=0)
    
    
    train_targets_global = gather_targets(train_targets)
    train_preds_global = gather_targets(train_preds)
    train_probs_global = gather_probs(train_probs)
    train_attention_global = gather_probs(torch.cat(train_attention, dim=0).to(device))
    train_seqs_global = gather_sequences(train_seqs)
    train_embs_global = gather_probs(train_embs.to(device)).cpu()

    if dist.get_rank() == 0:
        train_acc = (train_preds_global == train_targets_global).float().mean().item()
        train_loss = epoch_train_loss / len(train_dataloader)

    train_acc = torch.tensor(train_acc).to(device)
    dist.broadcast(train_acc, src=0)
    
    train_loss = torch.tensor(train_loss).to(device)
    dist.broadcast(train_loss, src=0)

    train_dict = {
            'probs': train_probs_global.cpu(),
            'preds': train_preds_global.cpu(),
            'targets': train_targets_global.cpu(),
            'sequences': train_seqs_global,
            'attention_weights': train_attention_global.cpu().numpy(),
            'embeddings': train_embs_global.numpy()
        }

    return train_acc.item(), train_loss.item(),train_dict

def validate(classifier, attention_layer, val_dataloader, criterion, 
             rnafm_model, batch_converter, device, world_size, 
             hier_encoder, species_encoder, trainable=False):
    classifier.eval(); attention_layer.eval()
    epoch_val_loss = 0
    val_acc = 0
    val_loss = 0
    val_preds, val_targets, val_probs,val_logits, val_embs = [], [], [], [], []
    val_seqs, val_attention = [], []

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation", leave=False):
            seqs, labels, organs, cells, subcells, species = batch
            labels = labels.to(device)
            val_seqs.extend(list(seqs))

            token_embeddings, embeddings, attention_weights = get_rnafm_with_hier_species_before_attention(
                sequences=list(seqs), organs=list(organs), cells=list(cells), subcells=list(subcells), species=list(species),
                rnafm_model=rnafm_model, batch_converter=batch_converter, attention_layer=attention_layer, device=device,
                hier_encoder=hier_encoder, species_encoder=species_encoder, trainable=bool(trainable)
            )
            outputs = classifier(embeddings)
            logits = outputs
            val_attention.append(attention_weights.detach().cpu())

            loss = criterion(outputs, labels)
            reduced_loss = loss.detach().clone()
            dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
            reduced_loss /= world_size
            epoch_val_loss += reduced_loss.item()

            probs = F.softmax(outputs, dim=1)
            preds = torch.max(probs, 1)[1]

            val_preds.append(preds)
            val_targets.append(labels)
            val_probs.append(probs)
            val_logits.append(logits)
            val_embs.append(embeddings.detach().cpu())

    val_logits = torch.cat(val_logits, dim=0) 
    val_preds = torch.cat(val_preds, dim=0)
    val_targets = torch.cat(val_targets, dim=0)
    val_probs = torch.cat(val_probs, dim=0)
    val_embs = torch.cat(val_embs, dim=0)

    val_logits_global   = gather_probs(val_logits)
    val_targets_global = gather_targets(val_targets)
    val_preds_global = gather_targets(val_preds)
    val_probs_global = gather_probs(val_probs)
    val_attention_global = gather_probs(torch.cat(val_attention, dim=0).to(device))
    val_seqs_global = gather_sequences(val_seqs)
    val_embs_global = gather_probs(val_embs.to(device)).cpu()

    if dist.get_rank() == 0:
        val_acc = (val_preds_global == val_targets_global).float().mean().item()
        val_loss = epoch_val_loss / len(val_dataloader)

    val_acc = torch.tensor(val_acc).to(device)
    dist.broadcast(val_acc, src=0)
    
    val_loss = torch.tensor(val_loss).to(device)
    dist.broadcast(val_loss, src=0)

    val_dict = {
            "logits": val_logits_global.cpu() , 
            'probs': val_probs_global.cpu(),
            'preds': val_preds_global.cpu(),
            'targets': val_targets_global.cpu(),
            'sequences': val_seqs_global,
            'attention_weights': val_attention_global.cpu().numpy(),
            'embeddings': val_embs_global.numpy()
        }

    return val_acc.item(), val_loss.item(), val_dict

# Test function

def test(classifier, attention_layer, test_dataloader, best_model_path, num_classes,
         rnafm_model, batch_converter, device, label_encoder, hier_encoder, species_encoder, trainable=False, rank=0):
    # Load checkpoint dict
    checkpoint = torch.load(best_model_path, map_location=device)
    unwrap_module(classifier).load_state_dict(checkpoint["classifier"])
    unwrap_module(attention_layer).load_state_dict(checkpoint["attention"])
    print("Loaded classifier and attention layer weights.")

    classifier.eval();attention_layer.eval()

    test_preds, test_targets, test_probs, test_attention, test_seqs, test_logits, test_embs = [], [], [], [], [], [], []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing", leave=False):
            seqs, labels, organs, cells, subcells, species = batch
            labels = labels.to(device)
            token_embeddings, embeddings, attention_weights = get_rnafm_with_hier_species_before_attention(
                sequences=list(seqs), organs=list(organs), cells=list(cells), subcells=list(subcells), species=list(species),
                rnafm_model=rnafm_model, batch_converter=batch_converter, attention_layer=attention_layer, device=device,
                hier_encoder=hier_encoder, species_encoder=species_encoder, trainable=bool(trainable)
            )
            outputs = classifier(embeddings)
            logits = outputs.detach().cpu()

            probs = F.softmax(outputs, dim=1)
            preds = torch.max(probs, dim=1)[1]

            test_preds.append(preds)
            test_targets.append(labels)
            test_probs.append(probs)
            test_logits.append(logits)
            test_attention.append(attention_weights.detach().cpu())
            test_seqs.extend(list(seqs))
            test_embs.append(embeddings.detach().cpu())

    test_preds = torch.cat(test_preds, dim=0)
    test_targets = torch.cat(test_targets, dim=0)
    test_probs = torch.cat(test_probs, dim=0)
    test_logits    = torch.cat(test_logits, dim=0) 
    test_attention = torch.cat(test_attention, dim=0)
    test_embs = torch.cat(test_embs, dim=0)

    overall_acc = (test_preds == test_targets).float().mean().cpu().item()
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_targets.cpu(), test_preds.cpu(), average="weighted", labels=range(num_classes), zero_division=1
    )
    mcc = matthews_corrcoef(test_targets.cpu().numpy(), test_preds.cpu().numpy())

    if rank == 0:
        print("\n=== Overall Metrics ===")
        print(f"Accuracy: {overall_acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"MCC: {mcc:.4f}")

        # Per-class metrics
        print("\n=== Per-class Metrics ===")
        for i in range(num_classes):
            mask = (test_targets == i)
            label = label_encoder.classes_[i]
            if mask.sum() > 0:
                acc_i = accuracy_score(test_targets[mask].cpu(), test_preds[mask].cpu())
            else:
                acc_i = 0
            prec_i, rec_i, f1_i, _ = precision_recall_fscore_support(
                test_targets.cpu(), test_preds.cpu(), average=None, labels=[i], zero_division=1
            )
            mcc_i = matthews_corrcoef(
                (test_targets.cpu().numpy() == i).astype(int),
                (test_preds.cpu().numpy() == i).astype(int)
            )
            print(f"Class {label}:")
            print(f"  Accuracy: {acc_i:.4f}")
            print(f"  Precision: {prec_i[0]:.4f}")
            print(f"  Recall: {rec_i[0]:.4f}")
            print(f"  F1-score: {f1_i[0]:.4f}")
            print(f"  MCC: {mcc_i:.4f}")

    test_dict = {
        "logits": test_logits.cpu() , 
        "probs": test_probs.cpu(),
        "preds": test_preds.cpu(),
        "targets": test_targets.cpu(),
        "sequences": test_seqs,
        "attention_weights": test_attention.numpy(),
        "embeddings": test_embs.numpy()
    }
    return overall_acc, test_probs, test_preds, test_targets, test_dict