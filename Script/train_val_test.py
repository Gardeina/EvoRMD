import torch
import torch.nn.functional as F
from tqdm import tqdm
from .utils import get_rna_fm_embeddings, gather_targets, gather_probs, gather_sequences
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef, accuracy_score


def train(classifier, attention_layer, train_dataloader, criterion, optimizer, model, batch_converter, device, world_size, threshold=0.2):
    classifier.train()
    attention_layer.train()
    epoch_train_loss = 0
    train_acc = 0
    train_loss = 0
    train_preds = []
    train_targets = []
    train_probs = []
    train_seqs = []
    train_attention = []

    for batch in tqdm(train_dataloader, desc="Training", leave=False):
        seqs, labels = batch
        labels = labels.to(device)
        for seq in seqs:
            if len(seq) >= 21:
                train_seqs.append(seqs)
            else:
                train_seqs.append(seqs)

        token_embeddings,embeddings,attention_weights = get_rna_fm_embeddings(seqs, model, batch_converter, device, attention_layer)
        outputs = classifier(embeddings)

        train_attention.append(attention_weights.detach().cpu())

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

    train_preds = torch.cat(train_preds, dim=0)
    train_targets = torch.cat(train_targets, dim=0)
    train_probs = torch.cat(train_probs, dim=0)
    
    train_targets_global = gather_targets(train_targets)
    train_preds_global = gather_targets(train_preds)
    train_probs_global = gather_probs(train_probs)
    train_attention_global = gather_probs(torch.cat(train_attention, dim=0).to(device))
    train_seqs_global = gather_sequences(train_seqs)

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
            'attention_weights': train_attention_global.cpu().numpy()
        }

    return train_acc.item(), train_loss.item(),train_dict

def validate(classifier, attention_layer, val_dataloader, criterion, model, batch_converter, device, world_size, threshold=0.2):
    classifier.eval()
    attention_layer.eval()
    epoch_val_loss = 0
    val_acc = 0
    val_loss = 0
    val_preds = []
    val_targets = []
    val_probs = []
    val_seqs = []
    val_attention = []

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation", leave=False):
            seqs, labels = batch
            labels = labels.to(device)
            for seq in seqs:
                if len(seq) >= 21:
                    val_seqs.append(seqs)
                else:
                    
                    val_seqs.append(seqs)

            token_embeddings,embeddings ,attention_weights= get_rna_fm_embeddings(seqs, model, batch_converter, device, attention_layer)
            outputs = classifier(embeddings)
            val_attention.append(attention_weights.detach().cpu())

            loss = criterion(outputs, labels)
            reduced_loss = loss.detach().clone()
            dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
            reduced_loss /= world_size
            epoch_val_loss += reduced_loss.item()

            probs = F.softmax(outputs, dim=1)
            preds = torch.max(probs, 1)[1]
            preds[torch.max(probs, 1)[0] < threshold] = 13

            val_preds.append(preds)
            val_targets.append(labels)
            val_probs.append(probs)

    val_preds = torch.cat(val_preds, dim=0)
    val_targets = torch.cat(val_targets, dim=0)
    val_probs = torch.cat(val_probs, dim=0)
    val_targets_global = gather_targets(val_targets)
    val_preds_global = gather_targets(val_preds)
    val_probs_global = gather_probs(val_probs)
    val_attention_global = gather_probs(torch.cat(val_attention, dim=0).to(device))
    val_seqs_global = gather_sequences(val_seqs)

    if dist.get_rank() == 0:
        val_acc = (val_preds_global == val_targets_global).float().mean().item()
        val_loss = epoch_val_loss / len(val_dataloader)

    val_acc = torch.tensor(val_acc).to(device)
    dist.broadcast(val_acc, src=0)
    
    val_loss = torch.tensor(val_loss).to(device)
    dist.broadcast(val_loss, src=0)

    val_dict = {
            'probs': val_probs_global.cpu(),
            'preds': val_preds_global.cpu(),
            'targets': val_targets_global.cpu(),
            'sequences': val_seqs_global,
            'attention_weights': val_attention_global.cpu().numpy()
        }

    return val_acc.item(), val_loss.item(), val_dict

# Test function

def test(
    classifier,
    attention_layer,
    test_dataloader,
    best_model_path,
    num_classes,
    model,
    batch_converter,
    device,
    label_encoder,
    threshold=0.2,
    rank=0
):
    # Load checkpoint dict
    checkpoint = torch.load(best_model_path, map_location=device)
    classifier.load_state_dict(checkpoint["classifier"])
    attention_layer.load_state_dict(checkpoint["attention"])
    print("Loaded classifier and attention layer weights.")

    classifier.eval()
    attention_layer.eval()

    all_preds, all_targets, all_probs, all_attention, all_seqs = [], [], [], [], []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing", leave=False):
            seqs, labels = batch
            labels = labels.to(device)
            token_embeddings, embeddings, attention_weights = get_rna_fm_embeddings(
                seqs, model, batch_converter, device, attention_layer
            )
            outputs = classifier(embeddings)

            probs = F.softmax(outputs, dim=1)
            preds = torch.max(probs, dim=1)[1]

            all_preds.append(preds)
            all_targets.append(labels)
            all_probs.append(probs)
            all_attention.append(attention_weights.cpu())
            all_seqs.extend(seqs)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_probs = torch.cat(all_probs, dim=0)
    all_attention = torch.cat(all_attention, dim=0)

    overall_acc = (all_preds == all_targets).float().mean().cpu().item()
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets.cpu(), all_preds.cpu(), average="weighted", labels=range(num_classes), zero_division=1
    )
    mcc = matthews_corrcoef(all_targets.cpu().numpy(), all_preds.cpu().numpy())

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
            mask = (all_targets == i)
            label = label_encoder.classes_[i]
            if mask.sum() > 0:
                acc_i = accuracy_score(all_targets[mask].cpu(), all_preds[mask].cpu())
            else:
                acc_i = 0
            prec_i, rec_i, f1_i, _ = precision_recall_fscore_support(
                all_targets.cpu(), all_preds.cpu(), average=None, labels=[i], zero_division=1
            )
            mcc_i = matthews_corrcoef(
                (all_targets.cpu().numpy() == i).astype(int),
                (all_preds.cpu().numpy() == i).astype(int)
            )
            print(f"Class {label}:")
            print(f"  Accuracy: {acc_i:.4f}")
            print(f"  Precision: {prec_i[0]:.4f}")
            print(f"  Recall: {rec_i[0]:.4f}")
            print(f"  F1-score: {f1_i[0]:.4f}")
            print(f"  MCC: {mcc_i:.4f}")

    test_dict = {
        "probs": all_probs.cpu(),
        "preds": all_preds.cpu(),
        "targets": all_targets.cpu(),
        "sequences": all_seqs,
        "attention_weights": all_attention.numpy(),
    }
    return overall_acc, all_probs, all_preds, all_targets, test_dict
