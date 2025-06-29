import argparse
import torch
import torch.nn.functional as F
import fm
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from collections import Counter
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, accuracy_score,matthews_corrcoef
from tqdm import tqdm
import pickle
from sklearn.metrics import confusion_matrix
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

class RNAFMDataset(Dataset):
    def __init__(self, data):
        self.sequences = data['Sequence_41'].tolist()
        self.labels = data['modType'].values

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        labels = self.labels[idx]
        return seq, labels

# Define the classifier
class MulticlassClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(MulticlassClassifier, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(self.relu(x))
        x = self.fc2(x)
        return x
    
# Trainable Attention Layer for embedding
class TrainableAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(TrainableAttention, self).__init__()
        self.attention = nn.Linear(embedding_dim, 1)  # Learnable scoring function

    def forward(self, token_embeddings):
        # token_embeddings: (batch_size, seq_length, embedding_dim)
        attention_scores = self.attention(token_embeddings).squeeze(-1)  # Shape: (batch_size, seq_length)
        attention_weights = F.softmax(attention_scores, dim=-1)  # Normalize to probabilities
        return attention_weights
    
# Function to extract embeddings for a batch of sequences
def get_rna_fm_embeddings(sequences, model, batch_converter, device, attention_layer):
    # Convert sequences into batch format
    batch_labels, batch_strs, batch_tokens = batch_converter([(None, seq) for seq in sequences])
    batch_tokens = batch_tokens.to(device)

    with torch.set_grad_enabled(model.training):
        results = model.module(batch_tokens, repr_layers=[12])
    
    # Extract embeddings for all tokens in layer 12
    token_embeddings = results["representations"][12]  # Shape: (batch_size, seq_length, embedding_dim)

    # Compute trainable attention weights
    attention_weights = attention_layer(token_embeddings)  # Shape: (batch_size, seq_length)

    # Compute weighted sum of all tokens using attention
    mil_embeddings = torch.bmm(
        attention_weights.unsqueeze(1), token_embeddings
    ).squeeze(1)  # Shape: (batch_size, embedding_dim)

    return token_embeddings,mil_embeddings,attention_weights

# Function to count class distribution in a dataset
def count_class_distribution(dataset):
    class_counts = Counter()
    for _, label in dataset:
        class_counts[label] += 1
    return class_counts

def setup_distributed():
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    return rank, dist.get_world_size()

def cleanup_distributed():
    dist.destroy_process_group()

def reduce_tensor(tensor: torch.Tensor):
    rt = tensor.clone()  
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()  
    return rt

def gather_targets(tensor: torch.Tensor):
    world_size = dist.get_world_size()
    gather_list = [torch.zeros_like(tensor, dtype=torch.long) for _ in range(world_size)]
    dist.all_gather(gather_list, tensor)
    return torch.cat(gather_list, dim=0)

def gather_probs(tensor: torch.Tensor):
    world_size = dist.get_world_size()
    gather_list = [torch.zeros_like(tensor, dtype=torch.float32) for _ in range(world_size)]
    dist.all_gather(gather_list, tensor)
    return torch.cat(gather_list, dim=0)

def gather_sequences(local_seqs: list, device=None):
    world_size = dist.get_world_size()
    gathered_list = [None] * world_size
    dist.all_gather_object(gathered_list, local_seqs)
    global_seqs = [seq for sublist in gathered_list for seq in sublist]
    return global_seqs

# Train function
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
    attention_embeddings = []
    rnafm_embeddings = []
    all_labels = []

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
        #preds[torch.max(probs, 1)[0] < threshold] = 13  # Assign to 'no modification'

        train_preds.append(preds)
        train_targets.append(labels)
        train_probs.append(probs)
        # rnafm_embeddings.append(token_embeddings)
        # attention_embeddings.append(embeddings)

    train_preds = torch.cat(train_preds, dim=0)
    train_targets = torch.cat(train_targets, dim=0)
    train_probs = torch.cat(train_probs, dim=0)
    
    train_targets_global = gather_targets(train_targets)
    train_preds_global = gather_targets(train_preds)
    train_probs_global = gather_probs(train_probs)
    train_attention_global = gather_probs(torch.cat(train_attention, dim=0).to(device))
    train_seqs_global = gather_sequences(train_seqs)
    # rnafm_embeddings_global = gather_probs(torch.cat(rnafm_embeddings, dim=0))
    # attention_embeddings_global = gather_probs(torch.cat(attention_embeddings, dim=0))

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
    attention_embeddings = []
    rnafm_embeddings = []

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
            # rnafm_embeddings.append(token_embeddings)
            # attention_embeddings.append(embeddings)

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
def test(classifier, attention_layer, test_dataloader, best_model_path,num_classes, model, batch_converter, device, threshold=0.2):
    # Load the best model
    classifier = torch.load(best_model_path, map_location=device)
    #classifier.load_state_dict(torch.load(best_model_path))
    print(f'Model load: {classifier.state_dict()}')
    if isinstance(classifier, DDP):
        classifier.module.to(device)
    else:
        classifier.to(device)

    classifier.eval()
    attention_layer.eval()

    test_preds = []
    test_targets = []
    test_probs = []
    test_seqs = []
    test_attention = []
    attention_embeddings = []
    rnafm_embeddings = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing", leave=False):
            seqs, labels = batch
            labels = labels.to(device)
            for seq in seqs:
                if len(seq) >= 21:
                    test_seqs.append(seq)
                else:
                    
                    test_seqs.append(seq)

            token_embeddings,embeddings ,attention_weights = get_rna_fm_embeddings(seqs, model, batch_converter, device, attention_layer)
            outputs = classifier(embeddings)
            test_attention.append(attention_weights.detach().cpu())

            probs = F.softmax(outputs, dim=1)
            preds = torch.max(probs, 1)[1]
           # preds[torch.max(probs, 1)[0] < threshold] = 13

            test_preds.append(preds)
            test_targets.append(labels)
            test_probs.append(probs)

    test_preds = torch.cat(test_preds, dim=0)
    test_targets = torch.cat(test_targets, dim=0)
    test_probs = torch.cat(test_probs, dim=0)

    test_attention = torch.cat(test_attention, dim=0)
    test_acc = (test_preds == test_targets).float().mean().cpu()
    
    test_dict = {
        'test_probs': test_probs.cpu(),
        'test_preds': test_preds.cpu(),
        'test_targets': test_targets.cpu(),
        'sequences': test_seqs,
        'attention_weights':test_attention
    }

    precision, recall, f1, _ = precision_recall_fscore_support(
        test_targets.cpu(), test_preds.cpu(), average='weighted', labels=range(num_classes), zero_division=1)
    mcc = matthews_corrcoef(test_targets.cpu().numpy(),test_preds.cpu().numpy())
    print(f"Overall accuracy: {test_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"MCC: {mcc:.4f}")

    return test_acc, test_probs, test_preds, test_targets,test_dict

# Main function
def main(args):
    rank, world_size = setup_distributed()
    device = torch.device(f'cuda:{rank}')
    print(f"Using device: {device}")
    torch.cuda.set_device(device)
    

    model, alphabet = fm.pretrained.rna_fm_t12()
    batch_converter = alphabet.get_batch_converter()
    model.to(device)
    model = DDP(model,device_ids=[rank])

    data = pd.read_pickle('../Data/11_modif_preprocessed_data.pkl')
    with open('../Data/11_modif_label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

    classifier_label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
    num_classes = len(classifier_label_mapping)

    dataset = RNAFMDataset(data)
    train_size = int(len(dataset) * 0.8)
    val_size = int(len(dataset) * 0.1)
    test_size = len(dataset) - train_size - val_size

    torch.manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle = True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle = False)

    train_class_distribution = count_class_distribution(train_dataset)
    val_class_distribution = count_class_distribution(val_dataset)
    test_class_distribution = count_class_distribution(test_dataset)
    if rank == 0:
        print("Training set class distribution:", train_class_distribution)
        print("Validation set class distribution:", val_class_distribution)
        print("Test set class distribution:", test_class_distribution)

    train_dataloader = DataLoader(train_dataset, batch_size=128, sampler = train_sampler, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=128, sampler = val_sampler , shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    classifier = MulticlassClassifier(embedding_dim=640, num_classes=num_classes).to(device)
    attention_layer = TrainableAttention(embedding_dim=640).to(device)
    classifier = DDP(classifier, device_ids=[rank])
    attention_layer = DDP(attention_layer, device_ids=[rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        list(classifier.parameters()) + list(attention_layer.parameters()) + list(model.parameters()), lr=0.001
    )

    best_val_loss = float('inf')
    best_model_path = "best_model.pth"
    best_train_dict = None
    best_val_dict = None

    for epoch in range(args.num_epochs):
        train_sampler.set_epoch(epoch)
        train_acc, train_loss,train_dict = train(classifier, attention_layer, train_dataloader, criterion, optimizer, model, batch_converter, device,world_size, args.threshold)
        val_acc, val_loss, val_dict = validate(classifier, attention_layer, val_dataloader, criterion, model, batch_converter, device,world_size, args.threshold)

        if rank == 0 :
            print(f"Epoch {epoch+1}/{args.num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        if rank == 0 and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(classifier, best_model_path)
            print(f'Model saved: {classifier.state_dict()}')
            best_train_dict = train_dict  
            best_val_dict = val_dict
            print(f"New best model saved at epoch {epoch+1}.")

    test_acc, test_probs, test_preds, test_targets,test_dict = test(classifier, attention_layer, test_dataloader, best_model_path,num_classes, model, batch_converter, device, args.threshold)
    if rank == 0:
        best_save_dict = {
            'train': best_train_dict,
            'val': best_val_dict,
            'test': test_dict
        }
        with open("../Model/EvoRMD.pkl", "wb") as f:
            pickle.dump(best_save_dict, f)
        print("best_model_train_val_test_results.pkl has saved")

        # best_embedding_dict={'train':best_train_emb,'val':best_val_emb,'test':test_embedding_dict}
        # with open("/home/bingxing2/ailab/group/ai4agr/yzh/my/Attention/analysis/embedding.pkl", "wb") as f:
        #     pickle.dump(best_embedding_dict, f)
        # print("best_embedding_dict.pkl has saved")
    #Calculate accuracy for each class
    if rank ==0:
        class_accuracy = {}
        for i in range(num_classes):
            class_mask = (test_targets == i)
            original_label = label_encoder.classes_[i]
            if class_mask.sum() > 0: 
                accuracy = accuracy_score(test_targets[class_mask].cpu(), test_preds[class_mask].cpu())
                class_accuracy[original_label] = accuracy
            else:
                class_accuracy[original_label] = 0

            precision_i, recall_i, f1_i, _ = precision_recall_fscore_support(
                test_targets.cpu(), test_preds.cpu(), average=None, labels=[i], zero_division=1)
            mcc_i = matthews_corrcoef(test_targets.cpu().numpy(),test_preds.cpu().numpy())

            print(f"Class {original_label}:")
            print(f"  Accuracy: {accuracy:.4f},Precision: {precision_i[0]:.4f},Recall: {recall_i[0]:.4f},F1-score: {f1_i[0]:.4f},MCC:{mcc_i:.4f}")

    cleanup_distributed()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a MuticlassClassifier model with specified hyperparameters.")
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--threshold', type=float, default=0.2, help='If the highest probability is below the threshold, classify as no modification')
    parser.add_argument('--trainable', type=int, choices=[0, 1], default=0, help='Set RNA-FM model to be trainable (1) or not (0)')

    args = parser.parse_args()
    main(args)
