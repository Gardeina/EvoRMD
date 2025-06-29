import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import pickle
from torch.utils.data import DataLoader, random_split, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from src.dataset import RNAFMDataset
from src.model import MulticlassClassifier, TrainableAttention
from src.train_val_test import train,validate, test  
from src.utils import setup_distributed, cleanup_distributed
import fm
from src.calibration import (
    find_best_thresholds_torch,
    evaluate_on_test_torch,
    grid_search_thresholds
)

def main(args):
    rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    model, alphabet = fm.pretrained.rna_fm_t12()
    batch_converter = alphabet.get_batch_converter()
    model.to(device)
    model = DDP(model, device_ids=[rank])

    data = pd.read_pickle(args.data)
    with open(args.label, "rb") as f:
        label_encoder = pickle.load(f)
    num_classes = len(label_encoder.classes_)

    dataset = RNAFMDataset(data)
    train_size = int(len(dataset)*0.8)
    val_size = int(len(dataset)*0.1)
    test_size = len(dataset)-train_size-val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler = DistributedSampler(test_set, num_replicas=world_size, rank=rank, shuffle=False)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, sampler=val_sampler)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, sampler=test_sampler)

    classifier = MulticlassClassifier(embedding_dim=640, num_classes=num_classes).to(device)
    attention = TrainableAttention(embedding_dim=640).to(device)
    classifier = DDP(classifier, device_ids=[rank])
    attention = DDP(attention, device_ids=[rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        list(classifier.parameters()) + list(attention.parameters()), lr=args.lr
    )

    best_val_loss = float('inf')
    best_model_path = args.model
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        train_acc, train_loss,train_dict = train(
            classifier, attention, train_loader, criterion, optimizer,
            model, batch_converter, device, world_size, args.threshold
        )
        val_acc, val_loss,val_dict = validate(classifier, attention, val_loader, criterion, model, batch_converter, device, world_size, args.threshold)
        
        if rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        if rank == 0 and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(classifier, best_model_path)
            print(f"New best model saved at epoch {epoch+1}.")
            best_train_dict = train_dict  
            best_val_dict = val_dict

        test_acc, test_probs, test_preds, test_targets, test_dict = test(classifier, attention, test_loader, model_path, num_classes, model, batch_converter, device, args.threshold)
        if rank == 0:
            best_save_dict = {
                'train': best_train_dict,
                'val': best_val_dict,
                'test': test_dict
            }
            with open(args.results, "wb") as f:
                pickle.dump(best_save_dict, f)
            print("best_model_results.pkl has saved")

    cleanup_distributed()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="./RNAdata/11_modif_preprocessed_data.pkl")
    parser.add_argument('--label', type=str, default="./RNAdata/11_label_encoder.pkl")
    parser.add_argument('--model', type=str, default="./Model/11_best_model.pth")
    parser.add_argument('--result', type=str, default="./11_best_model_results.pkl")
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--threshold', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    main(args)
