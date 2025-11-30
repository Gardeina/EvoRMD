# main.py
import os
import argparse
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import fm  # RNA-FM
# ==== Your own modules (local files in the same project) ====
from utils import setup_distributed, cleanup_distributed, count_class_distribution, ddp_barrier, unwrap_module
from downsampling import downsampling_from_raw
from model import RNAFMDataset, MulticlassClassifier, TrainableAttention
from embedding import build_hier_encoder_from_df, build_species_encoder_from_df
from train_val_test import train, validate, test

# ---------- Random seed ----------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    # ====== Initialize distributed training ======
    rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    if rank == 0:
        print(f"[Init] world_size={world_size}, device={device}")

    set_seed(args.seed)

    # ====== Load RNA-FM ======
    rnafm_model, alphabet = fm.pretrained.rna_fm_t12()
    batch_converter = alphabet.get_batch_converter()
    rnafm_model.to(device)
    for p in rnafm_model.parameters():
        p.requires_grad = bool(args.trainable)

    # Wrap RNA-FM with DDP only if it is trainable
    if any(p.requires_grad for p in rnafm_model.parameters()):
        rnafm_model = DDP(rnafm_model, device_ids=[rank])

    # ====== Downsample raw data ======
    data, label_encoder = downsampling_from_raw(
        raw_csv=args.raw_csv,
        alpha=args.alpha,
        target_mods=args.target_mods,
        min_keep=args.min_keep,
        seed=args.downsamplingseed
    )

    num_classes = len(label_encoder.classes_)
    if rank == 0:
        print(f"[Data] total samples={len(data)}, num_classes={num_classes}")

    # ====== Build encoders (vocab based on training set only) ======
    # First split dataset, then build vocab from train_df
    full_dataset = RNAFMDataset(data)  # This will check columns organ/cell/subcell/species (and fill 'unknown' if missing)

    train_size = int(len(full_dataset) * 0.8)
    val_size   = int(len(full_dataset) * 0.1)
    test_size  = len(full_dataset) - train_size - val_size

    set_seed(args.seed)
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    # To build vocab, we need the DataFrame subset corresponding to the training indices
    train_indices = train_dataset.indices
    train_df = data.iloc[train_indices].reset_index(drop=True)

    # Build hierarchical/species encoders
    META_DIM_HIER = 128
    SPECIES_DIM   = 32
    D_FM          = 640  # Representation dimension of RNA-FM T12

    hier_encoder, organ_vocab, cell_vocab, subcell_vocab = build_hier_encoder_from_df(
        train_df, meta_dim_hier=META_DIM_HIER, device=device
    )
    species_encoder, species_vocab = build_species_encoder_from_df(
        train_df, species_dim=SPECIES_DIM, device=device
    )

    # Freeze hierarchical/species encoders
    for p in hier_encoder.parameters():
        p.requires_grad = False
    for p in species_encoder.parameters():
        p.requires_grad = False

    # ====== Dataloaders + Distributed Samplers ======
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler   = DistributedSampler(val_dataset,   num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=128, sampler=train_sampler, shuffle=False)
    val_loader   = DataLoader(val_dataset,   batch_size=128, sampler=val_sampler,   shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=128, shuffle=False)

    if rank == 0:
        tr_dist = count_class_distribution(train_dataset)
        va_dist = count_class_distribution(val_dataset)
        te_dist = count_class_distribution(test_dataset)
        print("[Class Dist] train:", tr_dist)
        print("[Class Dist] valid:", va_dist)
        print("[Class Dist] test :", te_dist)

    # ====== Build Attention + Classifier ======
    EMB_DIM = D_FM + META_DIM_HIER + SPECIES_DIM  # 640 + 128 + 32 = 800
    attention_layer = TrainableAttention(EMB_DIM).to(device)
    classifier = MulticlassClassifier(EMB_DIM, num_classes, mlp_depth=args.mlp_depth).to(device)

    attention_layer = DDP(attention_layer, device_ids=[rank])
    classifier      = DDP(classifier,      device_ids=[rank])

    # ====== Loss and optimizer ======
    criterion = nn.CrossEntropyLoss()
    # If RNA-FM is frozen, do not include its parameters in the optimizer;
    # only add them when we fine-tune RNA-FM.
    params = list(classifier.parameters()) + list(attention_layer.parameters())
    if args.trainable == 1:
        params += list(rnafm_model.parameters())
    optimizer = optim.Adam(params, lr=1e-3)

    # ====== Training loop ======
    best_val_loss = float("inf")
    best_model_path = args.o
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
    best_train_dict = None
    best_val_dict   = None

    for epoch in range(args.num_epochs):
        train_sampler.set_epoch(epoch)

        train_acc, train_loss, train_dict = train(
            classifier=classifier,
            attention_layer=attention_layer,
            train_dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            rnafm_model=rnafm_model,
            batch_converter=batch_converter,
            device=device,
            world_size=world_size,
            hier_encoder=hier_encoder,
            species_encoder=species_encoder,
            trainable=bool(args.trainable),
        )
        val_acc, val_loss, val_dict = validate(
            classifier=classifier,
            attention_layer=attention_layer,
            val_dataloader=val_loader,
            criterion=criterion,
            rnafm_model=rnafm_model,
            batch_converter=batch_converter,
            device=device,
            world_size=world_size,
            hier_encoder=hier_encoder,
            species_encoder=species_encoder,
            trainable=bool(args.trainable),
        )

        if rank == 0:
            print(
                f"Epoch {epoch+1}/{args.num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}"
            )

        # ====== Save best model (only state_dict, to avoid DDP wrapper issues) ======
        if rank == 0 and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "classifier": unwrap_module(classifier).state_dict(),
                "attention": unwrap_module(attention_layer).state_dict(),
                "meta": {
                    "D_FM": D_FM,
                    "META_DIM_HIER": META_DIM_HIER,
                    "SPECIES_DIM": SPECIES_DIM,
                    "EMB_DIM": EMB_DIM,
                    "num_classes": num_classes,
                    "organ_vocab": organ_vocab,
                    "cell_vocab":  cell_vocab,
                    "subcell_vocab": subcell_vocab,
                    "species_vocab": species_vocab,
                }
            }, best_model_path)
            best_train_dict = train_dict
            best_val_dict   = val_dict
            print(f"[Checkpoint] New best saved at epoch {epoch+1}: {best_model_path}")
        ddp_barrier()

    # ====== Testing ======
    ddp_barrier()
    test_acc, test_probs, test_preds, test_targets, test_dict = test(
        classifier=classifier,
        attention_layer=attention_layer,
        test_dataloader=test_loader,
        best_model_path=best_model_path,
        num_classes=num_classes,
        rnafm_model=rnafm_model,
        batch_converter=batch_converter,
        device=device,
        label_encoder=label_encoder,
        hier_encoder=hier_encoder,
        species_encoder=species_encoder,
        trainable=False,
        rank=rank
    )

    # ====== Aggregate & save results (rank 0 only) ======
    if rank == 0:
        save_path = args.result
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        best_save_dict = {"train": best_train_dict, "val": best_val_dict, "test": test_dict}
        with open(save_path, "wb") as f:
            pickle.dump(best_save_dict, f)
        print(f"[Result] train/val/test dict saved to {save_path}")

    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train EvoRMD with RNA-FM + hierarchical/species pre-attention fusion."
    )
    parser.add_argument("--raw_csv", type=str, required=True,
                        help="Path to original raw dataset CSV file (before downsampling)")
    parser.add_argument("--alpha", type=float, default=0.6,
                        help="Log-compression coefficient for downsampling")
    parser.add_argument("--target_mods", nargs="+", default=["m6a", "m5c"],
                        help="Target modifications to downsample")
    parser.add_argument("--min_keep", type=int, default=1,
                        help="Minimum number of samples to keep per species")
    parser.add_argument("--downsamplingseed", type=int, default=42,
                        help="Random seed for downsampling")

    parser.add_argument("--mlp_depth", type=int, choices=[1, 2, 3], default=1,
                        help="Depth of MLP classifier: 1=Linear, 2=Two-layer, 3=Three-layer")

    parser.add_argument("--o", type=str, required=True,
                        help="Path to save trained model (.pth)")
    parser.add_argument("--result", type=str, required=True,
                        help="Path to save training/validation/test results (.pkl)")

    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--trainable", type=int, choices=[0, 1], default=0,
                        help="Whether to finetune RNA-FM (0: freeze, 1: train)")
    args = parser.parse_args()
    main(args)
