
import os
import torch
import torch.distributed as dist
from collections import Counter
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel

def setup_distributed():
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    return rank, dist.get_world_size()

def cleanup_distributed():
    dist.destroy_process_group()

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

def count_class_distribution(dataset):
    class_counts = Counter()
    for item in dataset:
        label = item[1]  
        class_counts[label] += 1
    return class_counts

def _unwrap_module(m):
    if isinstance(m, (DDP, nn.DataParallel)):
        return m.module
    return m

def unwrap_module(m):
    return m.module if isinstance(m, (DDP, DataParallel)) else m

def ddp_barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
