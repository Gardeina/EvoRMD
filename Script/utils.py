import torch
import torch.distributed as dist

def setup_distributed():
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    return rank, dist.get_world_size()

def cleanup_distributed():
    dist.destroy_process_group()

def get_rna_fm_embeddings(sequences, model, batch_converter, device, attention_layer):
    batch_labels, batch_strs, batch_tokens = batch_converter([(None, seq) for seq in sequences])
    batch_tokens = batch_tokens.to(device)
    with torch.set_grad_enabled(model.training):
        results = model.module(batch_tokens, repr_layers=[12])
    token_embeddings = results["representations"][12]
    attention_weights = attention_layer(token_embeddings)
    mil_embeddings = torch.bmm(
        attention_weights.unsqueeze(1), token_embeddings
    ).squeeze(1)
    return token_embeddings, mil_embeddings, attention_weights

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
