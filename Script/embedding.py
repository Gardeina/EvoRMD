# embedding.py — Hierarchical (organ / cell / subcell) + species (separate) + RNA-FM,
# fused together before the Attention layer

from __future__ import annotations
import re
import hashlib
from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn
import pandas as pd
from utils import _unwrap_module

_SPLIT_RE = re.compile(r"\s*,\s*")

def _split_multi(s: str) -> List[str]:
    """Split comma-separated string into a list of tokens."""
    s = str(s).strip()
    return [t for t in _SPLIT_RE.split(s)] if ("," in s) else [s]

def _build_vocab_from_df(df: pd.DataFrame, col: str) -> Dict[str, int]:
    """
    Build a vocabulary from a dataframe column.
    The column may contain comma-separated multi-label entries.
    Always ensure that 'unknown' is included in the vocabulary.
    """
    bag = []
    for v in df[col].astype(str).tolist():
        bag.extend(_split_multi(v))
    uniq = sorted(set(bag) | {"unknown"})  # make sure 'unknown' is always included
    return {tok: i for i, tok in enumerate(uniq)}

def _multi_hot_hybrid(items, vocab, device, hash_dim=256, normalize=False):
    """
    Multi-hot encoding with an additional hashed bucket region.

    items : iterable of tokens
    vocab : mapping token -> index
    hash_dim : size of the hashed region appended after the vocab region
    normalize : if True, L1-normalize the resulting vector
    """
    V, H = len(vocab), hash_dim
    v = torch.zeros(V + H, dtype=torch.float32, device=device)
    for it in items:
        if it in vocab:
            v[vocab[it]] = 1.0
        else:
            # Map OOV token to a hashed bucket
            h = int(
                hashlib.blake2b(str(it).encode(), digest_size=8).hexdigest(),
                16
            ) % H
            v[V + h] = 1.0
    if normalize:
        v = v / v.sum().clamp_min(1.0)
    return v

# ========== ① Hierarchical encoder (organ / cell / subcellular) ==========
class HierEncoder(nn.Module):
    """
    Encode hierarchical metadata (organ / cell / subcellular_location)
    into a shared latent space, with per-level encoders and LayerNorm.
    Each level uses a multi-hot + hash-based representation.
    """
    def __init__(
        self,
        organ_vocab: Dict[str, int],
        cell_vocab: Dict[str, int],
        subcell_vocab: Dict[str, int],
        meta_dim_hier: int = 128,
        normalize_per_level: bool = False,
        device: Optional[torch.device] = None,
        o_hash_dim: int = 256,
        c_hash_dim: int = 256,
        s_hash_dim: int = 256
    ):
        super().__init__()
        self.organ_vocab = organ_vocab
        self.cell_vocab = cell_vocab
        self.subcell_vocab = subcell_vocab
        self.normalize = normalize_per_level

        self.o_hash_dim = int(o_hash_dim)
        self.c_hash_dim = int(c_hash_dim)
        self.s_hash_dim = int(s_hash_dim)

        self.encoder_o = nn.Linear(
            len(organ_vocab) + o_hash_dim, meta_dim_hier, bias=True
        )
        self.encoder_c = nn.Linear(
            len(cell_vocab) + c_hash_dim, meta_dim_hier, bias=True
        )
        self.encoder_s = nn.Linear(
            len(subcell_vocab) + s_hash_dim, meta_dim_hier, bias=True
        )

        # Xavier init for all three linear layers
        for l in (self.encoder_o, self.encoder_c, self.encoder_s):
            nn.init.xavier_uniform_(l.weight)
            nn.init.zeros_(l.bias)

        self.ln_o = nn.LayerNorm(meta_dim_hier)
        self.ln_c = nn.LayerNorm(meta_dim_hier)
        self.ln_s = nn.LayerNorm(meta_dim_hier)

        if device is not None:
            self.to(device)

    def forward(self, organs: List[str], cells: List[str], subcells: List[str]) -> torch.Tensor:
        """
        organs, cells, subcells can be lists of strings or single strings.

        Uses _multi_hot_hybrid (with hashed region) to support multi-label
        annotations per level.
        """
        device = next(self.parameters()).device

        # Organ level
        if isinstance(organs, str):
            O = torch.stack(
                [
                    _multi_hot_hybrid(
                        _split_multi(organs),
                        self.organ_vocab,
                        device,
                        hash_dim=self.o_hash_dim,
                        normalize=self.normalize,
                    )
                ],
                dim=0,
            )
        else:
            O = torch.stack(
                [
                    _multi_hot_hybrid(
                        _split_multi(x),
                        self.organ_vocab,
                        device,
                        hash_dim=self.o_hash_dim,
                        normalize=self.normalize,
                    )
                    for x in organs
                ],
                dim=0,
            )

        # Cell level
        if isinstance(cells, str):
            C = torch.stack(
                [
                    _multi_hot_hybrid(
                        _split_multi(cells),
                        self.cell_vocab,
                        device,
                        hash_dim=self.c_hash_dim,
                        normalize=self.normalize,
                    )
                ],
                dim=0,
            )
        else:
            C = torch.stack(
                [
                    _multi_hot_hybrid(
                        _split_multi(x),
                        self.cell_vocab,
                        device,
                        hash_dim=self.c_hash_dim,
                        normalize=self.normalize,
                    )
                    for x in cells
                ],
                dim=0,
            )

        # Subcellular level
        if isinstance(subcells, str):
            S = torch.stack(
                [
                    _multi_hot_hybrid(
                        _split_multi(subcells),
                        self.subcell_vocab,
                        device,
                        hash_dim=self.s_hash_dim,
                        normalize=self.normalize,
                    )
                ],
                dim=0,
            )
        else:
            S = torch.stack(
                [
                    _multi_hot_hybrid(
                        _split_multi(x),
                        self.subcell_vocab,
                        device,
                        hash_dim=self.s_hash_dim,
                        normalize=self.normalize,
                    )
                    for x in subcells
                ],
                dim=0,
            )

        o_vec = self.ln_o(self.encoder_o(O))
        c_vec = self.ln_c(self.encoder_c(C))
        s_vec = self.ln_s(self.encoder_s(S))

        # Aggregate three levels by simple averaging
        meta_hier = (o_vec + c_vec + s_vec) / 3.0
        return meta_hier  # (B, meta_dim_hier)

def build_hier_encoder_from_df(
    df: pd.DataFrame,
    organ_col="organ",
    cell_col="cell",
    subcell_col="subcellular_location",
    meta_dim_hier=128,
    device: Optional[torch.device] = None,
    o_hash_dim: int = 256,
    c_hash_dim: int = 256,
    s_hash_dim: int = 256,
) -> Tuple[HierEncoder, Dict[str, int], Dict[str, int], Dict[str, int]]:
    """
    Build a HierEncoder and its vocabularies from a dataframe.

    Returns:
        encoder, organ_vocab, cell_vocab, subcell_vocab
    """
    organ_vocab = _build_vocab_from_df(df, organ_col)
    cell_vocab = _build_vocab_from_df(df, cell_col)
    sub_vocab = _build_vocab_from_df(df, subcell_col)

    encoder = HierEncoder(
        organ_vocab,
        cell_vocab,
        sub_vocab,
        meta_dim_hier=meta_dim_hier,
        device=device,
        o_hash_dim=o_hash_dim,
        c_hash_dim=c_hash_dim,
        s_hash_dim=s_hash_dim,
    )
    return encoder, organ_vocab, cell_vocab, sub_vocab

# ========== ② Species encoder (separate) ==========
class SpeciesEncoder(nn.Module):
    """
    Encode species information into a latent vector.
    Uses a multi-hot + hash-based representation similar to HierEncoder.
    """
    def __init__(
        self,
        species_vocab: Dict[str, int],
        species_dim: int = 32,
        normalize: bool = False,
        device: Optional[torch.device] = None,
        hash_dim: int = 256,
    ):
        super().__init__()
        self.species_vocab = species_vocab
        self.normalize = normalize

        self.hash_dim = int(hash_dim)

        self.encoder = nn.Linear(
            len(species_vocab) + hash_dim, species_dim, bias=True
        )
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        self.ln = nn.LayerNorm(species_dim)
        if device is not None:
            self.to(device)

    def forward(self, species: List[str]) -> torch.Tensor:
        """
        species: list of species strings (can also handle comma-separated multi-species labels).
        """
        device = next(self.parameters()).device
        S = torch.stack(
            [
                _multi_hot_hybrid(
                    _split_multi(x),
                    self.species_vocab,
                    device,
                    hash_dim=self.hash_dim,
                    normalize=self.normalize,
                )
                for x in species
            ],
            dim=0,
        )
        return self.ln(self.encoder(S))  # (B, species_dim)

def build_species_encoder_from_df(
    df: pd.DataFrame,
    species_col="species",
    species_dim=32,
    device: Optional[torch.device] = None,
    hash_dim: int = 256,
) -> Tuple[SpeciesEncoder, Dict[str, int]]:
    """
    Build a SpeciesEncoder and its vocabulary from a dataframe.

    Returns:
        encoder, species_vocab
    """
    species_vocab = _build_vocab_from_df(df, species_col)
    encoder = SpeciesEncoder(
        species_vocab,
        species_dim=species_dim,
        device=device,
        hash_dim=hash_dim,
    )
    return encoder, species_vocab

# ========== ③ Fusion with RNA-FM (before Attention) ==========
def get_rnafm_with_hier_species_before_attention(
    sequences: List[str],
    organs: List[str],
    cells: List[str],
    subcells: List[str],
    species: List[str],
    rnafm_model,
    batch_converter,
    device: torch.device,
    attention_layer,   # input dim = D_fm + meta_dim_hier + species_dim
    hier_encoder: HierEncoder,
    species_encoder: SpeciesEncoder,
    trainable: bool = False,
):
    """
    Compute token-level representations by fusing:
      - RNA-FM token embeddings
      - Hierarchical metadata embeddings
      - Species embeddings
    BEFORE the attention module, then perform MIL pooling via attention.

    Returns:
        token_concat      : (B, L, D_fm + D_hier + D_species)
        mil_embeddings    : (B, D_fm + D_hier + D_species)
        attention_weights : (B, L)
    """
    # 1) RNA-FM tokens
    batch_labels, batch_strs, batch_tokens = batch_converter(
        [(None, seq) for seq in sequences]
    )
    batch_tokens = batch_tokens.to(device)
    enc = _unwrap_module(rnafm_model)

    # Make sure parameters are set correctly for trainable / frozen mode
    for param in enc.parameters():
        param.requires_grad = True

    if trainable:
        results = enc(batch_tokens, repr_layers=[12])
    else:
        with torch.no_grad():
            results = enc(batch_tokens, repr_layers=[12])

    token_embeddings = results["representations"][12]  # (B, L, D_fm)

    # 2) Hierarchical and species metadata (encoded separately)
    meta_hier = hier_encoder(organs, cells, subcells)   # (B, meta_dim_hier)
    meta_species = species_encoder(species)             # (B, species_dim)

    # 3) Broadcast along sequence length and concatenate along feature dimension
    B, L, D_fm = token_embeddings.shape
    D_hier = meta_hier.shape[-1]
    D_sp = meta_species.shape[-1]

    hier_bc = meta_hier.unsqueeze(1).expand(B, L, D_hier)
    sp_bc = meta_species.unsqueeze(1).expand(B, L, D_sp)
    token_concat = torch.cat(
        [token_embeddings, hier_bc, sp_bc],
        dim=-1,
    )  # (B, L, D_fm + D_hier + D_sp)

    # 4) Attention + MIL pooling
    attention_weights = attention_layer(token_concat)  # (B, L)
    mil_embeddings = torch.bmm(
        attention_weights.unsqueeze(1), token_concat
    ).squeeze(1)  # (B, D_fm + D_hier + D_sp)

    return token_concat, mil_embeddings, attention_weights
