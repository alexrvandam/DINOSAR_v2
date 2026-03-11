"""
DINOSAR-v2: Contrastive Species Learning for Zero-Shot Classification
=========================================================================

This script implements contrastive learning on top of DINOv3 for progressive
species discovery and classification. Key features:

1. Uses DINOv3 backbone (frozen or fine-tunable) for feature extraction
2. Adds projection head for contrastive embedding space
3. Implements supervised contrastive learning with multi-view support
4. Progressive species discovery: A, B, C, D... (incremental clustering)
5. Zero-shot classification: classify new specimens against known species
6. TPS grid overlay capability to identify discriminative regions
7. Reuses v82 preprocessing: rembg, cropping, resizing
8. Multi-modal auxiliary learning (v24):
   a. Trait regression heads predict continuous measurements from vision features
   b. Meristic heads predict count data (Poisson NLL)
   c. Categorical heads predict binary traits (BCE)
   d. DNA encoder (1D CNN on COI barcodes) with species CE + cross-modal alignment
   e. Morphometric encoder (MLP on measurement vectors) with cross-modal alignment
   f. All auxiliary modalities are OPTIONAL — specimens missing data skip those losses

Architecture:
- DINOv3 Backbone → [CLS] token or mean-pooled patches
- Projection Head (MLP) → Contrastive embedding space (128-512D)
- Species Memory Bank: stores embeddings for known species
- Incremental Clustering: assigns new specimens to existing or new species
- TraitRegressionHead → continuous morphological traits from vision features
- MeristicHead → count-based traits from vision features
- CategoricalHead → binary presence/absence traits from vision features
- DNAEncoder (1D CNN) → DNA barcode embeddings → species CE + vision alignment
- MorphEncoder (MLP) → morphometric embeddings → vision alignment

Training Strategy:
- Positive pairs: different views/specimens of same species (or same specimen)
- Negative pairs: specimens from different species (or different specimens)
- Loss: SupCon + optional CE + trait MSE + meristic PoissonNLL + categorical BCE
        + DNA species CE + cross-modal alignment (vision↔DNA, vision↔morph)
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
from PIL import Image
import cv2
from sklearn.cluster import DBSCAN  # noqa: F401 (kept for compatibility)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from datetime import datetime
import yaml
import random
import colorsys
import csv
import math
import re as _re

# Import preprocessing from v82
try:
    from rembg import remove  # noqa: F401
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    print("Warning: rembg not available. Install with: pip install rembg")


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class SpeciesCluster:
    """Represents a discovered species cluster (for memory bank)"""
    species_id: str  # "A", "B", "C", etc. or real species name
    embeddings: List[np.ndarray]  # List of embeddings for this species
    specimen_ids: List[str]       # Specimen identifiers
    image_paths: List[str]        # Paths to images
    view_ids: List[str]           # View identifiers (e.g. H, D, P, unknown)
    centroid: Optional[np.ndarray] = None  # Mean embedding

    def update_centroid(self):
        """Compute mean centroid of all embeddings"""
        if self.embeddings:
            self.centroid = np.mean(self.embeddings, axis=0)

    def add_specimen(self, embedding: np.ndarray, specimen_id: str, image_path: str, view_id: str = "unknown"):
        """Add a new specimen (image) to this species cluster"""
        self.embeddings.append(embedding)
        self.specimen_ids.append(specimen_id)
        self.image_paths.append(image_path)
        self.view_ids.append(view_id)
        self.update_centroid()


class SpeciesMemoryBank:
    """Memory bank storing known species clusters"""

    def __init__(self, similarity_threshold: float = 0.85):
        self.species_clusters: Dict[str, SpeciesCluster] = {}
        self.similarity_threshold = similarity_threshold
        self.next_species_id = ord('A')  # Start with 'A'

    def get_next_species_id(self) -> str:
        """Generate next species ID (A, B, C, ... Z, AA, AB, ...)"""
        if self.next_species_id <= ord('Z'):
            species_id = chr(self.next_species_id)
            self.next_species_id += 1
            return species_id
        else:
            # Handle Z+: AA, AB, AC, etc.
            offset = self.next_species_id - ord('Z') - 1
            first_char = chr(ord('A') + (offset // 26))
            second_char = chr(ord('A') + (offset % 26))
            self.next_species_id += 1
            return f"{first_char}{second_char}"

    def classify_specimen(
        self,
        embedding: np.ndarray,
        specimen_id: str,
        image_path: str,
        return_scores: bool = False,
        view_id: str = "unknown",
    ) -> Tuple[str, float, bool]:
        """
        Classify a specimen: assign to existing species or create new one.

        Returns:
            (species_id, similarity_score, is_new_species)
        """
        if not self.species_clusters:
            # First specimen creates species A
            species_id = self.get_next_species_id()
            self.species_clusters[species_id] = SpeciesCluster(
                species_id=species_id,
                embeddings=[embedding],
                specimen_ids=[specimen_id],
                image_paths=[image_path],
                view_ids=[view_id],
                centroid=embedding,
            )
            return species_id, 1.0, True

        # Compare to all existing species centroids
        similarities = {}
        for sp_id, cluster in self.species_clusters.items():
            sim = cosine_similarity(
                embedding.reshape(1, -1),
                cluster.centroid.reshape(1, -1),
            )[0, 0]
            similarities[sp_id] = sim

        # Best match
        best_species = max(similarities, key=similarities.get)
        best_similarity = similarities[best_species]

        if return_scores:
            print(f"\n  Similarity scores for {specimen_id}:")
            for sp_id in sorted(similarities.keys()):
                print(f"    Species {sp_id}: {similarities[sp_id]:.4f}")

        if best_similarity >= self.similarity_threshold:
            # Assign to existing species
            self.species_clusters[best_species].add_specimen(
                embedding, specimen_id, image_path, view_id=view_id
            )
            return best_species, best_similarity, False
        else:
            # Create new species
            new_species_id = self.get_next_species_id()
            self.species_clusters[new_species_id] = SpeciesCluster(
                species_id=new_species_id,
                embeddings=[embedding],
                specimen_ids=[specimen_id],
                image_paths=[image_path],
                view_ids=[view_id],
                centroid=embedding,
            )
            return new_species_id, best_similarity, True

    def get_summary(self) -> Dict:
        """Get summary statistics of discovered species"""
        summary = {
            "total_species": len(self.species_clusters),
            "total_specimens": sum(len(c.specimen_ids) for c in self.species_clusters.values()),
            "species": {},
        }

        for sp_id, cluster in self.species_clusters.items():
            summary["species"][sp_id] = {
                "n_specimens": len(cluster.specimen_ids),
                "specimen_ids": cluster.specimen_ids,
                "image_paths": cluster.image_paths,
                "view_ids": cluster.view_ids,
            }

        return summary



# ============================================================================
# Model Architecture
# ============================================================================

class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning"""

    def __init__(self, input_dim: int, hidden_dim: int = 2048, output_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


# ============================================================================
# Multi-Modal Auxiliary Heads (v24)
# ============================================================================

class TraitRegressionHead(nn.Module):
    """
    Predict continuous morphological traits from vision backbone features.

    Inputs: (B, feature_dim) frozen or fine-tuned DINOv3 features
    Outputs: (B, n_traits) predicted continuous values

    The same head handles continuous, meristic, and categorical traits in
    separate output groups so a single forward pass produces all predictions.
    """

    def __init__(
        self,
        input_dim: int,
        n_continuous: int = 0,
        n_meristic: int = 0,
        n_categorical: int = 0,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_continuous = n_continuous
        self.n_meristic = n_meristic
        self.n_categorical = n_categorical
        self.n_total = n_continuous + n_meristic + n_categorical

        if self.n_total > 0:
            self.shared = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(inplace=True),
            )
            # Separate output layers for each trait type
            if n_continuous > 0:
                self.continuous_out = nn.Linear(hidden_dim // 2, n_continuous)
            if n_meristic > 0:
                # Outputs log-rate for Poisson NLL (must be positive after exp)
                self.meristic_out = nn.Linear(hidden_dim // 2, n_meristic)
            if n_categorical > 0:
                # Outputs logits for BCE
                self.categorical_out = nn.Linear(hidden_dim // 2, n_categorical)
        else:
            self.shared = None

    def forward(self, features: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        """
        Returns dict with keys 'continuous', 'meristic', 'categorical',
        each either a tensor or None.
        """
        out = {"continuous": None, "meristic": None, "categorical": None}
        if self.shared is None:
            return out

        h = self.shared(features)

        if self.n_continuous > 0:
            out["continuous"] = self.continuous_out(h)
        if self.n_meristic > 0:
            out["meristic"] = self.meristic_out(h)  # log-rate
        if self.n_categorical > 0:
            out["categorical"] = self.categorical_out(h)  # logits
        return out


class DNAEncoder(nn.Module):
    """
    1D CNN encoder for COI barcode sequences (~658bp).

    Encodes nucleotide sequences into an embedding space that can be:
      1) Classified into species (CE loss)
      2) Aligned with vision embeddings (cross-modal contrastive)

    Input: (B, max_seq_len) LongTensor of base indices:
           A=0, C=1, G=2, T=3, N/gap/other=4
    Output: (B, output_dim) L2-normalized embeddings
    """

    def __init__(self, output_dim: int = 256, max_seq_len: int = 700):
        super().__init__()
        self.output_dim = output_dim
        self.max_seq_len = max_seq_len

        self.embed = nn.Embedding(num_embeddings=5, embedding_dim=64, padding_idx=4)
        self.conv = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(256, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, seq_len) LongTensor → (B, output_dim) normalized embeddings"""
        h = self.embed(x)           # (B, seq_len, 64)
        h = h.transpose(1, 2)       # (B, 64, seq_len)
        h = self.conv(h).squeeze(-1) # (B, 256)
        h = self.fc(h)              # (B, output_dim)
        return F.normalize(h, dim=1)


class MorphEncoder(nn.Module):
    """
    MLP encoder for morphometric feature vectors (continuous measurements,
    meristic counts, categorical indicators — all stacked into one vector).

    Input: (B, morph_dim) float tensor of standardized features
    Output: (B, output_dim) L2-normalized embeddings
    """

    def __init__(self, input_dim: int, output_dim: int = 128, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1)


class CrossModalAlignmentLoss(nn.Module):
    """
    Symmetric cross-modal contrastive loss (NT-Xent / InfoNCE style).

    Aligns embeddings from two modalities (e.g., vision ↔ DNA) such that
    same-specimen pairs are pulled together and different-specimen pairs
    are pushed apart.

    This operates on *specimen-level* embeddings: if a specimen has both
    a vision embedding and a DNA embedding, they form a positive pair.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        """
        emb_a: (N, D) embeddings from modality A (e.g. vision)
        emb_b: (N, D) embeddings from modality B (e.g. DNA)
        Paired by index: emb_a[i] and emb_b[i] are the same specimen.

        Returns scalar loss.
        """
        N = emb_a.shape[0]
        if N < 2:
            return torch.tensor(0.0, device=emb_a.device)

        # Concatenate: [A0, A1, ..., B0, B1, ...]
        emb_all = torch.cat([emb_a, emb_b], dim=0)  # (2N, D)
        sim = torch.matmul(emb_all, emb_all.T) / self.temperature  # (2N, 2N)

        # Positive pairs: (i, i+N) and (i+N, i)
        labels = torch.cat([torch.arange(N) + N, torch.arange(N)], dim=0).to(emb_a.device)

        # Mask out self-similarity
        mask = torch.eye(2 * N, device=emb_a.device).bool()
        sim.masked_fill_(mask, -1e9)

        loss = F.cross_entropy(sim, labels)
        return loss


# ============================================================================
# Multi-Modal Data Loading Helpers (v24)
# ============================================================================

_BASE_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}


def _encode_dna_sequence(seq: str, max_len: int = 700) -> np.ndarray:
    """Encode a DNA sequence string to integer array for DNAEncoder.

    A=0, C=1, G=2, T=3, anything else (N, -, gap, IUPAC ambiguity)=4.
    Pads or truncates to max_len.
    """
    arr = np.full(max_len, 4, dtype=np.int64)  # default = padding/unknown
    seq = seq.upper()
    for i, base in enumerate(seq[:max_len]):
        arr[i] = _BASE_MAP.get(base, 4)
    return arr


def load_trait_data(
    trait_file: str,
    specimen_id_col: str = "specimen_id",
) -> Tuple[Dict[str, Dict[str, float]], List[str], List[str], List[str]]:
    """
    Load morphological trait data from a TSV file.

    Returns:
        traits_by_specimen: {specimen_id: {trait_name: value}}
        continuous_names:   list of continuous trait column names
        meristic_names:     list of meristic trait column names (integer counts)
        categorical_names:  list of categorical trait column names (binary 0/1)

    Column naming convention for auto-detection:
      - Columns starting with 'cat_' or 'has_' → categorical
      - Columns starting with 'count_' or 'n_' → meristic
      - Everything else (numeric) → continuous
    """
    print(f"\n  Loading trait data from: {trait_file}")
    traits_by_specimen: Dict[str, Dict[str, float]] = {}
    continuous_names: List[str] = []
    meristic_names: List[str] = []
    categorical_names: List[str] = []

    with open(trait_file, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    if not rows:
        print("  ⚠ No trait data found!")
        return traits_by_specimen, continuous_names, meristic_names, categorical_names

    # Identify trait columns (everything except specimen_id and species_id)
    skip_cols = {specimen_id_col, "species_id", "species", "image_path", "view_id"}
    all_cols = [c for c in rows[0].keys() if c not in skip_cols]

    # Categorize columns
    for col in all_cols:
        col_lower = col.lower()
        if col_lower.startswith("cat_") or col_lower.startswith("has_") or col_lower.startswith("is_"):
            categorical_names.append(col)
        elif col_lower.startswith("count_") or col_lower.startswith("n_") or col_lower.startswith("num_"):
            meristic_names.append(col)
        else:
            # Check if values are all integers (likely meristic)
            sample_vals = []
            for r in rows[:50]:
                try:
                    v = float(r.get(col, ""))
                    sample_vals.append(v)
                except (ValueError, TypeError):
                    pass
            if sample_vals:
                all_int = all(v == int(v) for v in sample_vals)
                all_binary = all(v in (0.0, 1.0) for v in sample_vals)
                if all_binary and len(set(sample_vals)) <= 2:
                    categorical_names.append(col)
                elif all_int and min(sample_vals) >= 0 and max(sample_vals) < 100:
                    meristic_names.append(col)
                else:
                    continuous_names.append(col)
            else:
                continuous_names.append(col)

    # Parse values
    for row in rows:
        sid = row.get(specimen_id_col, "").strip()
        if not sid:
            continue
        trait_dict: Dict[str, float] = {}
        for col in continuous_names + meristic_names + categorical_names:
            try:
                trait_dict[col] = float(row.get(col, ""))
            except (ValueError, TypeError):
                trait_dict[col] = float("nan")
        traits_by_specimen[sid] = trait_dict

    print(f"  ✓ Loaded traits for {len(traits_by_specimen)} specimens")
    print(f"    Continuous: {len(continuous_names)} ({', '.join(continuous_names[:5])}{'...' if len(continuous_names) > 5 else ''})")
    print(f"    Meristic:   {len(meristic_names)} ({', '.join(meristic_names[:5])}{'...' if len(meristic_names) > 5 else ''})")
    print(f"    Categorical:{len(categorical_names)} ({', '.join(categorical_names[:5])}{'...' if len(categorical_names) > 5 else ''})")

    return traits_by_specimen, continuous_names, meristic_names, categorical_names


def load_dna_data(
    fasta_file: str,
    max_seq_len: int = 700,
) -> Dict[str, np.ndarray]:
    """
    Load COI barcode sequences from FASTA and encode as integer arrays.

    The FASTA header first token (before '|' or whitespace) is treated as
    specimen_id for linking to image data.

    Returns:
        dna_by_specimen: {specimen_id: np.ndarray of shape (max_seq_len,)}
    """
    print(f"\n  Loading DNA data from: {fasta_file}")
    dna_by_specimen: Dict[str, np.ndarray] = {}

    cur_id = None
    cur_seq: List[str] = []

    with open(fasta_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    seq_str = "".join(cur_seq)
                    dna_by_specimen[cur_id] = _encode_dna_sequence(seq_str, max_seq_len)
                # Parse specimen_id from header: first token before | or space
                header = line[1:]
                cur_id = header.split("|")[0].split()[0].strip()
                cur_seq = []
            else:
                cur_seq.append(line)

    if cur_id is not None:
        seq_str = "".join(cur_seq)
        dna_by_specimen[cur_id] = _encode_dna_sequence(seq_str, max_seq_len)

    print(f"  ✓ Loaded DNA for {len(dna_by_specimen)} specimens")
    # Show sequence length stats
    if dna_by_specimen:
        lens = [int((arr != 4).sum()) for arr in dna_by_specimen.values()]
        print(f"    Effective lengths: min={min(lens)}, median={sorted(lens)[len(lens)//2]}, max={max(lens)}")

    return dna_by_specimen


def load_morph_features(
    morph_file: str,
    specimen_id_col: str = "specimen_id",
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Load morphometric feature vectors from TSV.

    Returns:
        morph_by_specimen: {specimen_id: np.ndarray of shape (n_features,)}
        feature_names: list of feature column names
    """
    print(f"\n  Loading morphometric features from: {morph_file}")
    morph_by_specimen: Dict[str, np.ndarray] = {}

    with open(morph_file, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    if not rows:
        return morph_by_specimen, []

    skip_cols = {specimen_id_col, "species_id", "species", "image_path", "view_id"}
    feature_names = [c for c in rows[0].keys() if c not in skip_cols]

    for row in rows:
        sid = row.get(specimen_id_col, "").strip()
        if not sid:
            continue
        vals = []
        for col in feature_names:
            try:
                vals.append(float(row.get(col, "")))
            except (ValueError, TypeError):
                vals.append(0.0)
        morph_by_specimen[sid] = np.array(vals, dtype=np.float32)

    # Standardize across all specimens
    if morph_by_specimen:
        all_vecs = np.stack(list(morph_by_specimen.values()), axis=0)
        means = np.nanmean(all_vecs, axis=0)
        stds = np.nanstd(all_vecs, axis=0) + 1e-8
        for sid in morph_by_specimen:
            morph_by_specimen[sid] = ((morph_by_specimen[sid] - means) / stds).astype(np.float32)
            morph_by_specimen[sid] = np.nan_to_num(morph_by_specimen[sid], nan=0.0)

    print(f"  ✓ Loaded {len(morph_by_specimen)} specimens × {len(feature_names)} features")
    return morph_by_specimen, feature_names


def compute_trait_standardization(
    traits_by_specimen: Dict[str, Dict[str, float]],
    trait_names: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and std for trait standardization (continuous traits only)."""
    vals = []
    for sid, td in traits_by_specimen.items():
        row = [td.get(t, float("nan")) for t in trait_names]
        vals.append(row)
    arr = np.array(vals, dtype=np.float64)
    means = np.nanmean(arr, axis=0)
    stds = np.nanstd(arr, axis=0) + 1e-8
    return means.astype(np.float32), stds.astype(np.float32)


def build_trait_tensors_for_batch(
    specimen_ids: List[str],
    traits_by_specimen: Dict[str, Dict[str, float]],
    continuous_names: List[str],
    meristic_names: List[str],
    categorical_names: List[str],
    cont_means: Optional[np.ndarray],
    cont_stds: Optional[np.ndarray],
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
    """
    For a batch of specimen_ids, build target tensors for trait prediction.

    Returns:
        cont_targets:  (M, n_continuous) or None
        meris_targets: (M, n_meristic) or None
        cat_targets:   (M, n_categorical) or None
        valid_mask:    (B,) bool tensor — True for specimens with trait data
    """
    B = len(specimen_ids)
    valid_mask = torch.zeros(B, dtype=torch.bool, device=device)

    cont_list = []
    meris_list = []
    cat_list = []

    for i, sid in enumerate(specimen_ids):
        sid = str(sid)
        if sid not in traits_by_specimen:
            continue
        td = traits_by_specimen[sid]

        # Check we have at least some valid values
        has_any = False

        if continuous_names:
            vals = [td.get(c, float("nan")) for c in continuous_names]
            if any(np.isfinite(v) for v in vals):
                has_any = True

        if meristic_names:
            vals_m = [td.get(c, float("nan")) for c in meristic_names]
            if any(np.isfinite(v) for v in vals_m):
                has_any = True

        if categorical_names:
            vals_c = [td.get(c, float("nan")) for c in categorical_names]
            if any(np.isfinite(v) for v in vals_c):
                has_any = True

        if has_any:
            valid_mask[i] = True

    valid_indices = valid_mask.nonzero(as_tuple=True)[0]
    M = int(valid_indices.shape[0])

    if M == 0:
        return None, None, None, valid_mask

    if continuous_names:
        arr = np.zeros((M, len(continuous_names)), dtype=np.float32)
        for j, idx in enumerate(valid_indices.cpu().tolist()):
            sid = str(specimen_ids[idx])
            td = traits_by_specimen[sid]
            for k, c in enumerate(continuous_names):
                v = td.get(c, float("nan"))
                if np.isfinite(v) and cont_means is not None:
                    arr[j, k] = (v - cont_means[k]) / cont_stds[k]
                else:
                    arr[j, k] = 0.0  # will be masked
        cont_list = torch.tensor(arr, dtype=torch.float32, device=device)
    else:
        cont_list = None

    if meristic_names:
        arr = np.zeros((M, len(meristic_names)), dtype=np.float32)
        for j, idx in enumerate(valid_indices.cpu().tolist()):
            sid = str(specimen_ids[idx])
            td = traits_by_specimen[sid]
            for k, c in enumerate(meristic_names):
                v = td.get(c, float("nan"))
                arr[j, k] = max(0.0, v) if np.isfinite(v) else 0.0
        meris_list = torch.tensor(arr, dtype=torch.float32, device=device)
    else:
        meris_list = None

    if categorical_names:
        arr = np.zeros((M, len(categorical_names)), dtype=np.float32)
        for j, idx in enumerate(valid_indices.cpu().tolist()):
            sid = str(specimen_ids[idx])
            td = traits_by_specimen[sid]
            for k, c in enumerate(categorical_names):
                v = td.get(c, float("nan"))
                arr[j, k] = float(v) if np.isfinite(v) else 0.0
        cat_list = torch.tensor(arr, dtype=torch.float32, device=device)
    else:
        cat_list = None

    return cont_list, meris_list, cat_list, valid_mask


def build_dna_tensor_for_batch(
    specimen_ids: List[str],
    dna_by_specimen: Dict[str, np.ndarray],
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """
    For a batch of specimen_ids, build DNA input tensor.

    Returns:
        dna_input: (M, max_seq_len) LongTensor or None
        valid_mask: (B,) bool tensor — True for specimens with DNA data
    """
    B = len(specimen_ids)
    valid_mask = torch.zeros(B, dtype=torch.bool, device=device)
    matched = []

    for i, sid in enumerate(specimen_ids):
        sid = str(sid)
        if sid in dna_by_specimen:
            valid_mask[i] = True
            matched.append(dna_by_specimen[sid])

    if not matched:
        return None, valid_mask

    dna_arr = np.stack(matched, axis=0)
    dna_input = torch.tensor(dna_arr, dtype=torch.long, device=device)
    return dna_input, valid_mask


def build_morph_tensor_for_batch(
    specimen_ids: List[str],
    morph_by_specimen: Dict[str, np.ndarray],
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """
    For a batch of specimen_ids, build morphometric input tensor.

    Returns:
        morph_input: (M, morph_dim) FloatTensor or None
        valid_mask:  (B,) bool tensor
    """
    B = len(specimen_ids)
    valid_mask = torch.zeros(B, dtype=torch.bool, device=device)
    matched = []

    for i, sid in enumerate(specimen_ids):
        sid = str(sid)
        if sid in morph_by_specimen:
            valid_mask[i] = True
            matched.append(morph_by_specimen[sid])

    if not matched:
        return None, valid_mask

    morph_arr = np.stack(matched, axis=0)
    morph_input = torch.tensor(morph_arr, dtype=torch.float32, device=device)
    return morph_input, valid_mask


class DINOSAR_v2(nn.Module):
    """
    DINOSAR-v2: DINOv3 + Contrastive Learning for Species Classification
    """

    def _unfreeze_last_transformer_blocks(self, n: int) -> int:
        blocks = getattr(self.backbone, "blocks", None)
        if blocks is None:
            print("[WARN] Backbone has no attribute 'blocks' — cannot unfreeze last blocks.")
            return 0
        n = int(max(0, min(n, len(blocks))))
        for blk in blocks[-n:]:
            for p in blk.parameters():
                p.requires_grad = True
        return n

    def get_trainable_backbone_params(self):
        """Return ONLY backbone params that are currently trainable (requires_grad=True)."""
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def __init__(
        self,
        dinov3_model: str = "dinov3_vitb14",
        dinov3_checkpoint: Optional[str] = None,
        projection_dim: int = 256,
        freeze_backbone: bool = True,
        unfreeze_last_n_blocks: int = 0,        # ✅ ADD THIS
        use_cls_token: bool = True,
        use_classifier_head: bool = False,
        num_species: Optional[int] = None,
    ):
        super().__init__()

        self.backbone = self._load_dinov3_backbone(dinov3_model, dinov3_checkpoint)
        self.use_cls_token = use_cls_token
        self.unfreeze_last_n_blocks = int(unfreeze_last_n_blocks)

        # ---- Freeze / unfreeze backbone ----
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

            unfrozen = 0
            if self.unfreeze_last_n_blocks > 0:
                unfrozen = self._unfreeze_last_transformer_blocks(self.unfreeze_last_n_blocks)

            if unfrozen > 0:
                print(f"✓ Backbone frozen, but unfroze last {unfrozen} transformer blocks")
                self.backbone.train()   # important if you want those blocks to learn
            else:
                print("✓ DINOv3 backbone frozen")
                self.backbone.eval()    # good default when fully frozen (more stable)
        else:
            for p in self.backbone.parameters():
                p.requires_grad = True
            print("✓ DINOv3 backbone trainable")
            self.backbone.train()

        self.feature_dim = self.backbone.embed_dim

        # Projection head
        self.projection_head = ProjectionHead(
            input_dim=self.feature_dim,
            hidden_dim=2048,
            output_dim=projection_dim,
        )

        # Optional classifier head
        self.use_classifier_head = use_classifier_head
        if self.use_classifier_head:
            if num_species is None:
                raise ValueError("use_classifier_head=True requires num_species.")
            self.classifier_head = nn.Linear(projection_dim, num_species)
            print(f"✓ Classifier head added: {num_species} species")
        else:
            self.classifier_head = None

        # ── Multi-modal auxiliary heads (v24) ──────────────────────────
        self.trait_head: Optional[TraitRegressionHead] = None
        self.dna_encoder: Optional[DNAEncoder] = None
        self.dna_classifier: Optional[nn.Linear] = None
        self.morph_encoder: Optional[MorphEncoder] = None
        self.xmodal_loss_fn: Optional[CrossModalAlignmentLoss] = None

        print(f"✓ Model initialized: {dinov3_model}")
        print(f"  Feature dim: {self.feature_dim}, Projection dim: {projection_dim}")

    def init_trait_head(
        self,
        n_continuous: int = 0,
        n_meristic: int = 0,
        n_categorical: int = 0,
    ):
        """Initialize trait regression head (call after model construction)."""
        n_total = n_continuous + n_meristic + n_categorical
        if n_total > 0:
            self.trait_head = TraitRegressionHead(
                input_dim=self.feature_dim,  # operates on backbone features, not projections
                n_continuous=n_continuous,
                n_meristic=n_meristic,
                n_categorical=n_categorical,
            )
            print(f"✓ Trait regression head added: {n_continuous} continuous, "
                  f"{n_meristic} meristic, {n_categorical} categorical")

    def init_dna_encoder(self, output_dim: int = 256, max_seq_len: int = 700, num_species: int = 0):
        """Initialize DNA encoder and optional DNA species classifier."""
        self.dna_encoder = DNAEncoder(output_dim=output_dim, max_seq_len=max_seq_len)
        if num_species > 0:
            self.dna_classifier = nn.Linear(output_dim, num_species)
            print(f"✓ DNA encoder added: seq→{output_dim}D + {num_species}-species classifier")
        else:
            print(f"✓ DNA encoder added: seq→{output_dim}D (no classifier)")

    def init_morph_encoder(self, morph_dim: int, output_dim: int = 128):
        """Initialize morphometric feature encoder."""
        self.morph_encoder = MorphEncoder(input_dim=morph_dim, output_dim=output_dim)
        print(f"✓ Morph encoder added: {morph_dim}→{output_dim}D")

    def init_cross_modal_loss(self, temperature: float = 0.1):
        """Initialize cross-modal alignment loss."""
        self.xmodal_loss_fn = CrossModalAlignmentLoss(temperature=temperature)
        print(f"✓ Cross-modal alignment loss (τ={temperature})")

    def _load_dinov3_backbone(self, model_name: str, checkpoint_path: Optional[str]):
        """
        Load DINOv3 model *locally* (no torch.hub, no downloads).
        """
        import sys

        try:
            try:
                from dinov3.models.vision_transformer import vit_small, vit_base, vit_large
            except ImportError:
                sys.path.insert(0, os.path.expanduser("~/models/git/dinov3"))
                from dinov3.models.vision_transformer import vit_small, vit_base, vit_large
        except Exception as e:
            try:
                from models.vision_transformer import vit_small, vit_base, vit_large
            except Exception as e2:
                raise RuntimeError(
                    "Could not import DINOv3 vision_transformer.\n"
                    "Make sure the DINOv3 repo is available either as a package "
                    "('dinov3') or at '~/models/git/dinov3', or that "
                    "'models/vision_transformer.py' is on PYTHONPATH.\n"
                    f"First error:  {e}\nSecond error: {e2}"
                ) from e2

        name = (model_name or "").lower()
        patch_size = None

        if name in ("dinov3_vits16",):
            patch_size = 16
            model_fn = lambda: vit_small(patch_size=patch_size)
        elif name in ("dinov3_vits14",):
            patch_size = 14
            model_fn = lambda: vit_small(patch_size=patch_size)
        elif name in ("dinov3_vitb14",):
            patch_size = 14
            model_fn = lambda: vit_base(patch_size=patch_size)
        elif name in ("dinov3_vitl14",):
            patch_size = 14
            model_fn = lambda: vit_large(patch_size=patch_size)
        else:
            raise ValueError(
                f"Unknown DINOv3 model name '{model_name}'. "
                "Expected one of: dinov3_vits14, dinov3_vits16, "
                "dinov3_vitb14, dinov3_vitl14."
            )

        print(f"Building local DINOv3 backbone: {model_name}, patch_size={patch_size}")
        model = model_fn()

        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"DINOv3 checkpoint not found at: {checkpoint_path}\n"
                "Please provide a valid --dinov3-local-ckpt path."
            )

        print(f"Loading DINOv3 weights from local checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "teacher" in checkpoint:
                state_dict = checkpoint["teacher"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        msg = model.load_state_dict(state_dict, strict=False)
        missing_keys = getattr(msg, "missing_keys", [])
        unexpected_keys = getattr(msg, "unexpected_keys", [])

        if missing_keys:
            print(f"  Note: missing keys in checkpoint: {len(missing_keys)}")
        if unexpected_keys:
            print(f"  Note: unexpected keys in checkpoint: {len(unexpected_keys)}")

        print("✓ Local DINOv3 checkpoint loaded successfully (no torch.hub)")

        return model

    def forward(
        self,
        x,
        return_features: bool = False,
        return_logits: bool = False,
    ):
        """
        Forward pass.

        Returns:
            - embeddings
            - optionally features, logits depending on flags
        """
        with torch.set_grad_enabled(self.training):
            output = self.backbone.forward_features(x)

            if self.use_cls_token:
                features = output["x_norm_clstoken"]
            else:
                features = output["x_norm_patchtokens"].mean(dim=1)

        embeddings = self.projection_head(features)

        logits = None
        if self.use_classifier_head and return_logits:
            logits = self.classifier_head(embeddings)

        if return_features and return_logits:
            return embeddings, features, logits
        if return_logits:
            return embeddings, logits
        if return_features:
            return embeddings, features
        return embeddings

    def get_embeddings(self, x):
        """Get embeddings for inference (no gradients, no classifier head)."""
        self.eval()
        with torch.no_grad():
            embeddings = self.forward(x, return_features=False, return_logits=False)
        return embeddings

    def predict_traits(self, features: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        """Predict morphological traits from backbone features.

        Args:
            features: (B, feature_dim) backbone features (NOT projection embeddings)

        Returns:
            dict with 'continuous', 'meristic', 'categorical' tensors or None
        """
        if self.trait_head is None:
            return {"continuous": None, "meristic": None, "categorical": None}
        return self.trait_head(features)

    def encode_dna(self, dna_input: torch.Tensor) -> torch.Tensor:
        """Encode DNA sequences.

        Args:
            dna_input: (B, max_seq_len) LongTensor of base indices

        Returns:
            (B, dna_output_dim) normalized embeddings
        """
        if self.dna_encoder is None:
            raise RuntimeError("DNA encoder not initialized. Call init_dna_encoder() first.")
        return self.dna_encoder(dna_input)

    def encode_morph(self, morph_input: torch.Tensor) -> torch.Tensor:
        """Encode morphometric feature vectors.

        Args:
            morph_input: (B, morph_dim) float tensor

        Returns:
            (B, morph_output_dim) normalized embeddings
        """
        if self.morph_encoder is None:
            raise RuntimeError("Morph encoder not initialized. Call init_morph_encoder() first.")
        return self.morph_encoder(morph_input)

    def compute_auxiliary_losses(
        self,
        features: torch.Tensor,
        embeddings: torch.Tensor,
        species_labels: torch.Tensor,
        # Trait data (looked up per batch)
        trait_valid_mask: Optional[torch.Tensor] = None,
        cont_targets: Optional[torch.Tensor] = None,
        meris_targets: Optional[torch.Tensor] = None,
        cat_targets: Optional[torch.Tensor] = None,
        # DNA data
        dna_input: Optional[torch.Tensor] = None,
        dna_valid_mask: Optional[torch.Tensor] = None,
        dna_species_labels: Optional[torch.Tensor] = None,
        # Morph data
        morph_input: Optional[torch.Tensor] = None,
        morph_valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all auxiliary losses for multi-modal training.

        Returns dict of named losses (each a scalar tensor); caller sums with weights.
        Only losses with available data are returned.
        """
        device = features.device
        losses: Dict[str, torch.Tensor] = {}

        # ── 1. Trait prediction losses (from vision backbone features) ──
        if self.trait_head is not None and trait_valid_mask is not None and trait_valid_mask.any():
            valid_idx = trait_valid_mask.nonzero(as_tuple=True)[0]
            valid_features = features[valid_idx]
            preds = self.trait_head(valid_features)

            if preds["continuous"] is not None and cont_targets is not None:
                # MSE loss for continuous traits (NaN-safe: already handled in build_trait_tensors)
                losses["trait_continuous"] = F.mse_loss(preds["continuous"], cont_targets)

            if preds["meristic"] is not None and meris_targets is not None:
                # Poisson NLL: model predicts log-rate, target is count
                log_rate = preds["meristic"]
                target_counts = meris_targets
                # F.poisson_nll_loss expects (log_input, target, log_input=True)
                losses["trait_meristic"] = F.poisson_nll_loss(
                    log_rate, target_counts, log_input=True, full=False, reduction="mean"
                )

            if preds["categorical"] is not None and cat_targets is not None:
                losses["trait_categorical"] = F.binary_cross_entropy_with_logits(
                    preds["categorical"], cat_targets, reduction="mean"
                )

        # ── 2. DNA losses (encoder + species CE + cross-modal alignment) ──
        if self.dna_encoder is not None and dna_input is not None and dna_valid_mask is not None and dna_valid_mask.any():
            dna_emb = self.dna_encoder(dna_input)  # (M, dna_dim)

            # DNA species classification
            if self.dna_classifier is not None and dna_species_labels is not None:
                dna_logits = self.dna_classifier(dna_emb)
                losses["dna_ce"] = F.cross_entropy(dna_logits, dna_species_labels)

            # Cross-modal alignment: vision ↔ DNA
            if self.xmodal_loss_fn is not None:
                valid_idx = dna_valid_mask.nonzero(as_tuple=True)[0]
                vision_emb_matched = embeddings[valid_idx]  # (M, proj_dim)
                # Project DNA to same dim as vision if needed
                if dna_emb.shape[1] != vision_emb_matched.shape[1]:
                    # Use a simple linear projection (lazy init)
                    if not hasattr(self, "_dna_vision_proj"):
                        self._dna_vision_proj = nn.Linear(
                            dna_emb.shape[1], vision_emb_matched.shape[1]
                        ).to(device)
                    dna_emb_proj = F.normalize(self._dna_vision_proj(dna_emb), dim=1)
                else:
                    dna_emb_proj = dna_emb
                losses["xmodal_vision_dna"] = self.xmodal_loss_fn(vision_emb_matched, dna_emb_proj)

        # ── 3. Morph losses (encoder + cross-modal alignment) ──
        if self.morph_encoder is not None and morph_input is not None and morph_valid_mask is not None and morph_valid_mask.any():
            morph_emb = self.morph_encoder(morph_input)  # (M, morph_dim)

            if self.xmodal_loss_fn is not None:
                valid_idx = morph_valid_mask.nonzero(as_tuple=True)[0]
                vision_emb_matched = embeddings[valid_idx]
                if morph_emb.shape[1] != vision_emb_matched.shape[1]:
                    if not hasattr(self, "_morph_vision_proj"):
                        self._morph_vision_proj = nn.Linear(
                            morph_emb.shape[1], vision_emb_matched.shape[1]
                        ).to(device)
                    morph_emb_proj = F.normalize(self._morph_vision_proj(morph_emb), dim=1)
                else:
                    morph_emb_proj = morph_emb
                losses["xmodal_vision_morph"] = self.xmodal_loss_fn(vision_emb_matched, morph_emb_proj)

        return losses


# ============================================================================
# Contrastive Loss Functions
# ============================================================================

class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss (SupCon)

    Paper: Supervised Contrastive Learning (Khosla et al., NeurIPS 2020)
    """

    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        """
        Args:
            features: [batch_size, embedding_dim] normalized embeddings
            labels: [batch_size] labels (species or specimen)

        Returns:
            loss: scalar
        """
        device = features.device
        batch_size = features.shape[0]

        # Similarity matrix
        similarity_matrix = torch.matmul(features, features.T)

        # Mask for positive pairs
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Remove diagonal
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size, device=device).view(-1, 1),
            0,
        )
        mask = mask * logits_mask

        logits = similarity_matrix / self.temperature
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1)

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        return loss


class TripletLoss(nn.Module):
    """
    Triplet Loss with hard negative mining
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: [batch_size, embedding_dim] normalized embeddings
            labels: [batch_size] labels (species or specimen)

        Returns:
            loss: scalar
        """
        pairwise_dist = torch.cdist(embeddings, embeddings, p=2)

        losses = []
        for i in range(len(embeddings)):
            anchor_label = labels[i]

            pos_mask = (labels == anchor_label) & (
                torch.arange(len(labels), device=embeddings.device) != i
            )
            if pos_mask.sum() == 0:
                continue

            pos_dist = pairwise_dist[i][pos_mask]
            hardest_pos_dist = pos_dist.max()

            neg_mask = labels != anchor_label
            if neg_mask.sum() == 0:
                continue

            neg_dist = pairwise_dist[i][neg_mask]
            hardest_neg_dist = neg_dist.min()

            loss = F.relu(hardest_pos_dist - hardest_neg_dist + self.margin)
            losses.append(loss)

        if losses:
            return torch.stack(losses).mean()
        return torch.tensor(0.0, device=embeddings.device)


# ============================================================================
# Dataset
# ============================================================================

class SpecimenDataset(Dataset):
    """
    Dataset for specimen images with species labels.

    Each sample is stored as:
        (image_path, species_idx, specimen_id, view_id)

    __getitem__ returns:
        With masks:
            (img_tensor, species_idx, specimen_id, view_id, img_path, mask_tensor)
        Without masks:
            (img_tensor, species_idx, specimen_id, view_id, img_path)
    """

    def __init__(
        self,
        data_dir: str,
        transform=None,
        remove_bg: bool = True,
        crop_to_fg: bool = True,
        target_size: int = 518,
        mask_erode_px: int = 0,
        return_masks: bool = False,
        metadata_file: Optional[str] = None,
        view_filter: Optional[List[str]] = None,
        coco_mask_file: Optional[str] = None,
        coco_mask_category: str = "foreground",
        build_immediately: bool = True,
    ):
        """
        Args:
            data_dir: Root directory containing images
            transform: torchvision transforms
            remove_bg: Use rembg to remove background
            crop_to_fg: Crop to foreground bounding box
            target_size: Target image size (square canvas)
            mask_erode_px: Erode mask (to remove halos)
            return_masks: If True, return masks
            metadata_file: CSV/TSV/JSON with image/species/specimen/view
            view_filter: Optional list of view IDs to include (e.g. ["H","D"])
            build_immediately: If False, skip scanning files (for preprocessing-only usage)
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.remove_bg = remove_bg and REMBG_AVAILABLE
        self.crop_to_fg = crop_to_fg
        self.target_size = target_size
        self.mask_erode_px = mask_erode_px
        self.return_masks = return_masks
        self.metadata_file = metadata_file
        self.view_filter = set(view_filter) if view_filter is not None else None

        self.coco_mask_file = coco_mask_file
        self.coco_mask_category = coco_mask_category
        self.coco_seg_by_file: Dict[str, List[List[float]]] = {}

        if self.coco_mask_file is not None:
            self._load_coco_masks()


        if self.remove_bg and not REMBG_AVAILABLE:
            print("Warning: rembg not available, background removal disabled")
            self.remove_bg = False

        self.samples: List[Tuple[str, int, str, str]] = []  # (img_path, species_idx, specimen_id, view_id)
        self.species_to_idx: Dict[str, int] = {}
        self.idx_to_species: Dict[int, str] = {}
        self.specimen_to_idx: Dict[str, int] = {}
        self.idx_to_specimen: Dict[int, str] = {}

        if build_immediately:
            if self.metadata_file:
                self._build_dataset_from_metadata()
            else:
                self._build_dataset_from_directory()

            # Apply view filter if requested
            if self.view_filter is not None:
                before = len(self.samples)
                self.samples = [s for s in self.samples if s[3] in self.view_filter]
                after = len(self.samples)
                print(f"✓ View filter applied: {self.view_filter} (kept {after}/{before} samples)")

            self._build_specimen_index()

    # ------------------------------------------------------------------ #
    # Dataset builders
    # ------------------------------------------------------------------ #

    def _build_dataset_from_directory(self):
        """Build dataset from directory structure"""
        species_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])

        for idx, species_dir in enumerate(species_dirs):
            species_name = species_dir.name
            self.species_to_idx[species_name] = idx
            self.idx_to_species[idx] = species_name

            image_files: List[Path] = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
                image_files.extend(species_dir.glob(ext))

            for img_path in sorted(image_files):
                stem = img_path.stem
                if "_" in stem:
                    specimen_id = stem.rsplit("_", 1)[0]
                    view_id = stem.rsplit("_", 1)[1]
                else:
                    specimen_id = stem
                    view_id = "unknown"

                self.samples.append((str(img_path), idx, specimen_id, view_id))

        print(f"✓ Dataset built from directory: {len(self.samples)} images from {len(self.species_to_idx)} species")
        for sp_name, sp_idx in sorted(self.species_to_idx.items()):
            n_images = sum(1 for _, idx, _, _ in self.samples if idx == sp_idx)
            print(f"  {sp_name}: {n_images} images")

    def _build_dataset_from_metadata(self):
        """
        Build dataset from metadata file (CSV, TSV, or JSON)

        Required fields: image_path, species_id
        Optional: specimen_id, view_id
        """
        metadata_path = Path(self.metadata_file)
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_file}")

        print(f"Loading metadata from: {self.metadata_file}")
        suffix = metadata_path.suffix.lower()

        if suffix == ".json":
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            if isinstance(metadata, dict):
                metadata = [metadata]
        elif suffix in [".csv", ".tsv"]:
            import csv
            delimiter = "\t" if suffix == ".tsv" else ","
            with open(metadata_path, "r", newline="") as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                metadata = list(reader)
        else:
            raise ValueError(f"Unsupported metadata file format: {suffix}. Use .csv, .tsv, or .json")

        unique_species = sorted({row["species_id"] for row in metadata})
        self.species_to_idx = {sp: idx for idx, sp in enumerate(unique_species)}
        self.idx_to_species = {idx: sp for sp, idx in self.species_to_idx.items()}

        for row in metadata:
            img_path = Path(row["image_path"])
            if not img_path.is_absolute():
                img_path = self.data_dir / img_path

            if not img_path.exists():
                print(f"Warning: Image not found: {img_path}")
                continue

            species_id = row["species_id"]
            if species_id not in self.species_to_idx:
                print(f"Warning: species_id '{species_id}' not in mapping; skipping {img_path}")
                continue
            species_idx = self.species_to_idx[species_id]

            specimen_id = row.get("specimen_id") or img_path.stem
            view_id = row.get("view_id") or "unknown"

            self.samples.append((str(img_path), species_idx, specimen_id, view_id))

        print(f"✓ Dataset built from metadata file: {len(self.samples)} images from {len(self.species_to_idx)} species")
        for sp_name, sp_idx in sorted(self.species_to_idx.items()):
            n_images = sum(1 for _, idx, _, _ in self.samples if idx == sp_idx)
            print(f"  {sp_name}: {n_images} images")

    def _build_specimen_index(self):
        """Build mapping specimen_id → integer index."""
        unique_specimen_ids = sorted({specimen_id for _, _, specimen_id, _ in self.samples})
        self.specimen_to_idx = {sid: i for i, sid in enumerate(unique_specimen_ids)}
        self.idx_to_specimen = {i: sid for sid, i in self.specimen_to_idx.items()}
        print(f"✓ Found {len(unique_specimen_ids)} unique specimens in dataset.")

    # ------------------------------------------------------------------ #
    # Preprocessing
    # ------------------------------------------------------------------ #
    def _load_coco_masks(self):
        """
        Load a COCO-style mask file and build a mapping:
            file_name -> [segmentation_polygons]
        Only masks with category == self.coco_mask_category are used.
        """
        coco_path = Path(self.coco_mask_file)
        if not coco_path.exists():
            raise FileNotFoundError(f"COCO mask file not found: {coco_path}")

        print(f"Loading COCO masks from: {coco_path}")
        with coco_path.open("r") as f:
            coco = json.load(f)

        # Map category_id -> name, and name -> id
        cat_id_to_name = {c["id"]: c["name"] for c in coco.get("categories", [])}
        name_to_cat_id = {v: k for k, v in cat_id_to_name.items()}

        if self.coco_mask_category not in name_to_cat_id:
            raise ValueError(
                f"Category '{self.coco_mask_category}' not found in COCO categories: "
                f"{sorted(name_to_cat_id.keys())}"
            )
        target_cat_id = name_to_cat_id[self.coco_mask_category]

        # Map image_id -> file_name
        image_id_to_name = {
            img["id"]: img["file_name"] for img in coco.get("images", [])
        }

        seg_by_file: Dict[str, List[List[float]]] = {}

        for ann in coco.get("annotations", []):
            if ann.get("category_id") != target_cat_id:
                continue
            img_id = ann["image_id"]
            file_name = image_id_to_name.get(img_id)
            if file_name is None:
                continue
            segs = ann.get("segmentation", [])
            if not isinstance(segs, list):
                continue
            seg_by_file.setdefault(file_name, []).extend(segs)

        self.coco_seg_by_file = seg_by_file
        print(
            f"✓ Loaded masks for {len(self.coco_seg_by_file)} images "
            f"(category='{self.coco_mask_category}')."
        )

    def _generate_foreground_mask_attention(
        self,
        img_pil: Image.Image,
        model: torch.nn.Module,
        device: torch.device,
        threshold: float = 0.25,
    ) -> np.ndarray:
        """
        Generate foreground mask using DINOv3 attention (alternative to rembg)
        """
        if self.transform is None:
            raise ValueError("transform must be set on dataset to use attention-based masks")

        img_tensor = self.transform(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model.backbone.forward_features(img_tensor)

            if "attn" in output:
                attn = output["attn"]
                attn_avg = attn.mean(dim=0)
                cls_attn = attn_avg[0, 1:]

                grid_size = int(np.sqrt(len(cls_attn)))
                attn_map = cls_attn.reshape(grid_size, grid_size).cpu().numpy()
                attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
                attn_map = cv2.resize(attn_map, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)

                mask = (attn_map >= threshold).astype(np.float32)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask = cv2.morphologyEx((mask * 255).astype(np.uint8), cv2.MORPH_OPEN, kernel)
                mask = (mask > 0).astype(np.float32)
                return mask
            else:
                print("Warning: Attention weights not available, using rembg")
                return self._generate_foreground_mask_rembg(img_pil)

    def _generate_foreground_mask_rembg(self, img_pil: Image.Image) -> np.ndarray:
        """
        Generate foreground mask using rembg.
        Returns binary mask where 1 = foreground, 0 = background.
        """
        try:
            from rembg import remove as rembg_remove  # local import

            fg = rembg_remove(img_pil)
            mask = (np.array(fg.convert("L")) > 0).astype(np.float32)
            return mask
        except Exception as e:
            print(f"Warning: rembg failed ({e}), using threshold-based mask")
            return self._generate_foreground_mask_threshold(img_pil)

    def _generate_foreground_mask_threshold(
        self, img_pil: Image.Image, white_thresh: int = 240
    ) -> np.ndarray:
        """
        Generate foreground mask using simple thresholding (fallback method)
        """
        img_np = np.array(img_pil.convert("RGB"))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        _, thresh = cv2.threshold(gray, white_thresh, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return np.ones(gray.shape, dtype=np.float32)

        largest = max(contours, key=cv2.contourArea)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)

        return (mask > 0).astype(np.float32)

    def _erode_mask(self, mask: np.ndarray, erode_px: int) -> np.ndarray:
        """
        Erode mask to remove halo artifacts.
        """
        if erode_px <= 0:
            return mask

        k = max(1, int(erode_px))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        eroded = cv2.erode((mask * 255).astype(np.uint8), kernel)

        return (eroded > 0).astype(np.float32)

    def _resize_preserve_aspect(self, img_pil: Image.Image, target_size: int) -> Image.Image:
        """
        Resize image preserving aspect ratio, center on white canvas.
        """
        orig_w, orig_h = img_pil.size
        scale = min(target_size / orig_w, target_size / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        img_resized = img_pil.resize((new_w, new_h), Image.LANCZOS)
        canvas = Image.new("RGB", (target_size, target_size), (255, 255, 255))

        offset_x = (target_size - new_w) // 2
        offset_y = (target_size - new_h) // 2
        canvas.paste(img_resized, (offset_x, offset_y))
        return canvas

    def _resize_mask_preserve_aspect(self, mask_pil: Image.Image, target_size: int) -> Image.Image:
        """
        Resize a single-channel foreground mask preserving aspect ratio,
        centering on a BLACK canvas. Foreground is assumed >0, background = 0.
        """
        orig_w, orig_h = mask_pil.size
        scale = min(target_size / orig_w, target_size / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        # Use nearest-neighbor to avoid creating gray values on edges
        mask_resized = mask_pil.resize((new_w, new_h), Image.NEAREST)

        # Black canvas => background will stay 0 after thresholding
        canvas = Image.new("L", (target_size, target_size), 0)

        offset_x = (target_size - new_w) // 2
        offset_y = (target_size - new_h) // 2
        canvas.paste(mask_resized, (offset_x, offset_y))
        return canvas


    def _preprocess_image(
        self,
        img_pil: Image.Image,
        seg_polys: Optional[List[List[float]]] = None,
    ) -> Tuple[Image.Image, Optional[np.ndarray]]:
        """
        Preprocess image using (optionally) COCO polygons:

        1. Build a binary foreground mask at the ORIGINAL resolution
           - COCO polygons -> mask
           - else rembg (if enabled)
           - else simple white-background threshold
        2. Tightly crop image + mask to the foreground bbox (with small padding).
        3. Resize both to target_size, preserving aspect ratio.
        4. Zero out everything outside the resized mask (set to white) so the
           model only sees the ant + tiny context.

        Returns:
            processed_img_pil, foreground_mask (H=W=target_size, values 0/1)
        """
        img_pil = img_pil.convert("RGB")
        w, h = img_pil.size

        # ----------------------------------------------------------
        # Step 1: build foreground mask at original resolution
        # ----------------------------------------------------------
        mask: Optional[np.ndarray] = None

        if seg_polys is not None:
            # Rasterize COCO polygons
            mask_raw = np.zeros((h, w), dtype=np.uint8)
            for poly in seg_polys:
                if not poly:
                    continue
                pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
                pts = np.round(pts).astype(np.int32)
                cv2.fillPoly(mask_raw, [pts], 255)
            mask = (mask_raw > 0).astype(np.float32)

        elif self.remove_bg:
            # rembg-based mask
            mask = self._generate_foreground_mask_rembg(img_pil)

        # If we still have no mask, fall back to threshold to at least crop
        if mask is None:
            img_np = np.array(img_pil)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            # everything that isn't near pure white becomes foreground
            mask = (gray < 250).astype(np.float32)

        # Optional erosion to remove halos
        if getattr(self, "mask_erode_px", 0) > 0:
            mask = self._erode_mask(mask, self.mask_erode_px)

        # ----------------------------------------------------------
        # Step 2: tight crop around foreground bbox (plus small pad)
        # ----------------------------------------------------------
        if self.crop_to_fg:
            coords = np.argwhere(mask > 0)
            if coords.size > 0:
                y0, x0 = coords.min(axis=0)
                y1, x1 = coords.max(axis=0)

                H, W = mask.shape
                # ~5% padding around bbox (you can tweak this)
                pad_y = int(0.05 * (y1 - y0))
                pad_x = int(0.05 * (x1 - x0))

                y0 = max(0, y0 - pad_y)
                y1 = min(H, y1 + pad_y)
                x0 = max(0, x0 - pad_x)
                x1 = min(W, x1 + pad_x)

                # Crop both image and mask
                img_pil = img_pil.crop((x0, y0, x1, y1))
                mask = mask[y0:y1, x0:x1]

        # ----------------------------------------------------------
        # Step 3: resize image + mask, then zero out padding
        # ----------------------------------------------------------
        # Resize RGB image with aspect-preserving square canvas
        img_pil = self._resize_preserve_aspect(img_pil, self.target_size)

        # Resize mask with aspect-preserving *black* canvas so outside = 0
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
        mask_pil = self._resize_mask_preserve_aspect(mask_pil, self.target_size)
        foreground_mask = (np.array(mask_pil) > 0).astype(np.float32)

        # Zero out everything outside the foreground mask (set to white)
        img_np = np.array(img_pil)
        mask_3ch = np.stack([foreground_mask, foreground_mask, foreground_mask], axis=-1)
        img_np = (img_np * mask_3ch + 255 * (1.0 - mask_3ch)).astype(np.uint8)
        img_pil = Image.fromarray(img_np)

        return img_pil, foreground_mask


    # ------------------------------------------------------------------ #
    # Dataset interface
    # ------------------------------------------------------------------ #

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # samples store: (img_path, species_idx, specimen_id, view_id)
        img_path, species_idx, specimen_id, view_id = self.samples[idx]

        img_path_obj = Path(img_path)
        if not img_path_obj.is_absolute():
            full_path = self.data_dir / img_path_obj
        else:
            full_path = img_path_obj

        # Look up COCO polygons if we have a mask file loaded
        seg_polys = None
        if hasattr(self, "coco_seg_by_file") and self.coco_seg_by_file:
            try:
                # key stored in COCO as path relative to data_dir, POSIX-style
                rel_key = str(full_path.relative_to(self.data_dir)).replace(os.sep, "/")
            except ValueError:
                # full_path not under data_dir; fall back to basename
                rel_key = full_path.name
            seg_polys = self.coco_seg_by_file.get(rel_key, None)

        img_pil = Image.open(full_path).convert("RGB")
        img_pil, foreground_mask = self._preprocess_image(img_pil, seg_polys=seg_polys)

        if self.transform is not None:
            img_tensor = self.transform(img_pil)
        else:
            img_tensor = T.ToTensor()(img_pil)

        if self.return_masks and foreground_mask is not None:
            mask_tensor = torch.from_numpy(foreground_mask).float().unsqueeze(0)
            return (
                img_tensor,
                torch.tensor(species_idx, dtype=torch.long),
                specimen_id,
                view_id,
                str(full_path),
                mask_tensor,
            )
        else:
            return (
                img_tensor,
                torch.tensor(species_idx, dtype=torch.long),
                specimen_id,
                view_id,
                str(full_path),
            )


# ============================================================================
# Training
# ============================================================================

def plot_training_progress(history, output_dir, epoch):
    """
    Plot:
      - Train loss + val CE loss
      - Similarity metrics (val set)
      - Separation + val accuracy
      - Text summary (current + best)
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    epochs = history.get("epoch", [])
    train_loss = history.get("train_loss", [])
    eval_epochs = history.get("eval_epoch", [])
    intra = history.get("intra_similarity", [])
    inter = history.get("inter_similarity", [])
    separation = history.get("separation", [])
    val_loss_ce = history.get("val_loss_ce", [])
    val_acc_top1 = history.get("val_acc_top1", [])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Training Progress - Epoch {epoch}", fontsize=16)

    # ------------------------------------------------------------
    # Panel 1: Train & val loss
    # ------------------------------------------------------------
    ax = axes[0, 0]
    if epochs and train_loss:
        n = min(len(epochs), len(train_loss))
        ax.plot(
            epochs[:n],
            train_loss[:n],
            "o-",
            linewidth=2,
            markersize=4,
            label="Train loss",
        )
    if eval_epochs and val_loss_ce:
        n = min(len(eval_epochs), len(val_loss_ce))
        ax.plot(
            eval_epochs[:n],
            val_loss_ce[:n],
            "s-",
            linewidth=2,
            markersize=4,
            label="Val CE loss",
        )
    ax.set_title("Training & Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle="--", alpha=0.3)
    if ax.lines:
        ax.legend()

    # ------------------------------------------------------------
    # Panel 2: Similarity metrics on validation set
    # ------------------------------------------------------------
    ax = axes[0, 1]
    if eval_epochs and intra and inter:
        n = min(len(eval_epochs), len(intra), len(inter))
        ax.plot(
            eval_epochs[:n],
            intra[:n],
            "o-",
            linewidth=2,
            markersize=4,
            label="Intra-class (same species)",
        )
        ax.plot(
            eval_epochs[:n],
            inter[:n],
            "o-",
            linewidth=2,
            markersize=4,
            label="Inter-class (different species)",
        )
        ax.set_title("Similarity Metrics (Validation)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cosine similarity")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()
    else:
        ax.axis("off")

    # ------------------------------------------------------------
    # Panel 3: Separation + val accuracy
    # ------------------------------------------------------------
    ax = axes[1, 0]
    have_sep = bool(eval_epochs and separation)
    if have_sep:
        n = min(len(eval_epochs), len(separation))
        sep_arr = np.array(separation[:n])
        ax.plot(
            eval_epochs[:n],
            sep_arr,
            "o-",
            linewidth=2,
            markersize=4,
            label="Separation (intra - inter)",
        )
        best_sep_idx = int(np.argmax(sep_arr))
        best_sep_epoch = eval_epochs[best_sep_idx]
        best_sep_val = sep_arr[best_sep_idx]
        ax.scatter(
            [best_sep_epoch],
            [best_sep_val],
            marker="*",
            s=150,
            label=f"Best sep: {best_sep_val:.4f} @ epoch {best_sep_epoch}",
        )
        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Separation")
        ax.grid(True, linestyle="--", alpha=0.3)

    # Optional: plot validation accuracy and/or open-set metrics on a second y-axis
    val_acc_top1 = history.get("val_accuracy_top1", [])
    eval_epochs = history.get("eval_epoch", list(range(1, len(val_acc_top1) + 1)))

    open_epochs = history.get("open_set_eval_epoch", [])
    novelty_auc = history.get("novelty_auc", [])
    knn_top1 = history.get("knn_top1", [])
    knn_top5 = history.get("knn_top5", [])
    centroid_top1 = history.get("centroid_top1", [])
    centroid_top5 = history.get("centroid_top5", [])

    def _xy(epochs, values):
        xs, ys = [], []
        for e, v in zip(epochs, values):
            if v is None:
                continue
            try:
                fv = float(v)
            except Exception:
                continue
            if not np.isfinite(fv):
                continue
            xs.append(int(e))
            ys.append(fv)
        return xs, ys

    have_acc = bool(eval_epochs and val_acc_top1)
    have_open = bool(open_epochs and (novelty_auc or knn_top1 or knn_top5 or centroid_top1 or centroid_top5))

    if have_acc or have_open:
        ax2 = ax.twinx()
        ax2.set_ylabel("Accuracy / AUC")

        if have_acc:
            xs, ys = _xy(eval_epochs, val_acc_top1)
            if xs:
                ax2.plot(xs, ys, "s--", linewidth=2, markersize=4, label="Val acc (top-1)")

        if have_open:
            xs, ys = _xy(open_epochs, novelty_auc)
            if xs:
                ax2.plot(xs, ys, "^-", linewidth=2, markersize=4, label="Novelty AUC")

            xs, ys = _xy(open_epochs, knn_top1)
            if xs:
                ax2.plot(xs, ys, "o-", linewidth=1.5, markersize=3, label="kNN top-1")

            xs, ys = _xy(open_epochs, knn_top5)
            if xs:
                ax2.plot(xs, ys, "o--", linewidth=1.5, markersize=3, label="kNN top-5")

            xs, ys = _xy(open_epochs, centroid_top1)
            if xs:
                ax2.plot(xs, ys, "d-", linewidth=1.5, markersize=3, label="Centroid top-1")

            xs, ys = _xy(open_epochs, centroid_top5)
            if xs:
                ax2.plot(xs, ys, "d--", linewidth=1.5, markersize=3, label="Centroid top-5")

        # Combined legend from both y-axes
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc="best")
    else:
        if ax.lines:
            ax.legend(loc="best")

    ax.set_title("Separation / Accuracy / Open-set AUC")

    # ------------------------------------------------------------

# Panel 4: Text summary
    # ------------------------------------------------------------
    ax = axes[1, 1]
    ax.axis("off")

    summary_lines = []
    if epochs and train_loss:
        summary_lines.append(f"Current Epoch: {epoch}")
        summary_lines.append("")
        summary_lines.append("Current Metrics:")
        summary_lines.append(f"  Train loss: {train_loss[-1]:.4f}")
    if val_loss_ce:
        summary_lines.append(f"  Val CE loss: {val_loss_ce[-1]:.4f}")
    if val_acc_top1:
        summary_lines.append(f"  Val acc (top-1): {val_acc_top1[-1]:.4f}")
    if intra and inter and separation:
        summary_lines.append(f"  Intra-class sim: {intra[-1]:.4f}")
        summary_lines.append(f"  Inter-class sim: {inter[-1]:.4f}")
        summary_lines.append(f"  Separation: {separation[-1]:.4f}")

    # Best separation
    if separation and eval_epochs:
        sep_arr_all = np.array(separation)
        best_sep_idx = int(np.argmax(sep_arr_all))
        best_sep_epoch = eval_epochs[best_sep_idx]
        best_sep_val = sep_arr_all[best_sep_idx]
        summary_lines.append("")
        summary_lines.append("Best separation:")
        summary_lines.append(f"  {best_sep_val:.4f} @ epoch {best_sep_epoch}")

    # Best validation accuracy
    if val_acc_top1 and eval_epochs:
        acc_arr_all = np.array(val_acc_top1)
        best_acc_idx = int(np.argmax(acc_arr_all))
        best_acc_epoch = eval_epochs[best_acc_idx]
        best_acc_val = acc_arr_all[best_acc_idx]
        summary_lines.append("")
        summary_lines.append("Best val acc:")
        summary_lines.append(f"  {best_acc_val:.4f} @ epoch {best_acc_epoch}")

    ax.text(
        0.01,
        0.99,
        "\n".join(summary_lines),
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "training_progress.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"  📊 Training plot updated: {out_path}")


def train_epoch(
    model,
    dataloader,
    optimizer,
    contrastive_loss_fn,
    ce_criterion,
    device,
    epoch,
    args,
    specimen_to_idx=None,
    # ── v24: Multi-modal auxiliary data ──
    traits_by_specimen=None,
    continuous_names=None,
    meristic_names=None,
    categorical_names=None,
    cont_means=None,
    cont_stds=None,
    dna_by_specimen=None,
    morph_by_specimen=None,
):
    """
    Train for one epoch with contrastive loss (+ optional CE classifier loss).
    v24: Also computes auxiliary trait, DNA, and morph losses when data is available.

    Batch formats supported:
      - 6-tuple: (img, species_idx, specimen_id, view_id, img_path, mask)
      - 5-tuple: (img, species_idx, specimen_id, view_id, img_path)
      - 4-tuple: (img, species_idx, specimen_id, img_path)
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    # v24: determine if we need features for trait heads
    need_features = (model.trait_head is not None and traits_by_specimen)

    for batch_idx, batch in enumerate(dataloader):
        if len(batch) == 6:
            images, species_labels, specimen_ids, view_ids, img_paths, masks = batch
        elif len(batch) == 5:
            images, species_labels, specimen_ids, view_ids, img_paths = batch
            masks = None
        elif len(batch) == 4:
            images, species_labels, specimen_ids, img_paths = batch
            view_ids = None
            masks = None
        else:
            raise ValueError(f"Unexpected batch structure of length {len(batch)}")

        images = images.to(device, non_blocking=True)
        species_labels = species_labels.to(device, non_blocking=True)

        # Build specimen indices if needed
        specimen_idx = None
        if args.supcon_label_mode in ("specimen", "both"):
            if specimen_to_idx is None:
                raise ValueError(
                    "specimen_to_idx must be provided when supcon_label_mode is "
                    "'specimen' or 'both'"
                )
            specimen_idx = torch.tensor(
                [specimen_to_idx[sid] for sid in specimen_ids],
                dtype=torch.long,
                device=device,
            )

        # Forward — v24: get features too when trait head is active
        if need_features or getattr(args, "use_classifier_head", False):
            embeddings, features, logits = model(images, return_features=True, return_logits=True)
        else:
            embeddings = model(images)
            features = None
            logits = None

        loss = torch.tensor(0.0, device=device)

        con_loss = None
        con_loss_species = None
        con_loss_specimen = None

        # Contrastive loss (SupCon or Triplet, but here we assume SupCon)
        if contrastive_loss_fn is not None and args.supcon_weight > 0.0:
            if args.supcon_label_mode == "species":
                # One SupCon on species labels
                con_loss = contrastive_loss_fn(embeddings, species_labels)
            elif args.supcon_label_mode == "specimen":
                # One SupCon on specimen indices
                con_loss = contrastive_loss_fn(embeddings, specimen_idx)
            elif args.supcon_label_mode == "both":
                # Two SupCon terms: species + specimen, averaged
                con_loss_species = contrastive_loss_fn(embeddings, species_labels)
                con_loss_specimen = contrastive_loss_fn(embeddings, specimen_idx)
                con_loss = 0.5 * (con_loss_species + con_loss_specimen)
            else:
                raise ValueError(f"Unknown supcon_label_mode: {args.supcon_label_mode}")

            loss = loss + args.supcon_weight * con_loss
        # (else: con_loss stays None)


        if args.use_classifier_head and ce_criterion is not None and args.ce_weight > 0.0:
            ce_loss = ce_criterion(logits, species_labels)
            loss = loss + args.ce_weight * ce_loss
        else:
            ce_loss = None

        # ── v24: Auxiliary multi-modal losses ──────────────────────────
        aux_losses = {}
        aux_loss_total = torch.tensor(0.0, device=device)

        if features is not None or dna_by_specimen or morph_by_specimen:
            # Build batch-level tensors for available modalities
            sid_list = [str(s) for s in specimen_ids]

            trait_valid_mask = None
            cont_t = meris_t = cat_t = None
            if traits_by_specimen and (continuous_names or meristic_names or categorical_names):
                cont_t, meris_t, cat_t, trait_valid_mask = build_trait_tensors_for_batch(
                    sid_list, traits_by_specimen,
                    continuous_names or [], meristic_names or [], categorical_names or [],
                    cont_means, cont_stds, device,
                )

            dna_input_t = None
            dna_valid_mask = None
            dna_sp_labels = None
            if dna_by_specimen:
                dna_input_t, dna_valid_mask = build_dna_tensor_for_batch(
                    sid_list, dna_by_specimen, device,
                )
                if dna_input_t is not None and dna_valid_mask.any():
                    dna_sp_labels = species_labels[dna_valid_mask]

            morph_input_t = None
            morph_valid_mask = None
            if morph_by_specimen:
                morph_input_t, morph_valid_mask = build_morph_tensor_for_batch(
                    sid_list, morph_by_specimen, device,
                )

            aux_losses = model.compute_auxiliary_losses(
                features=features if features is not None else embeddings,
                embeddings=embeddings,
                species_labels=species_labels,
                trait_valid_mask=trait_valid_mask,
                cont_targets=cont_t,
                meris_targets=meris_t,
                cat_targets=cat_t,
                dna_input=dna_input_t,
                dna_valid_mask=dna_valid_mask,
                dna_species_labels=dna_sp_labels,
                morph_input=morph_input_t,
                morph_valid_mask=morph_valid_mask,
            )

            # Apply weights from args
            trait_weight = getattr(args, "trait_weight", 0.5)
            dna_weight = getattr(args, "dna_weight", 0.5)
            xmodal_weight = getattr(args, "xmodal_weight", 0.3)

            for k, v in aux_losses.items():
                if k.startswith("trait_"):
                    aux_loss_total = aux_loss_total + trait_weight * v
                elif k == "dna_ce":
                    aux_loss_total = aux_loss_total + dna_weight * v
                elif k.startswith("xmodal_"):
                    aux_loss_total = aux_loss_total + xmodal_weight * v

            loss = loss + aux_loss_total

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if batch_idx % 10 == 0:
            # Build log message
            parts = [f"  Batch {batch_idx}/{len(dataloader)}, Total: {loss.item():.4f}"]
            if con_loss is not None:
                if args.supcon_label_mode == "both" and con_loss_species is not None and con_loss_specimen is not None:
                    parts.append(f"SupCon(avg)={con_loss.item():.4f} [sp={con_loss_species.item():.4f}, spec={con_loss_specimen.item():.4f}]")
                else:
                    parts.append(f"SupCon={con_loss.item():.4f}")
            if ce_loss is not None:
                parts.append(f"CE={ce_loss.item():.4f}")
            # v24: log auxiliary losses
            for k, v in aux_losses.items():
                parts.append(f"{k}={v.item():.4f}")
            print(" | ".join(parts))

    avg_loss = total_loss / max(1, n_batches)
    print(f"Epoch {epoch} - Avg Total Loss: {avg_loss:.4f}")
    return avg_loss


def evaluate_model(
    model,
    dataloader,
    device,
    compute_classification: bool = False,
    ce_criterion: Optional[nn.Module] = None,
):
    """
    Evaluate on a dataloader.

    Always computes:
      - mean intra-class cosine similarity
      - mean inter-class cosine similarity
      - separation (intra - inter)
      - per-view metrics

    Optionally (if compute_classification=True and ce_criterion is not None):
      - CE loss on classifier head
      - top-1 classification accuracy
    """
    model.eval()

    all_embeddings: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_specimen_ids: List[str] = []
    all_view_ids: List[str] = []
    all_paths: List[str] = []

    val_total = 0
    val_correct = 0
    val_loss_sum = 0.0

    with torch.no_grad():
        for batch in dataloader:

            # ------------------------------------------------------------
            # Accept either dict batches OR tuple/list batches
            # ------------------------------------------------------------
            if isinstance(batch, dict):
                images = batch["image"]
                labels = batch["species_idx"]
                specimen_ids = batch["specimen_id"]
                view_ids = batch["view_id"]
                img_paths = batch["image_path"]
            else:
                # Common tuple layouts:
                # (image, species_idx, specimen_id, view_id, image_path)          -> len=5
                # (image, species_idx, specimen_id, view_id, image_path, mask)    -> len=6
                if len(batch) == 6:
                    images, labels, specimen_ids, view_ids, img_paths, _mask = batch
                elif len(batch) == 5:
                    images, labels, specimen_ids, view_ids, img_paths = batch
                elif len(batch) == 4:
                    images, labels, specimen_ids, img_paths = batch
                    view_ids = ["unknown"] * len(specimen_ids)
                else:
                    raise TypeError(f"Unexpected batch tuple length: {len(batch)}")

            # Move tensors to device
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # ------------------------------------------------------------
            # Forward pass
            # ------------------------------------------------------------
            if compute_classification and ce_criterion is not None:
                embeddings, logits = model(images, return_logits=True)
                ce_loss = ce_criterion(logits, labels)
                val_loss_sum += float(ce_loss.item()) * images.size(0)

                preds = torch.argmax(logits, dim=1)
                val_correct += int((preds == labels).sum().item())
                val_total += int(images.size(0))
            else:
                embeddings = model.get_embeddings(images)

            all_embeddings.append(embeddings.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
            all_specimen_ids.extend([str(s) for s in specimen_ids])
            all_view_ids.extend([str(v) for v in view_ids])
            all_paths.extend(list(img_paths))

    if not all_embeddings:
        raise ValueError("No embeddings collected during evaluation.")

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_view_ids_arr = np.array(all_view_ids)

    # L2-normalize embeddings
    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True) + 1e-12
    all_emb_norm = all_embeddings / norms

    # ------------------------------------------------------------
    # Global pairwise similarities
    # ------------------------------------------------------------
    sim_matrix = all_emb_norm @ all_emb_norm.T
    same_mask = all_labels[:, None] == all_labels[None, :]
    np.fill_diagonal(same_mask, False)
    diff_mask = ~same_mask

    same_sims = sim_matrix[same_mask]
    diff_sims = sim_matrix[diff_mask]

    mean_intra = float(same_sims.mean()) if same_sims.size > 0 else 0.0
    mean_inter = float(diff_sims.mean()) if diff_sims.size > 0 else 0.0
    separation = mean_intra - mean_inter

    # ------------------------------------------------------------
    # Per-view metrics
    # ------------------------------------------------------------
    view_metrics: Dict[str, Dict[str, Any]] = {}
    unique_views = sorted(set(all_view_ids))
    for v in unique_views:
        mask = all_view_ids_arr == v
        n_v = int(mask.sum())
        if n_v < 2:
            continue

        emb_v = all_emb_norm[mask]
        labels_v = all_labels[mask]

        sim_v = emb_v @ emb_v.T
        same_mask_v = labels_v[:, None] == labels_v[None, :]
        np.fill_diagonal(same_mask_v, False)
        diff_mask_v = ~same_mask_v

        same_sims_v = sim_v[same_mask_v]
        diff_sims_v = sim_v[diff_mask_v]

        m_intra_v = float(same_sims_v.mean()) if same_sims_v.size > 0 else 0.0
        m_inter_v = float(diff_sims_v.mean()) if diff_sims_v.size > 0 else 0.0
        sep_v = m_intra_v - m_inter_v

        view_metrics[v] = {
            "n_images": n_v,
            "mean_intra_similarity": m_intra_v,
            "mean_inter_similarity": m_inter_v,
            "separation": sep_v,
        }

    # ------------------------------------------------------------
    # Package metrics
    # ------------------------------------------------------------
    metrics: Dict[str, Any] = {
        "mean_intra_similarity": mean_intra,
        "mean_inter_similarity": mean_inter,
        "separation": separation,
        "per_view": view_metrics,
    }

    if compute_classification and ce_criterion is not None and val_total > 0:
        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total
        metrics["val_loss_ce"] = val_loss
        metrics["val_accuracy_top1"] = val_acc
        metrics["n_val_samples"] = val_total

    # ------------------------------------------------------------
    # Pretty printing
    # ------------------------------------------------------------
    print("Evaluation Metrics:")
    print(f"  Mean intra-class similarity: {mean_intra:.4f}")
    print(f"  Mean inter-class similarity: {mean_inter:.4f}")
    print(f"  Separation: {separation:.4f}")
    if view_metrics:
        print("  Per-view separation:")
        for v, m in view_metrics.items():
            print(f"    View {v}: sep={m['separation']:.4f} (n={m['n_images']} images)")
    if "val_loss_ce" in metrics:
        print(f"  Val CE loss (classifier): {metrics['val_loss_ce']:.4f}")
        print(
            f"  Val accuracy (top-1): {metrics['val_accuracy_top1']:.4f} "
            f"({val_correct}/{val_total})"
        )

    return metrics, all_emb_norm, all_labels, all_specimen_ids, all_view_ids, all_paths



def collect_embeddings(
    model,
    dataloader,
    device: torch.device,
):
    """
    Lightweight embedding collection (no N×N similarity matrix).
    Returns:
      emb_norm: (N, D) float32 numpy
      labels:   (N,) int numpy
      specimen_ids: list[str]
      view_ids:     list[str]
      paths:        list[str]
    """
    model.eval()
    all_emb = []
    all_labels = []
    all_specimen_ids = []
    all_view_ids = []
    all_paths = []

    with torch.no_grad():
        for batch in dataloader:
            # Supported batch formats:
            # (img, species_idx, specimen_id, view_id, img_path, mask) -> len=6
            # (img, species_idx, specimen_id, view_id, img_path)       -> len=5
            # (img, species_idx, specimen_id, img_path)                -> len=4
            if isinstance(batch, (list, tuple)):
                if len(batch) == 6:
                    images, labels, specimen_ids, view_ids, img_paths, _mask = batch
                elif len(batch) == 5:
                    images, labels, specimen_ids, view_ids, img_paths = batch
                elif len(batch) == 4:
                    images, labels, specimen_ids, img_paths = batch
                    view_ids = ["unknown"] * len(specimen_ids)
                else:
                    raise TypeError(f"Unexpected batch tuple length: {len(batch)}")
            else:
                raise TypeError(f"Unexpected batch type: {type(batch)}")

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            emb = model.get_embeddings(images) if hasattr(model, "get_embeddings") else model(images)
            emb = emb.detach().float().cpu().numpy()

            # L2 normalize (important for cosine similarity)
            norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
            emb = emb / norms

            all_emb.append(emb.astype(np.float32))
            all_labels.append(labels.detach().cpu().numpy().astype(np.int64))

            all_specimen_ids.extend([str(s) for s in specimen_ids])
            all_view_ids.extend([str(v) for v in view_ids])
            all_paths.extend([str(p) for p in img_paths])

    if len(all_emb) == 0:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            [],
            [],
            [],
        )

    emb_norm = np.concatenate(all_emb, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return emb_norm, labels, all_specimen_ids, all_view_ids, all_paths


def aggregate_by_specimen(
    emb_norm: np.ndarray,
    labels: np.ndarray,
    specimen_ids: List[str],
):
    """
    Aggregate image embeddings to ONE embedding per specimen (mean over views).
    Returns:
      spec_emb_norm: (M, D)
      spec_labels:   (M,)
      spec_ids:      list[str]
    """
    if emb_norm.size == 0:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            [],
        )

    dim = emb_norm.shape[1]
    sum_map: Dict[str, np.ndarray] = {}
    count_map: Dict[str, int] = {}
    label_map: Dict[str, int] = {}

    for e, l, sid in zip(emb_norm, labels, specimen_ids):
        if sid not in sum_map:
            sum_map[sid] = np.zeros((dim,), dtype=np.float64)
            count_map[sid] = 0
            label_map[sid] = int(l)
        else:
            # sanity: specimen should not change species label
            if int(l) != int(label_map[sid]):
                raise ValueError(
                    f"Specimen {sid} has inconsistent species labels: "
                    f"{label_map[sid]} vs {int(l)}"
                )
        sum_map[sid] += e.astype(np.float64)
        count_map[sid] += 1

    spec_ids = sorted(sum_map.keys())
    spec_emb = np.stack([sum_map[sid] / max(1, count_map[sid]) for sid in spec_ids], axis=0).astype(np.float32)
    # re-normalize
    norms = np.linalg.norm(spec_emb, axis=1, keepdims=True) + 1e-12
    spec_emb_norm = spec_emb / norms
    spec_labels = np.array([label_map[sid] for sid in spec_ids], dtype=np.int64)
    return spec_emb_norm, spec_labels, spec_ids


def compute_species_centroids(
    train_spec_emb: np.ndarray,
    train_spec_labels: np.ndarray,
):
    """
    Compute per-species centroids from specimen embeddings (each specimen contributes once).
    Returns:
      centroid_emb_norm: (S, D)
      centroid_labels:   (S,)
    """
    if train_spec_emb.size == 0:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    dim = train_spec_emb.shape[1]
    sums: Dict[int, np.ndarray] = {}
    counts: Dict[int, int] = {}
    for e, l in zip(train_spec_emb, train_spec_labels):
        l = int(l)
        if l not in sums:
            sums[l] = np.zeros((dim,), dtype=np.float64)
            counts[l] = 0
        sums[l] += e.astype(np.float64)
        counts[l] += 1

    sp_labels = sorted(sums.keys())
    centroids = np.stack([sums[l] / max(1, counts[l]) for l in sp_labels], axis=0).astype(np.float32)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12
    centroids_norm = centroids / norms
    return centroids_norm, np.array(sp_labels, dtype=np.int64)


def _eer_threshold_from_roc(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray) -> Tuple[float, float]:
    """Return (eer, threshold_at_eer) from ROC arrays.

    We ignore non-finite thresholds (roc_curve often includes +inf as the first threshold).
    """
    fnr = 1.0 - tpr
    if thresholds.size == 0:
        return 1.0, float("nan")

    finite = np.isfinite(thresholds)
    if np.any(finite):
        fpr2 = fpr[finite]
        fnr2 = fnr[finite]
        thr2 = thresholds[finite]
        idx = int(np.argmin(np.abs(fpr2 - fnr2)))
        eer = float((fpr2[idx] + fnr2[idx]) / 2.0)
        thr = float(thr2[idx])
        return eer, thr

    # fallback (should be rare)
    idx = int(np.argmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    thr = float(thresholds[idx])
    return eer, thr



def _threshold_at_far(fpr: np.ndarray, thresholds: np.ndarray, far_target: float) -> Optional[float]:
    """Pick the *highest finite* threshold that achieves FPR <= far_target.

    Notes:
      - sklearn.metrics.roc_curve returns an initial threshold of +inf.
        If we don't filter it out, FAR-threshold queries often return "inf",
        which is not a usable similarity threshold.
    """
    if fpr.size == 0 or thresholds.size == 0:
        return None

    mask = (fpr <= far_target) & np.isfinite(thresholds)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return None
    # highest threshold => most conservative accept rule
    return float(np.max(thresholds[idx]))



def evaluate_open_set_retrieval(
    model,
    train_eval_loader,
    val_known_loader,
    device: torch.device,
    output_dir: str,
    epoch: Optional[int] = None,
    val_novel_loader=None,
    topk: Optional[List[int]] = None,
    far_targets: Optional[List[float]] = None,
):
    """
    Open-set evaluation focused on what matters for 'new species' clustering:

      1) Specimen-level kNN retrieval (val_known -> train) top-1 / top-k
      2) Species-centroid classification (val_known -> train centroids) top-1
      3) Novelty threshold calibration (known vs novel) using max-sim-to-centroid:
           - ROC/AUC
           - EER + threshold@EER
           - threshold@desired FAR targets

    If val_novel_loader is None, novelty calibration is skipped, but we still compute
    a verification ROC/AUC on val_known specimen pairs (same-species vs different-species).
    """
    if topk is None:
        topk = [1, 5]
    topk = sorted(set(int(k) for k in topk if int(k) >= 1))
    far_targets = far_targets or [0.01, 0.05]

    os.makedirs(output_dir, exist_ok=True)

    # -------------------------
    # Collect + aggregate
    # -------------------------
    tr_emb, tr_lab, tr_sid, _tr_vid, _tr_paths = collect_embeddings(model, train_eval_loader, device)
    vk_emb, vk_lab, vk_sid, _vk_vid, _vk_paths = collect_embeddings(model, val_known_loader, device)

    tr_spec_emb, tr_spec_lab, tr_spec_ids = aggregate_by_specimen(tr_emb, tr_lab, tr_sid)
    vk_spec_emb, vk_spec_lab, vk_spec_ids = aggregate_by_specimen(vk_emb, vk_lab, vk_sid)

    out: Dict[str, Any] = {
        "epoch": int(epoch) if epoch is not None else None,
        "train_images": int(len(tr_lab)),
        "train_specimens": int(len(tr_spec_lab)),
        "val_known_images": int(len(vk_lab)),
        "val_known_specimens": int(len(vk_spec_lab)),
        "topk": topk,
    }

    if tr_spec_emb.size == 0 or vk_spec_emb.size == 0:
        out["error"] = "Empty train/val embeddings; cannot compute open-set metrics."
        return out

    # -------------------------
    # 1) Specimen-level kNN retrieval: val_known -> train specimens
    # -------------------------
    sim_knn = vk_spec_emb @ tr_spec_emb.T  # cosine sim (normalized)
    sort_idx = np.argsort(-sim_knn, axis=1)

    for k in topk:
        kk = min(k, sort_idx.shape[1])
        topk_labels = tr_spec_lab[sort_idx[:, :kk]]  # (M, kk)
        correct = (topk_labels == vk_spec_lab[:, None]).any(axis=1)
        out[f"knn_top{k}_acc"] = float(correct.mean())

    out["knn_top1_mean_sim"] = float(sim_knn[np.arange(sim_knn.shape[0]), sort_idx[:, 0]].mean())

    # -------------------------
    # 2) Species-centroid top-1 classification
    # -------------------------
    cent_emb, cent_labels = compute_species_centroids(tr_spec_emb, tr_spec_lab)
    if cent_emb.size > 0:
        sim_cent = vk_spec_emb @ cent_emb.T
        pred_idx = np.argmax(sim_cent, axis=1)
        pred_species = cent_labels[pred_idx]
        out["centroid_top1_acc"] = float((pred_species == vk_spec_lab).mean())
        out["centroid_top1_mean_sim"] = float(sim_cent[np.arange(sim_cent.shape[0]), pred_idx].mean())
    else:
        out["centroid_top1_acc"] = None
        out["centroid_top1_mean_sim"] = None

    # -------------------------
    # 3) Novelty calibration (known vs novel) using max-sim-to-centroid
    # -------------------------
    if val_novel_loader is not None and cent_emb.size > 0:
        vn_emb, vn_lab, vn_sid, _vn_vid, _vn_paths = collect_embeddings(model, val_novel_loader, device)
        vn_spec_emb, vn_spec_lab, vn_spec_ids = aggregate_by_specimen(vn_emb, vn_lab, vn_sid)
        out["val_novel_images"] = int(len(vn_lab))
        out["val_novel_specimens"] = int(len(vn_spec_lab))

        if vn_spec_emb.size > 0:
            sim_vk = vk_spec_emb @ cent_emb.T
            sim_vn = vn_spec_emb @ cent_emb.T
            max_vk = sim_vk.max(axis=1)
            max_vn = sim_vn.max(axis=1)


            # --- DEBUG: per-specimen scores + predicted centroid label (what you deploy) ---
            # Writes a TSV for quick inspection: which specimens are close to which train centroids,
            # and whether known specimens rank their true species highly.
            try:
                def _first_path_map(sids, paths):
                    m = {}
                    for sid_i, p in zip(sids, paths):
                        sid_i = str(sid_i)
                        if sid_i not in m:
                            m[sid_i] = str(p)
                    return m

                vk_first_path = _first_path_map(vk_sid, _vk_paths)
                vn_first_path = _first_path_map(vn_sid, _vn_paths)

                # precompute top-k centroid predictions for known/novel
                def _debug_rows(spec_emb, spec_lab, spec_ids, split_name, first_path_map):
                    rows = []
                    sims = spec_emb @ cent_emb.T  # (N, C)
                    order = np.argsort(-sims, axis=1)
                    for i, sid_i in enumerate(spec_ids):
                        sid_i = str(sid_i)
                        true_lab = int(spec_lab[i])
                        top = order[i, : min(5, order.shape[1])]
                        top_labels = [int(cent_labels[j]) for j in top]
                        top_sims = [float(sims[i, j]) for j in top]
                        pred_lab = top_labels[0] if top_labels else None
                        max_sim = top_sims[0] if top_sims else None
                        second = top_sims[1] if len(top_sims) > 1 else None
                        margin = (max_sim - second) if (max_sim is not None and second is not None) else None

                        # rank of true label among centroids (only meaningful for val_known)
                        rank_true = None
                        if split_name == "val_known":
                            # find position of true label in sorted centroid labels
                            try:
                                # locate all matches (there should be exactly one centroid per label)
                                true_pos = np.where(cent_labels[order[i]] == true_lab)[0]
                                if true_pos.size > 0:
                                    rank_true = int(true_pos[0] + 1)  # 1-based
                            except Exception:
                                rank_true = None

                        rows.append({
                            "split": split_name,
                            "specimen_id": sid_i,
                            "true_species_idx": true_lab,
                            "pred_species_idx": pred_lab,
                            "max_sim_to_train_centroid": max_sim,
                            "margin_top1_top2": margin,
                            "rank_true_species_among_centroids": rank_true,
                            "top5_pred_species_idx": ",".join(str(x) for x in top_labels),
                            "top5_sims": ",".join(f"{x:.6f}" for x in top_sims),
                            "example_image_path": first_path_map.get(sid_i, ""),
                        })
                    return rows

                debug_rows = []
                debug_rows.extend(_debug_rows(vk_spec_emb, vk_spec_lab, vk_spec_ids, "val_known", vk_first_path))
                debug_rows.extend(_debug_rows(vn_spec_emb, vn_spec_lab, vn_spec_ids, "val_novel", vn_first_path))

                debug_path = os.path.join(output_dir, f"open_set_debug_epoch_{epoch:03d}.tsv")
                with open(debug_path, "w") as f:
                    cols = [
                        "split",
                        "specimen_id",
                        "true_species_idx",
                        "pred_species_idx",
                        "max_sim_to_train_centroid",
                        "margin_top1_top2",
                        "rank_true_species_among_centroids",
                        "top5_pred_species_idx",
                        "top5_sims",
                        "example_image_path",
                    ]
                    f.write("\t".join(cols) + "\n")
                    for r in debug_rows:
                        f.write("\t".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + "\n")
                out["open_set_debug_tsv"] = debug_path
                print(f"✓ Open-set debug TSV saved: {debug_path}")
            except Exception:
                pass

            # --- diagnostics: score distributions (specimen->centroid max cosine) ---
            def _q(x):
                if x.size == 0:
                    return {}
                qs = np.quantile(x, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
                return {
                    "q01": float(qs[0]),
                    "q05": float(qs[1]),
                    "q25": float(qs[2]),
                    "q50": float(qs[3]),
                    "q75": float(qs[4]),
                    "q95": float(qs[5]),
                    "q99": float(qs[6]),
                    "min": float(x.min()),
                    "max": float(x.max()),
                    "mean": float(x.mean()),
                    "std": float(x.std()),
                }

            out["novelty_score_quantiles"] = {
                "val_known": _q(max_vk),
                "val_novel": _q(max_vn),
            }

            if output_dir:
                try:
                    import matplotlib.pyplot as plt
                    fig = plt.figure(figsize=(6, 4))
                    plt.hist(max_vk, bins=40, alpha=0.6, label="val_known", density=True)
                    plt.hist(max_vn, bins=40, alpha=0.6, label="val_novel", density=True)
                    plt.xlabel("max cosine similarity to any TRAIN species centroid")
                    plt.ylabel("density")
                    plt.title("Open-set novelty score distributions")
                    plt.legend()
                    hist_path = os.path.join(output_dir, f"open_set_score_hist_epoch_{epoch:03d}.png")
                    plt.tight_layout()
                    plt.savefig(hist_path, dpi=180)
                    plt.close(fig)
                    out["novelty_score_hist_png"] = hist_path
                except Exception:
                    pass

            y_true = np.concatenate([np.ones_like(max_vk), np.zeros_like(max_vn)], axis=0)
            y_score = np.concatenate([max_vk, max_vn], axis=0)

            fpr, tpr, thr = roc_curve(y_true, y_score)
            out["novelty_auc"] = float(auc(fpr, tpr))

            # --- NEW: save novelty ROC curve ---
            try:
                roc_path = os.path.join(output_dir, f"open_set_novelty_roc_epoch_{epoch:03d}.png")
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.plot(fpr, tpr, linewidth=2)
                ax.plot([0, 1], [0, 1], linestyle='--', linewidth=1)
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title(f"Novelty ROC (AUC={out['novelty_auc']:.3f})")
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(roc_path, dpi=180)
                plt.close(fig)
                out['novelty_roc_png'] = roc_path
            except Exception:
                pass


            eer, thr_eer = _eer_threshold_from_roc(fpr, tpr, thr)
            out["novelty_eer"] = eer
            out["threshold_eer"] = thr_eer

            # FAR-based thresholds (FPR targets)
            far_out = {}
            for far_t in far_targets:
                th = _threshold_at_far(fpr, thr, float(far_t))
                far_out[str(far_t)] = th
            out["thresholds_at_far"] = far_out

            out["maxsim_known_mean"] = float(max_vk.mean())
            out["maxsim_known_p05"] = float(np.percentile(max_vk, 5))
            out["maxsim_novel_mean"] = float(max_vn.mean())
            out["maxsim_novel_p95"] = float(np.percentile(max_vn, 95))
        else:
            out["novelty_auc"] = None
            out["novelty_eer"] = None
            out["threshold_eer"] = None
            out["thresholds_at_far"] = {}
    else:
        out["val_novel_images"] = 0
        out["val_novel_specimens"] = 0

        # Pairwise verification AUC on val_known specimen embeddings
        # Positives: same-species pairs, Negatives: different-species pairs
        M = vk_spec_emb.shape[0]
        if M >= 3:
            sims = []
            y = []
            for i in range(M):
                for j in range(i + 1, M):
                    sims.append(float(vk_spec_emb[i].dot(vk_spec_emb[j])))
                    y.append(1 if int(vk_spec_lab[i]) == int(vk_spec_lab[j]) else 0)
            y = np.array(y, dtype=np.int64)
            sims = np.array(sims, dtype=np.float32)
            if y.sum() > 0 and (y == 0).sum() > 0:
                fpr, tpr, thr = roc_curve(y, sims)
                out["verification_auc"] = float(auc(fpr, tpr))
                # --- NEW: save verification ROC curve ---
                try:
                    roc_path = os.path.join(output_dir, f"verification_roc_epoch_{epoch:03d}.png")
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.plot(fpr, tpr, linewidth=2)
                    ax.plot([0, 1], [0, 1], linestyle='--', linewidth=1)
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title(f"Verification ROC (AUC={out['verification_auc']:.3f})")
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(roc_path, dpi=180)
                    plt.close(fig)
                    out['verification_roc_png'] = roc_path
                except Exception:
                    pass

                eer, thr_eer = _eer_threshold_from_roc(fpr, tpr, thr)
                out["verification_eer"] = eer
                out["verification_threshold_eer"] = thr_eer
            else:
                out["verification_auc"] = None
                out["verification_eer"] = None
                out["verification_threshold_eer"] = None
        else:
            out["verification_auc"] = None
            out["verification_eer"] = None
            out["verification_threshold_eer"] = None

    # Save
    tag = f"epoch_{epoch:03d}" if epoch is not None else "final"
    out_path = os.path.join(output_dir, f"open_set_metrics_{tag}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"✓ Open-set metrics saved: {out_path}")

    # Print a short summary
    if out.get("threshold_eer") is not None:
        print(
            f"  Open-set novelty: AUC={out['novelty_auc']:.3f}, "
            f"EER={out['novelty_eer']:.3f}, thr@EER={out['threshold_eer']:.3f}"
        )
        if out.get("thresholds_at_far"):
            for far_t, th in out["thresholds_at_far"].items():
                if th is not None:
                    print(f"    thr@FAR={far_t}: {th:.3f}")
    elif out.get("verification_auc") is not None:
        print(
            f"  Verification (same/diff species pairs): AUC={out['verification_auc']:.3f}, "
            f"EER={out['verification_eer']:.3f}, thr@EER={out['verification_threshold_eer']:.3f}"
        )

    for k in topk:
        if f"knn_top{k}_acc" in out:
            print(f"  kNN retrieval top-{k}: {out[f'knn_top{k}_acc']:.3f}")

    return out

def evaluate_open_set_retrieval_specimen_centroid(
    model: nn.Module,
    train_eval_loader: DataLoader,
    val_known_loader: DataLoader,
    val_novel_loader: DataLoader,
    device: torch.device,
    output_dir: Optional[str],
    epoch: int,
    topk: int = 5,
    far_targets: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Alias for evaluate_open_set_retrieval().

    Kept for clarity: this evaluation is specimen-level and centroid-based:
      TRAIN -> (image embeddings) -> specimen mean -> species centroid mean.
      VAL(specimen) -> max cosine to TRAIN species centroids -> ROC/AUC/EER.
    """
    return evaluate_open_set_retrieval(
        model=model,
        train_eval_loader=train_eval_loader,
        val_known_loader=val_known_loader,
        val_novel_loader=val_novel_loader,
        device=device,
        output_dir=output_dir,
        epoch=epoch,
        topk=topk,
        far_targets=far_targets,
    )





# ============================================================================
# Zero-Shot Species Prediction
# ============================================================================

def _write_predictions_tsv(results: List[Dict[str, Any]], tsv_path: str, multi_view: bool = False):
    import csv

    os.makedirs(os.path.dirname(tsv_path), exist_ok=True)

    if not results:
        with open(tsv_path, "w", newline="") as f:
            f.write("")
        return

    # Standard columns
    if not multi_view:
        fieldnames = [
            "specimen_id", "image_path",
            "predicted_species", "similarity_score", "is_new_species",
            "similarity_scores_json",
        ]
    else:
        fieldnames = [
            "specimen_id", "n_views_used", "image_paths_joined",
            "predicted_species", "similarity_score", "is_new_species",
            "similarity_scores_json",
        ]

    with open(tsv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for r in results:
            row = {}
            if not multi_view:
                row["specimen_id"] = r.get("specimen_id", "")
                row["image_path"] = r.get("image_path", "")
            else:
                row["specimen_id"] = r.get("specimen_id", "")
                row["n_views_used"] = r.get("n_views_used", "")
                img_paths = r.get("image_paths", [])
                row["image_paths_joined"] = ";".join(img_paths) if isinstance(img_paths, list) else str(img_paths)

            row["predicted_species"] = r.get("predicted_species", "")
            row["similarity_score"] = r.get("similarity_score", "")
            row["is_new_species"] = r.get("is_new_species", "")

            sims = r.get("similarity_scores", None)
            row["similarity_scores_json"] = json.dumps(sims) if isinstance(sims, dict) else ""

            writer.writerow(row)


def zero_shot_prediction(
    model,
    image_paths: List[str],
    memory_bank: SpeciesMemoryBank,
    device: torch.device,
    output_dir: str,
    transform=None,
    remove_bg: bool = True,
    crop_to_fg: bool = True,
    target_size: int = 518,
    store_all_similarities: bool = False,
    tsv_path: Optional[str] = None,
):
    model.eval()

    preproc_dataset = SpecimenDataset(
        data_dir=".",
        transform=None,
        remove_bg=remove_bg,
        crop_to_fg=crop_to_fg,
        target_size=target_size,
        mask_erode_px=0,
        return_masks=False,
        metadata_file=None,
        view_filter=None,
        build_immediately=False,
    )

    results = []

    print("\n" + "=" * 60)
    print("ZERO-SHOT SPECIES PREDICTION")
    print("=" * 60)
    print(f"Memory Bank: {len(memory_bank.species_clusters)} known species")
    print(f"Test Images: {len(image_paths)}")
    print("=" * 60 + "\n")

    for img_path in image_paths:
        img_pil = Image.open(img_path).convert("RGB")
        processed_img, _ = preproc_dataset._preprocess_image(img_pil)

        if transform:
            img_tensor = transform(processed_img).unsqueeze(0).to(device)
        else:
            img_tensor = T.ToTensor()(processed_img).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = model.get_embeddings(img_tensor).cpu().numpy()[0]

        specimen_id = Path(img_path).stem

        # similarities to existing known species (for Bayesian fusion)
        all_sims = None
        if store_all_similarities:
            all_sims = {}
            emb_norm = float(np.linalg.norm(embedding) + 1e-12)
            for sp_id, cluster in memory_bank.species_clusters.items():
                if cluster.centroid is None:
                    continue
                cen = cluster.centroid
                cen_norm = float(np.linalg.norm(cen) + 1e-12)
                all_sims[sp_id] = float(np.dot(embedding, cen) / (emb_norm * cen_norm))

        # Unknown view for external images; keep field for future extension
        species_id, similarity, is_new = memory_bank.classify_specimen(
            embedding, specimen_id, img_path, return_scores=True, view_id="unknown"
        )

        result = {
            "specimen_id": specimen_id,
            "image_path": img_path,
            "predicted_species": species_id,
            "similarity_score": float(similarity),
            "is_new_species": is_new,
        }
        if all_sims is not None:
            result["similarity_scores"] = all_sims

        results.append(result)

        status = "NEW SPECIES" if is_new else "ASSIGNED"
        print(f"  {specimen_id}: Species {species_id} (sim={similarity:.4f}) [{status}]")

    output_path = os.path.join(output_dir, "zero_shot_predictions.json")
    with open(output_path, "w") as f:
        json.dump(
            {"results": results, "memory_bank_summary": memory_bank.get_summary()},
            f,
            indent=2,
        )

    print(f"\n✓ Results saved: {output_path}")

    if tsv_path is not None:
        _write_predictions_tsv(results, tsv_path, multi_view=False)
        print(f"✓ TSV saved: {tsv_path}")

    return results, memory_bank


def zero_shot_prediction_multi_view(
    model,
    specimen_to_images: Dict[str, List[str]],
    memory_bank: SpeciesMemoryBank,
    device: torch.device,
    output_dir: str,
    transform=None,
    remove_bg: bool = True,
    crop_to_fg: bool = True,
    target_size: int = 518,
    store_all_similarities: bool = False,
    tsv_path: Optional[str] = None,
):
    model.eval()

    preproc_dataset = SpecimenDataset(
        data_dir=".",
        transform=None,
        remove_bg=remove_bg,
        crop_to_fg=crop_to_fg,
        target_size=target_size,
        mask_erode_px=0,
        return_masks=False,
        metadata_file=None,
        view_filter=None,
        build_immediately=False,
    )

    results = []

    print("\n" + "=" * 60)
    print("MULTI-VIEW ZERO-SHOT SPECIES PREDICTION (SPECIMEN-LEVEL)")
    print("=" * 60)
    print(f"Memory Bank: {len(memory_bank.species_clusters)} known species")
    print(f"Specimens to classify: {len(specimen_to_images)}")
    print("=" * 60 + "\n")

    for specimen_id, img_paths in specimen_to_images.items():
        embeddings_per_view = []
        used_paths = []

        for img_path in img_paths:
            if not os.path.exists(img_path):
                print(f"  [WARN] Image not found for specimen {specimen_id}: {img_path}")
                continue

            img_pil = Image.open(img_path).convert("RGB")
            processed_img, _ = preproc_dataset._preprocess_image(img_pil)

            if transform is not None:
                img_tensor = transform(processed_img).unsqueeze(0).to(device)
            else:
                img_tensor = T.ToTensor()(processed_img).unsqueeze(0).to(device)

            with torch.no_grad():
                emb = model.get_embeddings(img_tensor).cpu().numpy()[0]

            embeddings_per_view.append(emb)
            used_paths.append(img_path)

        if not embeddings_per_view:
            print(f"  [WARN] No valid images for specimen {specimen_id}; skipping.")
            continue

        # MOSAIC / MULTI-VIEW EMBEDDING: average over all views
        specimen_embedding = np.mean(np.stack(embeddings_per_view, axis=0), axis=0)

        # similarities to existing known species (for Bayesian fusion)
        all_sims = None
        if store_all_similarities:
            all_sims = {}
            emb_norm = float(np.linalg.norm(specimen_embedding) + 1e-12)
            for sp_id, cluster in memory_bank.species_clusters.items():
                if cluster.centroid is None:
                    continue
                cen = cluster.centroid
                cen_norm = float(np.linalg.norm(cen) + 1e-12)
                all_sims[sp_id] = float(np.dot(specimen_embedding, cen) / (emb_norm * cen_norm))

        # We label this as a special "mosaic" view in the memory bank
        representative_path = used_paths[0]
        species_id, similarity, is_new = memory_bank.classify_specimen(
            specimen_embedding,
            specimen_id=specimen_id,
            image_path=representative_path,
            return_scores=True,
            view_id="mosaic",
        )

        result = {
            "specimen_id": specimen_id,
            "image_paths": used_paths,
            "predicted_species": species_id,
            "similarity_score": float(similarity),
            "is_new_species": is_new,
            "n_views_used": len(used_paths),
        }
        if all_sims is not None:
            result["similarity_scores"] = all_sims

        results.append(result)

        status = "NEW SPECIES" if is_new else "ASSIGNED"
        print(
            f"  {specimen_id} ({len(used_paths)} views): "
            f"Species {species_id} (sim={similarity:.4f}) [{status}]"
        )

    output_path = os.path.join(output_dir, "zero_shot_multi_view_predictions.json")
    with open(output_path, "w") as f:
        json.dump(
            {"results": results, "memory_bank_summary": memory_bank.get_summary()},
            f,
            indent=2,
        )

    print(f"\n✓ Results saved: {output_path}")

    if tsv_path is not None:
        _write_predictions_tsv(results, tsv_path, multi_view=True)
        print(f"✓ TSV saved: {tsv_path}")

    return results, memory_bank



# ============================================================================
# Visualization
# ============================================================================

def visualize_species_clusters(
    embeddings: np.ndarray,
    labels: np.ndarray,
    specimen_ids: List[str],
    output_path: str,
):
    """Visualize species clusters using t-SNE"""
    try:
        from sklearn.manifold import TSNE

        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=min(30, max(5, len(embeddings) - 1)),
        )
        embeddings_2d = tsne.fit_transform(embeddings)

        plt.figure(figsize=(12, 10))
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        for label, color in zip(unique_labels, colors):
            mask = labels == label
            plt.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[color],
                label=f"Species {label}",
                s=100,
                alpha=0.7,
                edgecolors="black",
                linewidths=1,
            )

        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.title("Species Embedding Space (t-SNE)", fontsize=16, fontweight="bold")
        plt.xlabel("t-SNE Dimension 1", fontsize=12)
        plt.ylabel("t-SNE Dimension 2", fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"✓ Visualization saved: {output_path}")

    except ImportError:
        print("Warning: scikit-learn required for t-SNE visualization")



def visualize_specimen_tsne_by_species(
    embeddings: np.ndarray,
    labels: np.ndarray,
    specimen_ids: List[str],
    split_names: List[str],
    output_path: str,
    max_points: int = 5000,
    perplexity: int = 30,
    random_state: int = 42,
    label_to_name: dict = None,
    max_species_legend: int = 50,
):
    """
    t-SNE of *specimens* (not images):
      - first aggregate embeddings per specimen (mean over its views)
      - color points by species label (deterministic unique color per species)
      - marker shape indicates split (train/val_known/val_novel/test_singleton/other)

    Notes:
      - For very large datasets, we downsample to `max_points` specimens for speed.
      - We do NOT create a legend for species (too many). We only legend split markers.
    """
    try:
        from sklearn.manifold import TSNE
        import matplotlib.colors as mcolors

        if embeddings is None or len(embeddings) == 0:
            print("[WARN] t-SNE specimen plot skipped: no embeddings.")
            return

        # Aggregate per specimen
        by_sid = {}
        for emb, lab, sid, sp in zip(embeddings, labels, specimen_ids, split_names):
            sid = str(sid)
            if sid not in by_sid:
                by_sid[sid] = {"embs": [], "lab": int(lab), "splits": set()}
            by_sid[sid]["embs"].append(emb)
            by_sid[sid]["splits"].add(str(sp))

        agg_emb = []
        agg_lab = []
        agg_split = []
        agg_sid = []

        # Decide a single split tag per specimen (priority order)
        def _pick_split(splits_set):
            prio = ["train", "val_known", "val_novel", "test_singleton", "other"]
            for p in prio:
                if p in splits_set:
                    return p
            # fallback
            return list(splits_set)[0] if splits_set else "other"

        for sid, d in by_sid.items():
            embs = np.stack(d["embs"], axis=0)
            agg_emb.append(embs.mean(axis=0))
            agg_lab.append(int(d["lab"]))
            agg_split.append(_pick_split(d["splits"]))
            agg_sid.append(sid)

        agg_emb = np.asarray(agg_emb, dtype=np.float32)
        agg_lab = np.asarray(agg_lab, dtype=np.int64)

        # Optional downsample for speed (deterministic)
        if max_points is not None and len(agg_emb) > int(max_points):
            rng = np.random.RandomState(42)
            keep = rng.choice(len(agg_emb), size=int(max_points), replace=False)
            agg_emb = agg_emb[keep]
            agg_lab = agg_lab[keep]
            agg_split = [agg_split[i] for i in keep]
            agg_sid = [agg_sid[i] for i in keep]
            print(f"[INFO] t-SNE specimen plot: downsampled to {len(agg_emb)} specimens for speed.")

        # Run t-SNE
        perplexity = min(30, max(5, len(agg_emb) - 1))
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        emb2d = tsne.fit_transform(agg_emb)

        coords = emb2d
        # Deterministic "unique" color per species via HSV hashing
        def _color_for_label(lab_int):
            # Hue from golden ratio
            h = (lab_int * 0.61803398875) % 1.0

            # Cycle lightness + saturation for stronger visual separation
            light_bands = [0.45, 0.55, 0.65]
            sat_bands   = [0.55, 0.75, 0.95]

            l = light_bands[lab_int % len(light_bands)]
            s = sat_bands[(lab_int // len(light_bands)) % len(sat_bands)]

            r, g, b = colorsys.hls_to_rgb(h, l, s)
            return (r, g, b)


#        def _color_for_label(lab_int):
#            h = ((lab_int * 0.61803398875) % 1.0)  # golden-ratio hashing into [0,1)
#            s = 0.75
#            v = 0.90
#            return mcolors.hsv_to_rgb((h, s, v))

        # --- Write species->color key (so you can decode colors even when legend would be huge) ---
        try:
            unique_labs = sorted(set(int(x) for x in agg_lab.tolist()))
        except Exception:
            unique_labs = sorted(set(int(x) for x in list(agg_lab)))

        key_path = os.path.splitext(output_path)[0] + "_color_key.tsv"
        try:
            with open(key_path, "w", encoding="utf-8") as f:
                f.write("label\tspecies\thex\tr\tg\tb\n")
                for lab in unique_labs:
                    rgb = _color_for_label(lab)
                    hx = mcolors.to_hex(rgb)
                    spname = ""
                    if label_to_name is not None:
                        spname = str(label_to_name.get(lab, label_to_name.get(str(lab), lab)))
                    f.write(
                        f"{lab}\t{spname}\t{hx}\t{rgb[0]:.6f}\t{rgb[1]:.6f}\t{rgb[2]:.6f}\n"
                    )
        except Exception as e:
            print(f"[WARN] Could not write TSNE color key: {e}")

        # Marker per split
        split_to_marker = {"train": "o", "val_known": "^", "val_novel": "s"}

        fig, ax = plt.subplots(figsize=(12, 9))

        # Plot each split separately so we can use different markers,
        # while keeping species colors consistent.
        for split_name in sorted(set(agg_split)):
            idxs = [i for i, s in enumerate(agg_split) if s == split_name]
            if not idxs:
                continue
            x = coords[idxs, 0]
            y = coords[idxs, 1]
            c = [_color_for_label(int(agg_lab[i])) for i in idxs]
            ax.scatter(
                x,
                y,
                s=60,
                c=c,
                marker=split_to_marker.get(split_name, "o"),
                alpha=0.9,
                edgecolors="none",
                label=split_name,
            )

        ax.set_title("Specimen Embedding Space (t-SNE) — colored by species", fontsize=18)
        ax.set_xlabel("t-SNE Dimension 1", fontsize=14)
        ax.set_ylabel("t-SNE Dimension 2", fontsize=14)
        ax.grid(True, alpha=0.25)

        # Split legend
        split_leg = plt.legend(title="Split", loc="upper right")

        # Optional: if there are only a few species, also add a species-color legend.
        # (For thousands of species, a legend would be unusable; use *_color_key.tsv instead.)
        if label_to_name is not None and len(unique_labs) <= max_species_legend:
            from matplotlib.lines import Line2D
            species_handles = []
            for lab in unique_labs:
                spname = str(label_to_name.get(lab, label_to_name.get(str(lab), lab)))
                species_handles.append(
                    Line2D(
                        [0], [0],
                        marker="o",
                        linestyle="",
                        label=spname,
                        markerfacecolor=_color_for_label(lab),
                        markeredgecolor="none",
                        markersize=6,
                    )
                )
            if split_leg is not None:
                plt.gca().add_artist(split_leg)
            plt.legend(
                handles=species_handles,
                title="Species (color)",
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                fontsize=6,
            )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✓ Specimen t-SNE saved: {output_path}")

    except ImportError:
        print("Warning: scikit-learn required for t-SNE visualization")
    except Exception as e:
        print(f"[WARN] Specimen t-SNE failed: {e}")


def visualize_masks_and_attention(
    model,
    dataset: SpecimenDataset,
    device: torch.device,
    output_dir: str,
    epoch: int,
    num_samples: int = 10,
):
    """
    Randomly sample a subset of images from the dataset and save:
      - preprocessed image,
      - foreground mask (if available),
      - patch-norm heatmap (proxy for attention),
      - image+heatmap overlay,
      - and a .npz with raw arrays.

    This uses the L2 norm of DINOv3 patch tokens (x_norm_patchtokens) as an
    importance measure, similar to the fallback in v82 when true attention
    is not directly exposed.
    """
    model.eval()

    if num_samples <= 0 or len(dataset) == 0:
        return

    epoch_dir = os.path.join(output_dir, "attn_viz", f"epoch_{epoch:03d}")
    os.makedirs(epoch_dir, exist_ok=True)

    # Choose random indices without replacement
    num_samples = min(num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), num_samples)

    print(f"  🔍 Visualizing patch-norm heatmaps for {num_samples} samples (epoch {epoch})")

    # Same transform as training
    transform = dataset.transform

    for rank, idx in enumerate(indices):
        sample = dataset.samples[idx]

        # Handle both (img_path, species_idx, specimen_id) and (img_path, species_idx, specimen_id, view_id)
        if len(sample) == 4:
            img_rel_path, species_idx, specimen_id, view_id = sample
        elif len(sample) == 3:
            img_rel_path, species_idx, specimen_id = sample
            view_id = "unknown"
        else:
            print(f"    [WARN] Unexpected sample format at idx {idx}: {sample}")
            continue

        # Resolve full path
        if os.path.isabs(img_rel_path):
            img_path = img_rel_path
        else:
            img_path = os.path.join(str(dataset.data_dir), img_rel_path)

        if not os.path.exists(img_path):
            print(f"    [WARN] Image not found: {img_path}")
            continue
        try:
            img_pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"    [WARN] Failed to open {img_path}: {e}")
            continue

        # Look up COCO polygons for this image, same logic as __getitem__
        seg_polys = None
        if getattr(dataset, "coco_seg_by_file", None):
            try:
                rel_key = str(Path(img_path).relative_to(dataset.data_dir)).replace(os.sep, "/")
            except ValueError:
                # Fallback: just the filename
                rel_key = os.path.basename(img_path)

            seg_polys = dataset.coco_seg_by_file.get(rel_key, None)
            if seg_polys is None and dataset.coco_mask_file is not None:
                print(f"    [WARN] No COCO mask found for key '{rel_key}'")

        # Preprocess exactly like training, but now with seg_polys
        processed_img, fg_mask = dataset._preprocess_image(
            img_pil,
            seg_polys=seg_polys,
        )


        # To tensor
        if transform is not None:
            img_tensor = transform(processed_img).unsqueeze(0).to(device)
        else:
            img_tensor = T.ToTensor()(processed_img).unsqueeze(0).to(device)

        # ---- Patch-token "attention" from x_norm_patchtokens ----
        with torch.no_grad():
            feats = model.backbone.forward_features(img_tensor)

        if not isinstance(feats, dict) or "x_norm_patchtokens" not in feats:
            print("    [WARN] forward_features did not return 'x_norm_patchtokens'; skipping sample.")
            continue

        patch_tokens = feats["x_norm_patchtokens"]  # [B, num_patches, dim]
        if patch_tokens.ndim != 3 or patch_tokens.shape[0] != 1:
            print(f"    [WARN] Unexpected patch token shape: {patch_tokens.shape}; skipping.")
            continue

        patch_tokens = patch_tokens[0]  # [num_patches, dim]
        # L2 norm per patch as importance score
        scores = torch.norm(patch_tokens, dim=-1).cpu().numpy()  # [num_patches]

        num_patches = scores.shape[0]
        grid_size = int(np.sqrt(num_patches))
        if grid_size * grid_size != num_patches:
            print(
                f"    [WARN] Non-square number of patches ({num_patches}); "
                f"cannot reshape to grid; skipping idx {idx}"
            )
            continue

        heatmap = scores.reshape(grid_size, grid_size)
        # Normalize 0-1
        hm_min, hm_max = heatmap.min(), heatmap.max()
        heatmap_norm = (heatmap - hm_min) / (hm_max - hm_min + 1e-8)

        # Upsample to match processed_img size
        proc_w, proc_h = processed_img.size
        heatmap_up = cv2.resize(
            heatmap_norm, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR
        )
        # If we have a foreground mask from preprocessing, suppress background
        if fg_mask is not None:
            # fg_mask is 0/1 and same resolution as processed_img
            if fg_mask.shape == heatmap_up.shape:
                heatmap_up = heatmap_up * fg_mask
            else:
                # Just in case shapes mismatch for some reason, we skip masking
                print(
                    f"    [WARN] fg_mask shape {fg_mask.shape} "
                    f"!= heatmap_up shape {heatmap_up.shape}; not masking heatmap."
                )

        # Colorize heatmap (viridis)
        cmap = plt.get_cmap("viridis")
        heatmap_color = (cmap(heatmap_up)[..., :3] * 255).astype(np.uint8)  # HxWx3

        # Convert processed image to numpy
        img_np = np.array(processed_img)

        # Overlay
        alpha = 0.4
        overlay = (
            img_np.astype(np.float32) * (1.0 - alpha)
            + heatmap_color.astype(np.float32) * alpha
        ).astype(np.uint8)

        # Base filename
        base_name = (
            f"idx{idx:05d}_sp{species_idx}_spec{str(specimen_id)}_view{str(view_id)}"
        )

        # Save processed image
        processed_out = os.path.join(epoch_dir, base_name + "_image.png")
        processed_img.save(processed_out)

        # Save foreground mask if available
        if fg_mask is not None:
            mask_uint8 = (fg_mask * 255).astype(np.uint8)
            mask_pil = Image.fromarray(mask_uint8, mode="L")
            mask_out = os.path.join(epoch_dir, base_name + "_mask.png")
            mask_pil.save(mask_out)
        else:
            mask_out = None

        # Save heatmap image
        heatmap_out = os.path.join(epoch_dir, base_name + "_patchnorm.png")
        heatmap_pil = Image.fromarray(heatmap_color)
        heatmap_pil.save(heatmap_out)

        # Save overlay
        overlay_out = os.path.join(epoch_dir, base_name + "_overlay.png")
        overlay_pil = Image.fromarray(overlay)
        overlay_pil.save(overlay_out)

        # Save raw arrays for later analysis
        arrays_out = os.path.join(epoch_dir, base_name + "_arrays.npz")
        np.savez(
            arrays_out,
            patch_scores=heatmap,
            patch_scores_norm=heatmap_norm,
            heatmap_up=heatmap_up,
            fg_mask=fg_mask if fg_mask is not None else None,
        )

        print(f"    Saved viz for idx {idx} -> {overlay_out}")


def plot_open_set_roc(scores_known, scores_novel, out_png):
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc

        scores_known = np.asarray(scores_known, dtype=np.float64)
        scores_novel = np.asarray(scores_novel, dtype=np.float64)

        y_true = np.concatenate([np.ones_like(scores_known), np.zeros_like(scores_novel)])
        y_score = np.concatenate([scores_known, scores_novel])

        fpr, tpr, thr = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6.5, 6))
        plt.plot(fpr, tpr, lw=2)
        plt.plot([0, 1], [0, 1], lw=1, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate (novel accepted as known)")
        plt.ylabel("True Positive Rate (known accepted as known)")
        plt.title(f"Open-set ROC (AUC={roc_auc:.3f})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close()

        return float(roc_auc)
    except Exception as e:
        print(f"[WARN] ROC plot failed: {e}")
        return None

# ============================================================================
# Main Training Script
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="DINOSAR-v2: Contrastive Species Learning")

    # Data
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory with images (with species subdirs OR flat with --metadata-file)",
    )
    parser.add_argument(
        "--metadata-file",
        type=str,
        default=None,
        help="Path to CSV/TSV/JSON file with species/specimen assignments",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output_dinosar_v2",
        help="Output directory",
    )
    parser.add_argument(
        "--view-filter",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of view IDs to include (e.g. H D P). If set, only these views are used for training/eval.",
    )

    # Model
    parser.add_argument(
        "--dinov3-model",
        type=str,
        default="dinov3_vitb14",
        choices=["dinov3_vits14", "dinov3_vits16", "dinov3_vitb14", "dinov3_vitl14", "dinov3_vitg14"],
        help="DINOv3 model architecture (must match checkpoint)",
    )
    parser.add_argument(
        "--dinov3-local-ckpt",
        type=str,
        default=None,
        help="Path to local DINOv3 checkpoint",
    )
    parser.add_argument(
        "--dinov3-checkpoint",
        type=str,
        default=None,
        help="[DEPRECATED: use --dinov3-local-ckpt] Path to DINOv3 checkpoint",
    )
    parser.add_argument(
        "--projection-dim",
        type=int,
        default=256,
        help="Dimension of contrastive embedding",
    )
    # Backbone control (default: freeze backbone + use CLS token)
    # NOTE: The original script used store_true with default=True, which made it
    # impossible to disable these. We provide explicit --no-* flags instead.
    parser.add_argument(
        "--freeze-backbone",
        dest="freeze_backbone",
        action="store_true",
        default=True,
        help="Freeze DINOv3 backbone weights (default: enabled).",
    )
    parser.add_argument(
        "--no-freeze-backbone",
        dest="freeze_backbone",
        action="store_false",
        help="Do NOT freeze DINOv3 backbone weights (fine-tune backbone).",
    )

    parser.add_argument(
        "--use-cls-token",
        dest="use_cls_token",
        action="store_true",
        default=True,
        help="Use [CLS] token for embedding (default: enabled).",
    )
    parser.add_argument(
        "--no-use-cls-token",
        dest="use_cls_token",
        action="store_false",
        help="Do NOT use [CLS]; mean-pool patch tokens instead.",
    )

    # Training
    parser.add_argument(
        "--loss",
        type=str,
        default="supcon",
        choices=["supcon", "triplet"],
        help="Contrastive loss function",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset splits and sampling.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "DataLoader worker processes. "
            "Use 0 to avoid per-worker rembg/COCO-mask session bloat that can look like a memory leak."
        ),
    )

    # Open-set / zero-shot calibration
    parser.add_argument(
        "--species-holdout-fraction",
        type=float,
        default=0.0,
        help=(
            "If >0, hold out this fraction of species entirely (all specimens) for NOVEL validation. "
            "This mimics 'new species' at calibration time."
        ),
    )
    parser.add_argument(
        "--open-set-eval",
        action="store_true",
        help="Run open-set retrieval + threshold calibration at eval epochs.",
    )
    parser.add_argument(
        "--open-set-topk",
        type=int,
        nargs="+",
        default=[1, 5],
        help="Top-k values for specimen-level kNN retrieval (val -> train).",
    )
    parser.add_argument(
        "--open-set-far-targets",
        type=float,
        nargs="+",
        default=[0.01, 0.05],
        help="False-accept-rate targets for recommended 'known species' threshold.",
    )
    parser.add_argument(
        "--open-set-eval-every",
        type=int,
        default=1,
        help="Run open-set evaluation every N eval epochs (to save time).",
    )





    # Split strategy / data hygiene for open-set clustering
    parser.add_argument(
        "--split-strategy",
        type=str,
        default="global",
        choices=["global", "per_species"],
        help=(
            "How to create the KNOWN validation split. "
            "'global' uses --val-fraction over all candidate specimens. "
            "'per_species' does per-species holdout (leave-one-out-ish) using --per-species-val-fraction."
        ),
    )
    parser.add_argument(
        "--per-species-val-fraction",
        type=float,
        default=0.10,
        help=(
            "Only used when --split-strategy=per_species. "
            "Fraction of specimens per species to place into val_known (min 1), "
            "while keeping at least --min-train-specimens-per-species in train."
        ),
    )
    parser.add_argument(
        "--min-specimens-for-val",
        type=int,
        default=3,
        help=(
            "Only used when --split-strategy=per_species. "
            "Species must have at least this many specimens to contribute to val_known."
        ),
    )
    parser.add_argument(
        "--min-train-specimens-per-species",
        type=int,
        default=2,
        help=(
            "Only used when --split-strategy=per_species. "
            "Keep at least this many specimens per species in TRAIN."
        ),
    )
    parser.add_argument(
        "--exclude-singletons",
        action="store_true",
        help=(
            "Exclude species with only 1 specimen from training/validation (recommended for open-set). "
            "If --export-split-tsvs is enabled, these go to test_singletons.tsv."
        ),
    )
    parser.add_argument(
        "--export-split-tsvs",
        action="store_true",
        help=(
            "Export train.tsv, val_known.tsv, val_novel.tsv, test_singletons.tsv into --output-dir "
            "based on the actual split used in this run."
        ),
    )
    parser.add_argument(
        "--memory-bank-scope",
        type=str,
        default="train",
        choices=["train", "train+val_known", "full"],
        help=(
            "What to include when building memory_bank.json. "
            "For open-set evaluation, 'train' is recommended (avoids leaking held-out species)."
        ),
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.0,
        help=(
            "If >0, reserve this fraction of UNIQUE specimens for validation "
            "(specimen-disjoint split). 0.0 = no val split (legacy behaviour)."
        ),
    )    
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for AdamW optimizer",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="Temperature for SupCon loss",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.3,
        help="Margin for triplet loss",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="Evaluate every N epochs",
    )

    # Preprocessing
    parser.add_argument(
        "--no-remove-bg",
        action="store_true",
        help="Do not remove background with rembg",
    )
    parser.add_argument(
        "--use-attn-mask",
        action="store_true",
        help="Use DINOv3 attention for foreground mask (not wired into training loop)",
    )
    parser.add_argument(
        "--attn-mask-threshold",
        type=float,
        default=0.25,
        help="Attention threshold for foreground mask",
    )
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Do not crop to foreground",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=518,
        help="Target image size (square canvas)",
    )
    parser.add_argument(
        "--mask-erode-px",
        type=int,
        default=0,
        help="Erode foreground mask by N pixels to remove halos",
    )
    parser.add_argument(
        "--mask-in-loss",
        action="store_true",
        help="Return masks from dataset (for potential mask-weighted losses)",
    )

    # Zero-shot prediction
    parser.add_argument(
        "--predict",
        type=str,
        nargs="+",
        help="Paths to images for zero-shot prediction",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.85,
        help="Similarity threshold for species assignment",
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default=None,
        help="Load trained model checkpoint",
    )

    # Classifier / SupCon control
    parser.add_argument(
        "--use-classifier-head",
        action="store_true",
        help="Add a species classifier head with cross-entropy loss on top of embeddings.",
    )
    parser.add_argument(
        "--supcon-label-mode",
        type=str,
        default="species",
        choices=["species", "specimen", "both"],
        help=(
            "What labels to use for the contrastive loss: "
            "'species', 'specimen', or 'both' (two SupCon terms)."
        ),
    )
    parser.add_argument(
        "--supcon-weight",
        type=float,
        default=1.0,
        help="Weight for the contrastive loss term.",
    )
    parser.add_argument(
        "--ce-weight",
        type=float,
        default=1.0,
        help="Weight for the cross-entropy loss (if classifier head is used).",
    )

    parser.add_argument(
        "--predict-multi-view-json",
        type=str,
        default=None,
        help=(
            "Path to a JSON file mapping specimen_id to list of image paths "
            "for multi-view zero-shot prediction. "
            "Example structure: {\"spec1\": [\"img1.jpg\", \"img2.jpg\"], ...}"
        ),
    )
    parser.add_argument(
        '--viz-attn-num-samples',
        type=int,
        default=0,
        help=(
            'Number of random specimens/images to visualize with mask + attention '
            'at each evaluation epoch (0 = disable visualization).'
        ),
    )

    parser.add_argument(
        "--coco-mask-file",
        type=str,
        default=None,
        help="Path to COCO JSON with precomputed masks (foreground or sclerite).",
    )
    parser.add_argument(
        "--coco-mask-category",
        type=str,
        default="foreground",
        help="Category name in COCO mask file to use as mask "
             '(e.g. "foreground", "propodeum", "head").',
    )

    parser.add_argument("--unfreeze-last-n-blocks", type=int, default=0,
                        help="If freeze_backbone, unfreeze last N transformer blocks")
    parser.add_argument("--backbone-lr", type=float, default=1e-5,
                        help="LR for unfrozen backbone blocks")
    parser.add_argument("--prefetch-factor", type=int, default=2,
                        help="DataLoader prefetch_factor (only used when num_workers > 0).")
    
    parser.add_argument(
        "--singletons-as-novel",
        action="store_true",
        help=(
            "If set (recommended with --exclude-singletons), add singleton-species specimens "
            "to the NOVEL split used for open-set evaluation (val_novel). "
            "This increases novel test data without leaking into training/val_known."
        ),
    )

    # --- Prediction bookkeeping / outputs ---
    parser.add_argument("--memory-bank-in", type=str, default=None,
        help="Path to an existing memory_bank.json to LOAD in prediction mode. "
             "Default: <output-dir>/memory_bank.json")

    parser.add_argument("--memory-bank-out", type=str, default=None,
        help="Where to SAVE the memory bank after prediction. "
             "Default: <output-dir>/memory_bank_pred_<timestamp>.json (no overwrite)")

    parser.add_argument("--overwrite-memory-bank", action="store_true",
        help="Allow overwriting memory-bank-in in prediction mode (unsafe).")

    parser.add_argument("--no-save-memory-bank", action="store_true",
        help="Do not write any memory bank in prediction mode (read-only).")

    parser.add_argument("--predictions-tsv", type=str, default=None,
        help="Write per-image predictions TSV. Default: <output-dir>/zero_shot_predictions.tsv")

    parser.add_argument("--predictions-mv-tsv", type=str, default=None,
        help="Write multi-view predictions TSV. Default: <output-dir>/zero_shot_multi_view_predictions.tsv")

    parser.add_argument("--store-all-similarities", action="store_true",
        help="Store similarity-to-each-known-species in prediction JSON (bigger files, needed for Bayesian fusion).")

    # ── v24: Multi-Modal Auxiliary Training ──────────────────────────
    parser.add_argument("--trait-tsv", type=str, default=None,
        help="TSV file with morphological trait measurements keyed by specimen_id. "
             "Columns starting with 'cat_' → categorical (BCE), 'count_'/'n_' → meristic (Poisson), "
             "other numeric → continuous (MSE). Specimens without traits skip trait loss.")
    parser.add_argument("--trait-specimen-id-col", type=str, default="specimen_id",
        help="Column name for specimen ID in trait TSV (default: specimen_id).")
    parser.add_argument("--trait-weight", type=float, default=0.5,
        help="Weight for trait prediction losses (continuous MSE + meristic Poisson + categorical BCE).")

    parser.add_argument("--dna-fasta", type=str, default=None,
        help="FASTA file with COI barcode sequences keyed by specimen_id (header first token). "
             "Enables DNA encoder training with species CE + cross-modal alignment.")
    parser.add_argument("--dna-max-seq-len", type=int, default=700,
        help="Max DNA sequence length for the 1D CNN encoder (default: 700bp, covers standard COI-5P).")
    parser.add_argument("--dna-weight", type=float, default=0.5,
        help="Weight for DNA species classification CE loss.")

    parser.add_argument("--morph-tsv", type=str, default=None,
        help="TSV file with morphometric feature vectors (continuous measurements) keyed by specimen_id. "
             "Enables morph encoder training with cross-modal vision alignment.")
    parser.add_argument("--morph-specimen-id-col", type=str, default="specimen_id",
        help="Column name for specimen ID in morph TSV (default: specimen_id).")

    parser.add_argument("--xmodal-weight", type=float, default=0.3,
        help="Weight for cross-modal alignment losses (vision↔DNA, vision↔morph).")
    parser.add_argument("--xmodal-temperature", type=float, default=0.1,
        help="Temperature for cross-modal InfoNCE loss (default: 0.1).")

    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Argument hygiene: avoid conflicting split flags
    # ------------------------------------------------------------------ #
    if getattr(args, "split_strategy", "global") != "global" and getattr(args, "val_fraction", 0.0) > 0.0:
        print(
            "[WARN] You passed --val-fraction but --split-strategy is not 'global'. "
            "Ignoring --val-fraction and using the split strategy parameters instead."
        )
        args.val_fraction = 0.0


    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # ------------------------------------------------------------------ #
    # Zero-shot prediction modes
    # ------------------------------------------------------------------ #
    if args.predict or args.predict_multi_view_json:
        print("\n" + "=" * 60)
        print("ZERO-SHOT / MULTI-VIEW PREDICTION MODE")
        print("=" * 60)

        checkpoint_path = args.dinov3_local_ckpt or args.dinov3_checkpoint
        model = DINOSAR_v2(
            dinov3_model=args.dinov3_model,
            dinov3_checkpoint=checkpoint_path,
            projection_dim=args.projection_dim,
            freeze_backbone=args.freeze_backbone,
            unfreeze_last_n_blocks=getattr(args, "unfreeze_last_n_blocks", 0),  # <-- ADD (safe)
            use_cls_token=args.use_cls_token,
            use_classifier_head=False,  # we use embeddings + memory bank for zero-shot
            num_species=None,
        ).to(device)

        if args.load_checkpoint:
            print(f"Loading checkpoint: {args.load_checkpoint}")
            checkpoint = torch.load(args.load_checkpoint, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            print("✓ Checkpoint loaded")

        # Load or initialize memory bank
        # -----------------------------
        # Load or initialize memory bank (prediction mode)
        # -----------------------------
        memory_bank_in = args.memory_bank_in or os.path.join(args.output_dir, "memory_bank.json")

        # Decide output path (avoid overwriting by default)
        memory_bank_out = None
        if not args.no_save_memory_bank:
            if args.memory_bank_out:
                memory_bank_out = args.memory_bank_out
            elif args.overwrite_memory_bank:
                memory_bank_out = memory_bank_in
            else:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                memory_bank_out = os.path.join(args.output_dir, f"memory_bank_pred_{ts}.json")

        if os.path.exists(memory_bank_in):
            print(f"Loading memory bank: {memory_bank_in}")
            with open(memory_bank_in, "r") as f:
                bank_data = json.load(f)

            memory_bank = SpeciesMemoryBank(similarity_threshold=args.similarity_threshold)
            for sp_id, sp_data in bank_data.get("species", {}).items():
                emb_list = sp_data.get("embeddings", [])
                embeddings = [np.array(emb) for emb in emb_list]
                specimen_ids = sp_data.get("specimen_ids", [])
                image_paths = sp_data.get("image_paths", [])
                view_ids = sp_data.get("view_ids", ["unknown"] * len(embeddings))

                cluster = SpeciesCluster(
                    species_id=sp_id,
                    embeddings=embeddings,
                    specimen_ids=specimen_ids,
                    image_paths=image_paths,
                    view_ids=view_ids,
                )

                # Prefer stored centroid if present (faster, consistent)
                if sp_data.get("centroid") is not None:
                    cluster.centroid = np.array(sp_data["centroid"])
                else:
                    cluster.update_centroid()

                memory_bank.species_clusters[sp_id] = cluster

            # Robust next_species_id init: only consider single-letter A..Z cluster IDs
            letter_ids = [
                k for k in memory_bank.species_clusters.keys()
                if isinstance(k, str) and len(k) == 1 and k.isalpha() and k.isupper()
            ]
            memory_bank.next_species_id = ord(max(letter_ids)) + 1 if letter_ids else ord("A")

        else:
            print(f"[WARN] No memory bank found at: {memory_bank_in} (starting fresh)")
            memory_bank = SpeciesMemoryBank(similarity_threshold=args.similarity_threshold)


        # Default TSV outputs (always written unless you add a future --no-predictions-tsv flag)
        # Default TSV outputs
        tsv_path = args.predictions_tsv or os.path.join(args.output_dir, "zero_shot_predictions.tsv")
        tsv_mv_path = args.predictions_mv_tsv or os.path.join(args.output_dir, "zero_shot_multi_view_predictions.tsv")

        # --- Mode 1: per-image prediction (existing --predict) ---
        if args.predict:
            results, memory_bank = zero_shot_prediction(
                model=model,
                image_paths=args.predict,
                memory_bank=memory_bank,
                device=device,
                output_dir=args.output_dir,
                transform=transform,
                remove_bg=not args.no_remove_bg,
                crop_to_fg=not args.no_crop,
                target_size=args.target_size,
                store_all_similarities=args.store_all_similarities,
                tsv_path=tsv_path,
            )

        # --- Mode 2: specimen-level multi-view prediction (new) ---
        if args.predict_multi_view_json:
            with open(args.predict_multi_view_json, "r") as f:
                specimen_to_images = json.load(f)
            if not isinstance(specimen_to_images, dict):
                raise ValueError(
                    "--predict-multi-view-json must be a JSON object mapping specimen_id -> list of image paths."
                )

            results_mv, memory_bank = zero_shot_prediction_multi_view(
                model=model,
                specimen_to_images=specimen_to_images,
                memory_bank=memory_bank,
                device=device,
                output_dir=args.output_dir,
                transform=transform,
                remove_bg=not args.no_remove_bg,
                crop_to_fg=not args.no_crop,
                target_size=args.target_size,
                store_all_similarities=args.store_all_similarities,
                tsv_path=tsv_mv_path,
            )

        # Save updated memory bank
        bank_export = {"species": {}}
        for sp_id, cluster in memory_bank.species_clusters.items():
            bank_export["species"][sp_id] = {
                "embeddings": [emb.tolist() for emb in cluster.embeddings],
                "specimen_ids": cluster.specimen_ids,
                "image_paths": cluster.image_paths,
                "view_ids": cluster.view_ids,
                "centroid": cluster.centroid.tolist() if cluster.centroid is not None else None,
            }

        if memory_bank_out is not None:
            with open(memory_bank_out, "w") as f:
                json.dump(bank_export, f, indent=2)
            print(f"\n✓ Memory bank saved: {memory_bank_out}")
        else:
            print("\n(i) Memory bank not saved (--no-save-memory-bank).")

        print(f"\nFinal species count: {len(memory_bank.species_clusters)}")
        return

    # ------------------------------------------------------------------ #
    # Training mode
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("TRAINING MODE")
    print("=" * 60)

    print("\n📋 Exporting run configuration...")
    config_dict = {
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "hyperparameters": vars(args),
    }

    config_path = os.path.join(args.output_dir, "run_metadata.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    print(f"✓ Configuration saved: {config_path}")

    # Dataset
    print("\nLoading dataset...")
#    dataset = SpecimenDataset(
#        data_dir=args.data_dir,
#        transform=transform,
#        # Use rembg whenever available, unless the user explicitly disables it.
#        remove_bg=not args.no_remove_bg,
#        crop_to_fg=not args.no_crop,
#        target_size=args.target_size,
#        mask_erode_px=args.mask_erode_px,
#        return_masks=args.mask_in_loss,  # Only return masks if using them in loss
#        metadata_file=args.metadata_file,  # Pass metadata file if provided
#    )
    dataset = SpecimenDataset(
        data_dir=args.data_dir,
        transform=transform,
        remove_bg=not args.no_remove_bg,
        crop_to_fg=not args.no_crop,
        target_size=args.target_size,
        mask_erode_px=args.mask_erode_px,
        return_masks=args.mask_in_loss,
        metadata_file=args.metadata_file,
        coco_mask_file=args.coco_mask_file,
        coco_mask_category=args.coco_mask_category,
    )
    
    if args.metadata_file:
        print(f"✓ Using metadata file: {args.metadata_file}")
    else:
        print(f"✓ Using directory structure: {args.data_dir}")

    if args.use_attn_mask:
        print("✓ DINOv3 attention-based masking flag set (not wired into _preprocess_image).")

    if args.mask_erode_px > 0:
        print(f"✓ Mask erosion enabled: {args.mask_erode_px} pixels")

    if args.mask_in_loss:
        print("✓ Masks will be returned by the dataset (mask_in_loss enabled)")

    # Model
    checkpoint_path = args.dinov3_local_ckpt or args.dinov3_checkpoint
    num_species = len(dataset.species_to_idx)
    print(f"Detected {num_species} species in dataset.")

    model = DINOSAR_v2(
        dinov3_model=args.dinov3_model,
        dinov3_checkpoint=checkpoint_path,
        projection_dim=args.projection_dim,
        freeze_backbone=args.freeze_backbone,
        unfreeze_last_n_blocks=getattr(args, "unfreeze_last_n_blocks", 0),  # <-- ADD (safe)
        use_cls_token=args.use_cls_token,
        use_classifier_head=args.use_classifier_head,
        num_species=num_species if args.use_classifier_head else None,
    ).to(device)
    # --- DEBUG: confirm unfreezing actually happened ---
    n_trainable_bb = sum(p.requires_grad for p in model.backbone.parameters())
    n_total_bb = sum(1 for _ in model.backbone.parameters())
    print(f"[DEBUG] Trainable backbone params: {n_trainable_bb}/{n_total_bb} "
          f"(unfreeze_last_n_blocks={getattr(args,'unfreeze_last_n_blocks',0)})")

    if args.load_checkpoint:
        print(f"Loading training checkpoint: {args.load_checkpoint}")
        ckpt = torch.load(args.load_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print("✓ Training checkpoint loaded")

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    
    # ------------------------------------------------------------------ #
    # DataLoaders: full, train, val_known, (optional) val_novel (species-holdout)
    # ------------------------------------------------------------------ #
    from torch.utils.data import Subset
    from collections import defaultdict
    import random

    # IMPORTANT: keep ONE dataset object to avoid label-map mismatches
    full_dataset = dataset

    split_seed = int(getattr(args, "seed", 42))
    rng = random.Random(split_seed)

    # Full loader: used for final memory bank (and optional attention viz)
    full_loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )
    if args.num_workers > 0:
        # prefetch_factor is only valid when num_workers > 0
        full_loader_kwargs["prefetch_factor"] = args.prefetch_factor

    full_loader = DataLoader(full_dataset, **full_loader_kwargs)

        # Build maps from the FULL dataset samples list
    # NOTE: Real-world AntWeb/GBIF data can contain occasional specimen_id conflicts where the
    # same specimen_id appears under >1 species label (mis-IDs / synonymy / data errors).
    # For open-set learning, we MUST NOT mix species labels within a single specimen.
    #
    # Policy: DROP conflicted specimens from splitting/training/eval (default, safest).
    # We also write a small report to output_dir for auditing.
    spec_to_full_indices = defaultdict(list)     # specimen_id -> [full_idx, ...]
    specimen_to_species_set = defaultdict(set)   # specimen_id -> {species_idx, ...}

    for full_i, (_, sp_idx, specimen_id, _) in enumerate(full_dataset.samples):
        sid = str(specimen_id)
        sp = int(sp_idx)
        spec_to_full_indices[sid].append(full_i)
        specimen_to_species_set[sid].add(sp)

    conflicted_specimens = sorted([sid for sid, sps in specimen_to_species_set.items() if len(sps) > 1])
    if conflicted_specimens:
        print("\n[WARN] Found %d specimen_id conflicts (same specimen_id mapped to multiple species). "
              "These specimens will be DROPPED from splits/training/eval." % len(conflicted_specimens))
        # Write a report for transparency
        try:
            conflict_path = os.path.join(args.output_dir, "conflicted_specimens.tsv")
            with open(conflict_path, "w") as cf:
                cf.write("specimen_id\tspecies_indices\tn_images\n")
                for sid in conflicted_specimens:
                    sps = sorted(list(specimen_to_species_set[sid]))
                    cf.write("%s\t%s\t%d\n" % (sid, ",".join([str(x) for x in sps]), len(spec_to_full_indices.get(sid, []))))
            print("  Wrote conflict report:", conflict_path)
        except Exception as e:
            print("  [WARN] Could not write conflict report:", e)

    # Rebuild clean maps excluding conflicted specimens
    species_to_specimens = defaultdict(set)      # species_idx  -> {specimen_id, ...}
    specimen_to_species = {}                     # specimen_id -> species_idx

    for sid, idxs in spec_to_full_indices.items():
        if sid in set(conflicted_specimens):
            continue
        # exactly 1 species
        sp = list(specimen_to_species_set[sid])[0]
        specimen_to_species[sid] = sp
        species_to_specimens[sp].add(sid)

    all_specimens = [sid for sid in spec_to_full_indices.keys() if sid not in set(conflicted_specimens)]
    all_species = sorted(species_to_specimens.keys())

    # --------------------------------------------

    # --------------------------------------------
    # Optional SPECIES-HOLDOUT split for open-set calibration
    # val_novel: species not seen during training (held out entirely)
    # val_known: specimen-disjoint within the remaining training species
    # --------------------------------------------

    # Identify singleton species (only 1 specimen) for optional test holdout
    singleton_species = set()
    singleton_specimens = set()
    for sp_idx, sp_specimens in species_to_specimens.items():
        if len(sp_specimens) == 1:
            singleton_species.add(sp_idx)
            singleton_specimens |= set(sp_specimens)

    # Choose holdout species from non-singletons (so singletons can be reserved for test)
    holdout_species = set()
    all_species = list(species_to_specimens.keys())
    eligible_for_holdout = [sp for sp in all_species if sp not in singleton_species]
    rng.shuffle(eligible_for_holdout)

    if args.species_holdout_fraction > 0.0 and len(eligible_for_holdout) >= 3:
        n_holdout = int(round(len(eligible_for_holdout) * args.species_holdout_fraction))
        n_holdout = max(1, min(n_holdout, len(eligible_for_holdout) - 1))
        holdout_species = set(eligible_for_holdout[:n_holdout])
        print(f"\nCreating NOVEL species-holdout validation split (species_holdout_fraction={args.species_holdout_fraction:.2f})")
        print(f"  Holdout species: {len(holdout_species)} of {len(all_species)} total species")

    holdout_specimens = set()
    for sp in holdout_species:
        holdout_specimens |= set(species_to_specimens[sp])

    # Candidate specimens for training/known-val (exclude holdout; optionally exclude singletons)
    candidate_specimens = set(all_specimens) - holdout_specimens
    if args.exclude_singletons:
        candidate_specimens -= singleton_specimens
        if singleton_specimens:
            print(f"  Excluding singleton species from training/val: {len(singleton_specimens)} specimens ({len(singleton_species)} species)")

    candidate_specimens = list(candidate_specimens)
    rng.shuffle(candidate_specimens)

    # --------------------------------------------
    # KNOWN validation split inside candidate set
    # --------------------------------------------
    train_specimens = set(candidate_specimens)
    val_specimens = set()

    if args.split_strategy == "per_species" and len(candidate_specimens) >= 2:
        train_specimens = set()
        val_specimens = set()

        per_species_fraction = float(getattr(args, "per_species_val_fraction", 0.10))
        min_for_val = int(getattr(args, "min_specimens_for_val", 3))
        min_train_keep = int(getattr(args, "min_train_specimens_per_species", 2))

        print(f"\nCreating per-species KNOWN train/val split (per_species_val_fraction={per_species_fraction:.2f}, min_for_val={min_for_val}, min_train_keep={min_train_keep})")

        # Build species -> specimen list for candidate species only
        for sp_idx, sp_specimens in species_to_specimens.items():
            if sp_idx in holdout_species:
                continue
            # optionally skip singleton species entirely
            if args.exclude_singletons and sp_idx in singleton_species:
                continue

            sp_sids = list(sp_specimens)
            # keep only candidate specimens (not holdout)
            sp_sids = [sid for sid in sp_sids if sid in set(candidate_specimens)]
            if not sp_sids:
                continue

            rng.shuffle(sp_sids)
            n = len(sp_sids)
            if n < min_for_val:
                train_specimens |= set(sp_sids)
                continue

            n_val = max(1, int(round(n * per_species_fraction)))
            # keep at least min_train_keep in train
            n_val = min(n_val, max(0, n - min_train_keep))
            if n_val <= 0:
                train_specimens |= set(sp_sids)
                continue

            val_specimens |= set(sp_sids[:n_val])
            train_specimens |= set(sp_sids[n_val:])

        print(f"  Train specimens: {len(train_specimens)}")
        print(f"  Val_known specimens: {len(val_specimens)}")

    elif args.val_fraction > 0.0 and len(candidate_specimens) >= 2:
        print(f"\nCreating specimen-disjoint KNOWN train/val split (val_fraction={args.val_fraction:.2f})")

        n_val_spec = max(1, int(round(len(candidate_specimens) * args.val_fraction)))
        val_specimens = set(candidate_specimens[:n_val_spec])
        train_specimens = set(candidate_specimens[n_val_spec:])

        # ------------------------------------------------------------------ #
        # SPECIES-SAFETY FOR CLASSIFIER HEAD:
        # Ensure every TRAIN species has >= 1 specimen in TRAIN
        # ------------------------------------------------------------------ #
        moved = 0
        for sp_idx, sp_specimens in species_to_specimens.items():
            if sp_idx in holdout_species:
                continue
            if args.exclude_singletons and sp_idx in singleton_species:
                continue

            if train_specimens.isdisjoint(sp_specimens):
                candidates = list(val_specimens.intersection(sp_specimens))
                if candidates:
                    sid = candidates[0]
                    val_specimens.remove(sid)
                    train_specimens.add(sid)
                    moved += 1

        if moved:
            print(f"  (Species-safety) moved {moved} specimens from VAL -> TRAIN to cover all species in training.")

        print(f"  Train specimens: {len(train_specimens)}")
        print(f"  Val_known specimens:   {len(val_specimens)}")

    else:
        # No known-val split; use all candidates for training/eval
        if args.val_fraction > 0.0:
            print("\n[WARN] Not enough candidate specimens for val split; using all candidates for training/eval.")
        train_specimens = set(candidate_specimens)
        val_specimens = set()

    # Build indices
    train_indices = [i for sid in train_specimens for i in spec_to_full_indices[sid]]
    val_indices = [i for sid in val_specimens for i in spec_to_full_indices[sid]]

    print(
        f"  Train images: {len(train_indices)}\n"
        f"  Val_known images: {len(val_indices)}"
    )

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices) if len(val_indices) > 0 else None

    # Optional export of split TSVs for reproducibility
    if args.export_split_tsvs:
        import csv

        def _write_split_tsv(name, indices):
            out_path = os.path.join(args.output_dir, name)
            if os.path.exists(out_path):
                print(f"[SAFETY] Refusing to overwrite existing split TSV: {out_path}")
                return
            with open(out_path, "x", newline="", encoding="utf-8") as f:
                w = csv.writer(f, delimiter="	")
                w.writerow(["image_path", "species_id", "specimen_id", "view_id"])
                for full_i in indices:
                    img_path, sp_idx, sid, view_id = full_dataset.samples[full_i]
                    sp_name = full_dataset.idx_to_species[int(sp_idx)] if hasattr(full_dataset, "idx_to_species") else str(sp_idx)
                    w.writerow([str(img_path), str(sp_name), str(sid), str(view_id)])
            print(f"✓ Wrote {name}: {out_path}")

        # Index lists for novel + singleton exports
        # NOVEL specimens: species-holdout (+ optional singleton species) with leakage protection
        novel_specimens_for_eval = set(holdout_specimens)
        if getattr(args, "singletons_as_novel", False):
            if not args.exclude_singletons:
                print(
                    "[WARN] --singletons-as-novel is set without --exclude-singletons. "
                    "Any singleton specimens that ended up in TRAIN/VAL_KNOWN will be excluded from NOVEL to avoid leakage."
                )
            novel_specimens_for_eval |= set(singleton_specimens)

        # Safety: ensure NOVEL is disjoint from TRAIN/VAL_KNOWN
        leaked = (novel_specimens_for_eval & train_specimens) | (novel_specimens_for_eval & val_specimens)
        if leaked:
            novel_specimens_for_eval -= leaked
            print(f"[SAFETY] Removed {len(leaked)} NOVEL specimens that overlapped train/val_known.")

        novel_indices = [i for sid in novel_specimens_for_eval for i in spec_to_full_indices[sid]]

        # Keep exporting singletons separately too (optional, but useful for debugging)
        singleton_indices = [i for sid in singleton_specimens for i in spec_to_full_indices[sid]]

        #novel_indices = [i for sid in holdout_specimens for i in spec_to_full_indices[sid]]
        #singleton_indices = [i for sid in singleton_specimens for i in spec_to_full_indices[sid]]

        _write_split_tsv("train.tsv", train_indices)
        _write_split_tsv("val_known.tsv", val_indices)
        _write_split_tsv("val_novel.tsv", novel_indices)
        _write_split_tsv("test_singletons.tsv", singleton_indices)

    # Novel (species-holdout) dataset/loader
    # Novel (species-holdout) dataset/loader
    # Optionally add singleton-species specimens as extra NOVEL test data via --singletons-as-novel.
    val_novel_dataset = None
    novel_specimens_for_eval = set(holdout_specimens)
    if getattr(args, "singletons_as_novel", False):
        novel_specimens_for_eval |= set(singleton_specimens)

    # Safety: ensure NOVEL is disjoint from TRAIN/VAL_KNOWN (avoid accidental leakage)
    novel_specimens_for_eval -= (set(train_specimens) | set(val_specimens))

    if len(novel_specimens_for_eval) > 0:
        novel_indices = [i for sid in novel_specimens_for_eval for i in spec_to_full_indices[sid]]
        val_novel_dataset = Subset(full_dataset, novel_indices)
        tag = " (includes singleton species)" if getattr(args, "singletons_as_novel", False) else ""
        print(f"  Val_novel images: {len(novel_indices)} from {len(novel_specimens_for_eval)} specimens{tag}")

    #val_novel_dataset = None
    #if len(holdout_specimens) > 0:
    #    novel_indices = [i for sid in holdout_specimens for i in spec_to_full_indices[sid]]
    #    val_novel_dataset = Subset(full_dataset, novel_indices)

    # A non-shuffled loader over TRAIN (needed for retrieval eval + centroids)
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        drop_last=False,
    )

    # ------------------------------------------------------------------ #
    
    # ------------------------------------------------------------------ #
    # Batch sampling to ensure *in-batch positives* for SupCon
    #
    # Key point for your open-set goal:
    #   - specimen SupCon needs >=2 views per specimen in the SAME batch
    #   - species  SupCon needs >=2 specimens per species in the SAME batch
    #
    # For supcon_label_mode="both", we enforce BOTH simultaneously:
    #   species_per_batch × specimens_per_species × views_per_specimen == batch_size
    #
    # Example (batch_size=16):
    #   4 species × 2 specimens/species × 2 views/specimen = 16
    # ------------------------------------------------------------------ #
    def _pick_views_per_specimen_for_both(batch_size: int) -> int:
        for v in (4, 3, 2):
            if batch_size % v == 0:
                return v
        return 1

    class MultiViewSpecimenBatchSampler(torch.utils.data.Sampler):
        """
        Specimen-only sampler:
          specimens_per_batch distinct specimens
          views_per_specimen images per specimen
        """
        def __init__(self, subset, full_dataset, specimens_per_batch, views_per_specimen, seed=0):
            self.subset = subset
            self.full_dataset = full_dataset
            self.specimens_per_batch = int(specimens_per_batch)
            self.views_per_specimen = int(views_per_specimen)
            self.rng = random.Random(seed)

            # specimen_id -> subset indices (0..len(subset)-1)
            self.spec_to_subset = defaultdict(list)
            for subset_i, full_i in enumerate(self.subset.indices):
                _, _, sid, _ = self.full_dataset.samples[full_i]
                self.spec_to_subset[str(sid)].append(subset_i)

            self.specimen_ids = list(self.spec_to_subset.keys())

        def __iter__(self):
            ids = self.specimen_ids[:]
            self.rng.shuffle(ids)

            n_full = (len(ids) // self.specimens_per_batch) * self.specimens_per_batch
            ids = ids[:n_full]

            for start in range(0, len(ids), self.specimens_per_batch):
                chunk = ids[start:start + self.specimens_per_batch]
                batch = []
                for sid in chunk:
                    pool = self.spec_to_subset[sid]
                    if len(pool) >= self.views_per_specimen:
                        picks = self.rng.sample(pool, self.views_per_specimen)
                    else:
                        picks = [self.rng.choice(pool) for _ in range(self.views_per_specimen)]
                    batch.extend(picks)
                yield batch

        def __len__(self):
            return max(1, len(self.specimen_ids) // max(1, self.specimens_per_batch))

    class MultiViewSpeciesBatchSampler(torch.utils.data.Sampler):
        """
        Species(+specimen) sampler:
          species_per_batch distinct species
          specimens_per_species specimens per species
          views_per_specimen images per specimen

        Ensures in-batch positives for:
          - species (across specimens)
          - specimen (across views)
        """
        def __init__(
            self,
            subset,
            full_dataset,
            species_per_batch: int,
            specimens_per_species: int,
            views_per_specimen: int,
            seed=0,
        ):
            self.subset = subset
            self.full_dataset = full_dataset
            self.species_per_batch = int(species_per_batch)
            self.specimens_per_species = int(specimens_per_species)
            self.views_per_specimen = int(views_per_specimen)
            self.rng = random.Random(seed)

            # species -> specimen -> subset indices
            self.sp_to_sid_to_subset = defaultdict(lambda: defaultdict(list))
            for subset_i, full_i in enumerate(self.subset.indices):
                _p, sp_idx, sid, _v = self.full_dataset.samples[full_i]
                self.sp_to_sid_to_subset[int(sp_idx)][str(sid)].append(subset_i)

            self.species_ids = list(self.sp_to_sid_to_subset.keys())

        def __iter__(self):
            sp_ids = self.species_ids[:]
            self.rng.shuffle(sp_ids)

            # uniform batches; if too few species, sample with replacement
            n_batches = max(1, len(sp_ids) // max(1, self.species_per_batch))
            for _ in range(n_batches):
                if len(sp_ids) >= self.species_per_batch:
                    batch_species = self.rng.sample(sp_ids, self.species_per_batch)
                else:
                    batch_species = [self.rng.choice(sp_ids) for _ in range(self.species_per_batch)]

                batch = []
                for sp in batch_species:
                    sid_map = self.sp_to_sid_to_subset[sp]
                    sids = list(sid_map.keys())

                    # pick specimens within species
                    if len(sids) >= self.specimens_per_species:
                        pick_sids = self.rng.sample(sids, self.specimens_per_species)
                    else:
                        pick_sids = [self.rng.choice(sids) for _ in range(self.specimens_per_species)]

                    for sid in pick_sids:
                        pool = sid_map[sid]
                        if len(pool) >= self.views_per_specimen:
                            picks = self.rng.sample(pool, self.views_per_specimen)
                        else:
                            picks = [self.rng.choice(pool) for _ in range(self.views_per_specimen)]
                        batch.extend(picks)

                # safety: ensure exact batch_size
                if len(batch) != (self.species_per_batch * self.specimens_per_species * self.views_per_specimen):
                    # pad or trim deterministically
                    if len(batch) < (self.species_per_batch * self.specimens_per_species * self.views_per_specimen):
                        while len(batch) < (self.species_per_batch * self.specimens_per_species * self.views_per_specimen):
                            batch.append(batch[-1])
                    else:
                        batch = batch[: (self.species_per_batch * self.specimens_per_species * self.views_per_specimen)]
                yield batch

        def __len__(self):
            return max(1, len(self.species_ids) // max(1, self.species_per_batch))

    train_batch_sampler = None

    if args.supcon_label_mode == "both":
        views_per_specimen = _pick_views_per_specimen_for_both(args.batch_size)
        # prefer 2 specimens/species if possible
        specimens_per_species = 2 if (args.batch_size // views_per_specimen) % 2 == 0 else 1
        species_per_batch = max(1, args.batch_size // max(1, (views_per_specimen * specimens_per_species)))

        if views_per_specimen < 2 or specimens_per_species < 2:
            print(
                "[WARN] supcon_label_mode='both' but batch_size cannot factor into "
                ">=2 views/specimen and >=2 specimens/species. "
                "Consider batch_size=16, 24, 32 for best behavior."
            )

        print(
            f"✓ Using species+specimen batches: {species_per_batch} species/batch × "
            f"{specimens_per_species} specimens/species × {views_per_specimen} views/specimen "
            f"= {species_per_batch * specimens_per_species * views_per_specimen} images/batch"
        )
        train_batch_sampler = MultiViewSpeciesBatchSampler(
            subset=train_dataset,
            full_dataset=full_dataset,
            species_per_batch=species_per_batch,
            specimens_per_species=specimens_per_species,
            views_per_specimen=views_per_specimen,
            seed=split_seed,
        )

    elif args.supcon_label_mode == "species":
        # Species positives require >=2 specimens/species; views aren't required.
        views_per_specimen = 1
        specimens_per_species = 2 if args.batch_size % 2 == 0 else 1
        species_per_batch = max(1, args.batch_size // max(1, (views_per_specimen * specimens_per_species)))

        if specimens_per_species < 2:
            print("[WARN] supcon_label_mode='species' but batch_size is odd; species positives may be rare.")

        print(
            f"✓ Using species batches: {species_per_batch} species/batch × "
            f"{specimens_per_species} specimens/species × {views_per_specimen} views/specimen "
            f"= {species_per_batch * specimens_per_species * views_per_specimen} images/batch"
        )
        train_batch_sampler = MultiViewSpeciesBatchSampler(
            subset=train_dataset,
            full_dataset=full_dataset,
            species_per_batch=species_per_batch,
            specimens_per_species=specimens_per_species,
            views_per_specimen=views_per_specimen,
            seed=split_seed,
        )

    elif args.supcon_label_mode == "specimen":
        # Specimen positives require >=2 views/specimen in-batch.
        views_per_specimen = 2 if args.batch_size % 2 == 0 else 1
        specimens_per_batch = max(1, args.batch_size // max(views_per_specimen, 1))

        if views_per_specimen > 1:
            print(
                f"✓ Using specimen batches: {specimens_per_batch} specimens/batch × "
                f"{views_per_specimen} views/specimen = {specimens_per_batch * views_per_specimen} images/batch"
            )
            train_batch_sampler = MultiViewSpecimenBatchSampler(
                subset=train_dataset,
                full_dataset=full_dataset,
                specimens_per_batch=specimens_per_batch,
                views_per_specimen=views_per_specimen,
                seed=split_seed,
            )

    if train_batch_sampler is not None:
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=(args.num_workers > 0),
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )


    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
    else:
        val_loader = train_loader

    val_novel_loader = None
    if val_novel_dataset is not None:
        val_novel_loader = DataLoader(
            val_novel_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=(args.num_workers > 0),
            drop_last=False,
        )

    # Keep track of which species are actually represented in TRAIN (for open-set calibration)
    train_species_set = {int(specimen_to_species[sid]) for sid in train_specimens}
    # Losses
    if args.loss == "supcon":
        loss_fn = SupervisedContrastiveLoss(temperature=args.temperature)
        print(f"Loss: Supervised Contrastive (temperature={args.temperature})")
    else:
        loss_fn = TripletLoss(margin=args.margin)
        print(f"Loss: Triplet (margin={args.margin})")

    ce_criterion = nn.CrossEntropyLoss().to(device) if args.use_classifier_head else None

    # Optimizer
    # Optimizer
    if args.freeze_backbone:
        param_groups = [
            {"params": model.projection_head.parameters(), "lr": args.lr},
        ]

        if args.use_classifier_head and getattr(model, "classifier_head", None) is not None:
            param_groups.append({"params": model.classifier_head.parameters(), "lr": args.lr})

        # If we unfroze any backbone blocks, add ONLY those trainable backbone params
        if getattr(args, "unfreeze_last_n_blocks", 0) > 0:
            # Prefer model helper if you added it; otherwise fallback to requires_grad filter
            if hasattr(model, "get_trainable_backbone_params"):
                bb_params = model.get_trainable_backbone_params()
            else:
                bb_params = [p for p in model.backbone.parameters() if p.requires_grad]

            if len(bb_params) > 0:
                backbone_lr = float(getattr(args, "backbone_lr", args.lr))
                param_groups.append({"params": bb_params, "lr": backbone_lr})
                print(
                    f"Optimizer: AdamW proj{' + cls' if args.use_classifier_head else ''} "
                    f"+ backbone(last {args.unfreeze_last_n_blocks} blocks) "
                    f"(lr={args.lr}, backbone_lr={backbone_lr}, weight_decay={args.weight_decay})"
                )
            else:
                print(
                    f"[WARN] --unfreeze-last-n-blocks={args.unfreeze_last_n_blocks} "
                    "but no trainable backbone params were found. "
                    "Check that unfreezing is actually setting requires_grad=True."
                )
                print(
                    f"Optimizer: AdamW (lr={args.lr}, weight_decay={args.weight_decay}) "
                    f"on projection head{' + classifier head' if args.use_classifier_head else ''}"
                )

        else:
            print(
                f"Optimizer: AdamW (lr={args.lr}, weight_decay={args.weight_decay}) "
                f"on projection head{' + classifier head' if args.use_classifier_head else ''}"
            )

        optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    else:
        # Full fine-tuning (everything trainable) — simplest behavior
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print(f"Optimizer: AdamW (lr={args.lr}, weight_decay={args.weight_decay}) on full model (fine-tuning)")
 
 # counting the number of model parameters total 
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bb_total = sum(p.numel() for p in model.backbone.parameters())
    bb_trainable = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)

    print(f"[PARAMS] total={total/1e6:.2f}M  trainable={trainable/1e6:.2f}M")
    print(f"[PARAMS] backbone total={bb_total/1e6:.2f}M  backbone trainable={bb_trainable/1e6:.2f}M")

#    if args.freeze_backbone:
#        params = list(model.projection_head.parameters())
#        if args.use_classifier_head and model.classifier_head is not None:
#            params += list(model.classifier_head.parameters())
#        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
#        print(
#            f"Optimizer: AdamW (lr={args.lr}, weight_decay={args.weight_decay}) "
#            f"on projection head{' + classifier head' if args.use_classifier_head else ''}"
#        )
#    else:
#        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#        print(
#            f"Optimizer: AdamW (lr={args.lr}, weight_decay={args.weight_decay}) on full model (fine-tuning)"
#        )

    # Training loop
    print(f"\nStarting training: {args.epochs} epochs\n")
    best_separation = -np.inf
    best_model_score = -float("inf")        # <— not used yet, safe to keep or remove
    best_model_metric_name = None           # <— not used yet, safe to keep or remove
    
    history = {
        "epoch": [],
        "train_loss": [],
        "eval_epoch": [],
        "intra_similarity": [],
        "inter_similarity": [],
        "separation": [],
        # NEW: validation metrics from classifier head
        "val_loss_ce": [],
        "val_acc_top1": [],
    }

    # Specimen-level labels for SupCon if needed
    # Specimen-level labels for SupCon if needed
    if args.supcon_label_mode in ["specimen", "both"]:
        specimen_to_idx = dataset.specimen_to_idx
    else:
        specimen_to_idx = None

    # ── v24: Load multi-modal auxiliary data and init model heads ──────
    traits_by_specimen = None
    continuous_names = []
    meristic_names = []
    categorical_names = []
    cont_means = None
    cont_stds = None
    dna_by_specimen = None
    morph_by_specimen = None
    morph_feature_names = []

    if getattr(args, "trait_tsv", None):
        traits_by_specimen, continuous_names, meristic_names, categorical_names = load_trait_data(
            args.trait_tsv,
            specimen_id_col=getattr(args, "trait_specimen_id_col", "specimen_id"),
        )
        if continuous_names:
            cont_means, cont_stds = compute_trait_standardization(traits_by_specimen, continuous_names)

        # Init trait head on model
        model.init_trait_head(
            n_continuous=len(continuous_names),
            n_meristic=len(meristic_names),
            n_categorical=len(categorical_names),
        )
        model.trait_head = model.trait_head.to(device)

        # Add trait head params to optimizer
        trait_params = list(model.trait_head.parameters())
        optimizer.add_param_group({"params": trait_params, "lr": args.lr})
        print(f"  Added {sum(p.numel() for p in trait_params)} trait head params to optimizer")

    if getattr(args, "dna_fasta", None):
        dna_by_specimen = load_dna_data(
            args.dna_fasta,
            max_seq_len=getattr(args, "dna_max_seq_len", 700),
        )
        model.init_dna_encoder(
            output_dim=args.projection_dim,
            max_seq_len=getattr(args, "dna_max_seq_len", 700),
            num_species=num_species,
        )
        model.dna_encoder = model.dna_encoder.to(device)
        if model.dna_classifier is not None:
            model.dna_classifier = model.dna_classifier.to(device)

        dna_params = list(model.dna_encoder.parameters())
        if model.dna_classifier is not None:
            dna_params += list(model.dna_classifier.parameters())
        optimizer.add_param_group({"params": dna_params, "lr": args.lr})
        print(f"  Added {sum(p.numel() for p in dna_params)} DNA encoder params to optimizer")

    if getattr(args, "morph_tsv", None):
        morph_by_specimen, morph_feature_names = load_morph_features(
            args.morph_tsv,
            specimen_id_col=getattr(args, "morph_specimen_id_col", "specimen_id"),
        )
        if morph_feature_names:
            model.init_morph_encoder(
                morph_dim=len(morph_feature_names),
                output_dim=min(128, args.projection_dim),
            )
            model.morph_encoder = model.morph_encoder.to(device)

            morph_params = list(model.morph_encoder.parameters())
            optimizer.add_param_group({"params": morph_params, "lr": args.lr})
            print(f"  Added {sum(p.numel() for p in morph_params)} morph encoder params to optimizer")

    # Init cross-modal alignment loss if any cross-modal encoders are active
    if model.dna_encoder is not None or model.morph_encoder is not None:
        model.init_cross_modal_loss(temperature=getattr(args, "xmodal_temperature", 0.1))

    # Report multi-modal setup
    has_aux = any([traits_by_specimen, dna_by_specimen, morph_by_specimen])
    if has_aux:
        print("\n" + "─" * 60)
        print("MULTI-MODAL TRAINING ACTIVE (v24)")
        if traits_by_specimen:
            print(f"  Traits:  {len(traits_by_specimen)} specimens × "
                  f"({len(continuous_names)}C + {len(meristic_names)}M + {len(categorical_names)}B) traits")
        if dna_by_specimen:
            print(f"  DNA:     {len(dna_by_specimen)} specimens × COI barcodes")
        if morph_by_specimen:
            print(f"  Morph:   {len(morph_by_specimen)} specimens × {len(morph_feature_names)} features")
        print(f"  Weights: trait={getattr(args, 'trait_weight', 0.5)}, "
              f"dna={getattr(args, 'dna_weight', 0.5)}, "
              f"xmodal={getattr(args, 'xmodal_weight', 0.3)}")
        print("─" * 60)


    for epoch in range(1, args.epochs + 1):
        print("\n" + "=" * 60)
        print(f"Epoch {epoch}/{args.epochs}")
        print("=" * 60)

        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            contrastive_loss_fn=loss_fn,
            ce_criterion=ce_criterion,
            device=device,
            epoch=epoch,
            args=args,
            specimen_to_idx=specimen_to_idx,
            # v24: multi-modal auxiliary data
            traits_by_specimen=traits_by_specimen,
            continuous_names=continuous_names,
            meristic_names=meristic_names,
            categorical_names=categorical_names,
            cont_means=cont_means,
            cont_stds=cont_stds,
            dna_by_specimen=dna_by_specimen,
            morph_by_specimen=morph_by_specimen,
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            print("\nEvaluating...")
            metrics, embeddings, labels, specimen_ids, view_ids, img_paths = evaluate_model(
                model,
                val_loader,
                device,
                compute_classification=args.use_classifier_head,
                ce_criterion=ce_criterion,
            )

            # Optional: visualize attention + masks
            if args.viz_attn_num_samples > 0:
                visualize_masks_and_attention(
                    model,
                    full_dataset,  # visualize from the same preprocessing pipeline
                    device,
                    args.output_dir,
                    epoch,
                    num_samples=args.viz_attn_num_samples,
                )

            # ---- update history ----
            history["eval_epoch"].append(epoch)
            history["intra_similarity"].append(metrics["mean_intra_similarity"])
            history["inter_similarity"].append(metrics["mean_inter_similarity"])
            history["separation"].append(metrics["separation"])

            if "val_loss_ce" in metrics:
                history["val_loss_ce"].append(metrics["val_loss_ce"])
                history["val_acc_top1"].append(metrics["val_accuracy_top1"])

            # ---- save per-view metrics ----
            view_metrics_path = os.path.join(
                args.output_dir, f"view_metrics_epoch_{epoch}.json"
            )
            with open(view_metrics_path, "w") as f:
                json.dump(metrics["per_view"], f, indent=2)
            print(f"✓ Per-view metrics saved: {view_metrics_path}")
            open_set_out = None  # filled during open-set eval
            # ---- open-set retrieval + threshold calibration ----
            do_open_set = bool(args.open_set_eval or (args.species_holdout_fraction > 0.0))
            if do_open_set and (epoch % args.open_set_eval_every == 0 or epoch == args.epochs):
                try:
                    open_set_out = evaluate_open_set_retrieval(
                        model=model,
                        train_eval_loader=train_eval_loader,
                        val_known_loader=(val_loader if val_dataset is not None else train_eval_loader),
                        val_novel_loader=val_novel_loader,
                        device=device,
                        output_dir=args.output_dir,
                        epoch=epoch,
                        topk=args.open_set_topk,
                        far_targets=args.open_set_far_targets,
                    )
                    if isinstance(open_set_out, dict):
                        history.setdefault("open_set_eval_epoch", []).append(epoch)
                        history.setdefault("novelty_auc", []).append(open_set_out.get("novelty_auc"))
                        history.setdefault("novelty_eer", []).append(open_set_out.get("novelty_eer"))
                        history.setdefault("knn_top1", []).append(open_set_out.get("knn_top1"))
                        history.setdefault("knn_top5", []).append(open_set_out.get("knn_top5"))
                        history.setdefault("centroid_top1", []).append(open_set_out.get("centroid_top1"))
                        history.setdefault("centroid_top5", []).append(open_set_out.get("centroid_top5"))
                except Exception as e:
                    print(f"[WARN] Open-set evaluation failed: {e}")


            # ---- always save a 'latest' checkpoint ----
            latest_checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
                "open_set_metrics": open_set_out,
                "history": history,
                "args": vars(args),
            }
            latest_path = os.path.join(args.output_dir, "latest_model.pth")
            torch.save(latest_checkpoint, latest_path)
            print(f"✓ Latest model checkpoint saved: {latest_path}")

            # ---- update best separation (for diagnostics only) ----
            if metrics["separation"] > best_separation:
                best_separation = metrics["separation"]

            # ---- choose metric for best_model.pth ----
            # Priority (if available):
            #   1) novelty_auc (open-set eval enabled)  <-- best matches your deployment goal
            #   2) val_accuracy_top1 (if classifier head is on)
            #   3) separation (fallback embedding quality signal)
            score = None
            metric_name = "separation"

            if do_open_set and isinstance(open_set_out, dict):
                auc = open_set_out.get("novelty_auc", None)
                if auc is not None and np.isfinite(auc):
                    score = float(auc)
                    metric_name = "novelty_auc"

            if score is None:
                val_acc = metrics.get("val_accuracy_top1", None)
                if val_acc is not None:
                    score = float(val_acc)
                    metric_name = "val_accuracy_top1"
                else:
                    score = float(metrics["separation"])
                    metric_name = "separation"

            if score > best_model_score:
                best_model_score = score
                best_model_metric_name = metric_name
                best_path = os.path.join(args.output_dir, "best_model.pth")
                torch.save(latest_checkpoint, best_path)
                print(
                    f"✓ Best model saved: {best_path} "
                    f"({metric_name}={score:.4f})"
                )

            # ---- update training progress figure ----
            plot_training_progress(history, args.output_dir, epoch)

            # Final t-SNE visualizations at the end of training
            if epoch == args.epochs:
                # 1) Legacy plot (image-level, species clusters) — kept for backward compatibility
                vis_path = os.path.join(args.output_dir, "species_clusters_tsne.png")
                visualize_species_clusters(embeddings, labels, specimen_ids, vis_path)

                # 2) NEW: specimen-level t-SNE, colored by species, with train/val overlays
                try:
                    tr_emb, tr_lab, tr_sid, _tr_vid, _tr_paths = collect_embeddings(model, train_eval_loader, device)
                    vk_emb, vk_lab, vk_sid, _vk_vid, _vk_paths = collect_embeddings(model, val_loader, device)

                    all_emb = [tr_emb, vk_emb]
                    all_lab = [tr_lab, vk_lab]
                    all_sid = list(tr_sid) + list(vk_sid)
                    all_split = (["train"] * len(tr_sid)) + (["val_known"] * len(vk_sid))

                    if val_novel_loader is not None:
                        vn_emb, vn_lab, vn_sid, _vn_vid, _vn_paths = collect_embeddings(model, val_novel_loader, device)
                        all_emb.append(vn_emb)
                        all_lab.append(vn_lab)
                        all_sid += list(vn_sid)
                        all_split += (["val_novel"] * len(vn_sid))

                    all_emb = np.concatenate(all_emb, axis=0)
                    all_lab = np.concatenate(all_lab, axis=0)

                    vis_path2 = os.path.join(args.output_dir, "specimen_tsne_by_species.png")
                    visualize_specimen_tsne_by_species(
                        embeddings=all_emb,
                        labels=all_lab,
                        specimen_ids=all_sid,
                        split_names=all_split,
                        output_path=vis_path2,
                        max_points=getattr(args, "tsne_max_points", 5000),
                        perplexity=getattr(args, "tsne_perplexity", 30),
                        random_state=getattr(args, "seed", 42),
                        label_to_name=getattr(dataset, "idx_to_species", None),
                    )
                except Exception as _e:
                    print(f"[WARN] Specimen-level t-SNE skipped: {_e}")


    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best separation observed: {best_separation:.4f}")
    if best_model_metric_name is not None:
        print(
            f"Best model selection metric: {best_model_metric_name} = "
            f"{best_model_score:.4f}"
        )
    print(f"Output directory: {args.output_dir}")


    # Build memory bank from final embeddings
    print("\nBuilding memory bank from trained embeddings...")
    # NOTE: memory bank should be built from TRAIN only by default
    _mb_scope = getattr(args, 'memory_bank_scope', 'train')
    if _mb_scope == 'full':
        mb_loader = full_loader
    elif _mb_scope == 'train+val_known' and 'val_dataset' in locals() and val_dataset is not None:
        from torch.utils.data import ConcatDataset
        mb_dataset = ConcatDataset([train_dataset, val_dataset])
        mb_loader = DataLoader(mb_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    else:
        mb_loader = train_eval_loader

    metrics, embeddings, labels, specimen_ids, view_ids, img_paths = evaluate_model(
        model, mb_loader, device
    )

    memory_bank = SpeciesMemoryBank(similarity_threshold=args.similarity_threshold)
    # Aggregate to ONE embedding per specimen so each specimen contributes equally
    # (critical for open-set calibration and to avoid view-count bias).
    specimen_sum = {}
    specimen_count = {}
    specimen_label = {}
    specimen_rep_path = {}
    specimen_views = {}

    for emb, label, spec_id, view_id, img_path in zip(
        embeddings, labels, specimen_ids, view_ids, img_paths
    ):
        sid = str(spec_id)
        lab = int(label)
        if sid not in specimen_sum:
            specimen_sum[sid] = emb.astype(np.float64)
            specimen_count[sid] = 1
            specimen_label[sid] = lab
            specimen_rep_path[sid] = str(img_path)
            specimen_views[sid] = {str(view_id)}
        else:
            if specimen_label[sid] != lab:
                raise ValueError(f"Inconsistent species label for specimen {sid}: {specimen_label[sid]} vs {lab}")
            specimen_sum[sid] += emb.astype(np.float64)
            specimen_count[sid] += 1
            specimen_views[sid].add(str(view_id))

    specimen_embeddings = {}
    for sid in specimen_sum.keys():
        e = (specimen_sum[sid] / max(1, specimen_count[sid])).astype(np.float32)
        e = e / (np.linalg.norm(e) + 1e-12)
        specimen_embeddings[sid] = e

    # Add specimen-level embeddings to the memory bank
    for sid, emb in specimen_embeddings.items():
        species_name = dataset.idx_to_species[int(specimen_label[sid])]
        rep_path = specimen_rep_path[sid]
        view_tag = "MULTI:" + ",".join(sorted(specimen_views[sid]))

        if species_name not in memory_bank.species_clusters:
            cluster = SpeciesCluster(
                species_id=species_name,
                embeddings=[emb],
                specimen_ids=[sid],
                image_paths=[rep_path],
                view_ids=[view_tag],
            )
            cluster.update_centroid()
            memory_bank.species_clusters[species_name] = cluster
        else:
            memory_bank.species_clusters[species_name].add_specimen(
                emb, sid, rep_path, view_id=view_tag
            )

    bank_export = {"species": {}}
    for sp_id, cluster in memory_bank.species_clusters.items():
        bank_export["species"][sp_id] = {
            "embeddings": [emb.tolist() for emb in cluster.embeddings],
            "specimen_ids": cluster.specimen_ids,
            "image_paths": cluster.image_paths,
            "view_ids": cluster.view_ids,
            "centroid": cluster.centroid.tolist() if cluster.centroid is not None else None,
        }

    memory_bank_path = os.path.join(args.output_dir, "memory_bank.json")
    with open(memory_bank_path, "w") as f:
        json.dump(bank_export, f, indent=2)

    print(f"✓ Memory bank saved: {memory_bank_path}")
    print(f"  Total species: {len(memory_bank.species_clusters)}")
    for sp_id, cluster in memory_bank.species_clusters.items():
        print(f"    {sp_id}: {len(cluster.embeddings)} images (views: {set(cluster.view_ids)})")


if __name__ == "__main__":
    main()
