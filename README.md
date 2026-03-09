# DINOSAR v2 — DINOv3-based Species Delimitation and Recognition

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)

**DINOSAR v2** (**DINO**v3 **S**pecies **A**uto **R**ecovery version 2) is a multi-modal, Bayesian framework for automated species delimitation in hyperdiverse insect taxa. It fuses three independent evidence streams — vision embeddings, morphological measurements, and DNA barcodes — into a unified adaptive Bayesian classifier that scales to dark taxa with few labelled specimens.

DINOSAR v2 was developed with Dr. Francisco Hita Garcia as part of and designed to work hand-in-hand with the [Descriptron annotation platform](https://github.com/alexrvandam/Descriptron).

---

## Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DINOSAR v2 Pipeline                          │
│                                                                     │
│  Images ──► DINOv3 backbone ──► Contrastive embeddings             │
│                                        │                           │
│  Morphology (COCO JSON) ──► MFA ───────┼──► Adaptive Bayesian      │
│  (geometric + meristic + categorical)  │      Fusion               │
│                                        │        │                  │
│  COI sequences (FASTA) ──► Barcode ───►┘        ▼                  │
│                             gap gate         Species               │
│                                              assignments           │
│                                          (known + novel)           │
└─────────────────────────────────────────────────────────────────────┘
```

### Key features

- **Zero-shot species discovery** — progressive incremental clustering (A, B, C…) from unlabelled image collections using a DINOv3 contrastive backbone
- **Multi-view fusion** — dorsal, lateral, and frontal views combined into per-specimen embeddings
- **Adaptive Bayesian fusion** — per-specimen reliability gates prevent any single modality from dominating; DNA weight is hard-capped at ~30%
- **Open-set novel species detection** — specimens that fall outside known clusters are flagged rather than force-assigned
- **Hierarchical Bayesian genus collector** — novel species can be assigned to the correct genus even when species-level identity is unknown
- **MFA morphology module** — handles mixed continuous, meristic (count), and categorical character data via Multiple Factor Analysis
- **SMOTE oversampling** — balances rare species during classifier training
- **DNA barcode gap gate** — fast within/between distance QC before mPTP/bPTP species delimitation
- **BOLD data utilities** — parse BOLD Systems JSONL exports into QC-filtered COI FASTA + metadata TSV

---

## Repository structure

```
DINOSAR_v2/
├── README.md
├── environment.yml                              # Conda environment
├── LICENSE
│
├── # ── Core pipeline scripts ────────────────────────────────────
├── DINOSAR_v2_contrastive_species_learning_CE_v23r.py
│       DINOv3 backbone + supervised contrastive learning,
│       zero-shot classification, species memory bank,
│       multi-view embedding fusion.
│
├── DINOSAR_v2_unified_complete_MFA_Bayesian_v1.py
│       Complete end-to-end pipeline: vision + MFA morphology
│       + DNA → adaptive Bayesian fusion. Use this as the
│       single entry point for a full run.
│
├── DINOSAR_bayesian_fusion_v2_hierbayes.py
│       Hierarchical Bayesian fusion module (genus collector).
│       Fuses vision similarity TSVs, morphology matrices,
│       and COI distances; outputs per-specimen posteriors.
│
├── DINOSAR_species_delimiter_v2_adaptive_bayesian_MFA.py
│       Adaptive Bayesian species delimiter with MFA-based
│       morphology reliability; validation-learned priors;
│       product-of-experts fusion.
│
├── dinosaar_morphology_mfa_classifier.py
│       Standalone morphology classifier: geometric Procrustes
│       PCs, meristic counts, categorical presence/absence,
│       continuous measurements → MFA → Mahalanobis assignment.
│
├── dinosar_dna_barcode_gap_gate.py
│       COI barcode gap gate: within-cluster vs between-cluster
│       p-distance QC, PASS/FAIL per cluster, optional per-cluster
│       FASTA export for downstream mPTP/bPTP.
│
├── bold_jsonl_to_coi_fasta_qc.py
│       Parse BOLD Systems JSONL exports → QC-filtered COI FASTA
│       + metadata TSV + BIN/species count tables.
│
├── gm_mahanalobis_from_frozen_features.py
│       Post-hoc Mahalanobis assignment and outlier triage on
│       frozen PCA features; Hotelling's T² between putative
│       species; PC scatter plots with flags.
│
└── FHS_and_CLAHE_V17.py
        Image preprocessing: Felzenszwalb superpixel segmentation
        → CLAHE → LAB colour averaging; optional shine removal
        and colour-true dendrograms. Run before feature extraction.
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/alexrvandam/DINOSAR_v2.git
cd DINOSAR_v2
```

### 2. Create and activate the conda environment

```bash
conda env create -f environment.yml
conda activate dinosar_v2
```

> **GPU note**: the `environment.yml` installs PyTorch with CUDA 12.1. If your system uses a different CUDA version, replace the PyTorch channel line with the appropriate one from [pytorch.org](https://pytorch.org/get-started/locally/). CPU-only inference is supported but slow for large image sets.

### 3. Verify installation

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python DINOSAR_v2_contrastive_species_learning_CE_v23r.py --help
```

---

## Quick-start workflows

### A. Full pipeline (vision + morphology + DNA)

```bash
python DINOSAR_v2_unified_complete_MFA_Bayesian_v1.py \
  --train_tsv          data/train.tsv \
  --predictions_mv_tsv data/vision_predictions_multiview.tsv \
  --coco_json          data/morphology_annotations.json \
  --coi_fasta          data/sequences_qc.fasta \
  --val_known_tsv      data/val_known.tsv \
  --val_novel_tsv      data/val_novel.tsv \
  --test_tsv           data/test.tsv \
  --out_dir            output/
```

### B. Vision-only: species discovery from images

```bash
# Step 1 – extract embeddings and discover species clusters
python DINOSAR_v2_contrastive_species_learning_CE_v23r.py \
  --image_dir     images/ \
  --train_tsv     data/train.tsv \
  --out_dir       output/vision/ \
  --mode          discover

# Step 2 – zero-shot classification of new specimens
python DINOSAR_v2_contrastive_species_learning_CE_v23r.py \
  --image_dir     images/test/ \
  --memory_bank   output/vision/memory_bank.json \
  --out_dir       output/vision/zeroshot/ \
  --mode          classify
```

### C. Bayesian fusion from pre-computed modality outputs

```bash
python DINOSAR_bayesian_fusion_v2_hierbayes.py \
  --train_tsv           data/train.tsv \
  --predictions_mv_tsv  output/vision/predictions_multiview.tsv \
  --morphology_tsv      output/morphology/mfa_scores.tsv \
  --coi_distance_tsv    output/dna/barcode_gap_gate.tsv \
  --out_dir             output/fusion/
```

### D. DNA barcode QC and gap gate

```bash
# Parse BOLD export
python bold_jsonl_to_coi_fasta_qc.py \
  --bold-jsonl    raw/Tetramorium.jsonl \
  --out-prefix    data/tetramorium_bold_qc \
  --min-effective-len 600 \
  --max-ambig-frac 0.02 \
  --trim-trailing-ambig

# Run barcode gap gate
python dinosar_dna_barcode_gap_gate.py \
  --coi-fasta     data/tetramorium_bold_qc.fasta \
  --clusters-tsv  output/vision/predictions_multiview.tsv \
  --out-dir       output/dna/
```

### E. Image preprocessing

```bash
python FHS_and_CLAHE_V17.py \
  --input_dir  images/raw/ \
  --mask_dir   images/masks/ \
  --output_dir images/preprocessed/
```

---

## Input data formats

| File | Format | Description |
|---|---|---|
| `train.tsv` | TSV | `specimen_id`, `species`, `view` (H/D/P), optional metadata |
| `val_known.tsv` | TSV | Same format; species present in training set |
| `val_novel.tsv` | TSV | Same format; species absent from training set |
| `test.tsv` | TSV | `specimen_id`, `view`; species column optional |
| `morphology.json` | COCO JSON | Produced by Descriptron; contains keypoints + measurements |
| `sequences.fasta` | FASTA | COI barcodes keyed by `specimen_id` |
| `Genus.jsonl` | JSONL | Raw BOLD Systems export (one JSON object per line) |

---

## Integration with Descriptron

DINOSAR v2 is designed to consume annotation outputs produced by the [Descriptron](https://github.com/alexrvandam/Descriptron) morphological annotation platform. The standard workflow is:

1. Annotate specimens in Descriptron → export COCO JSON
2. Run Descriptron keypoint prediction + measurement scripts
3. Feed COCO JSON and measurement CSVs into DINOSAR v2 morphology module
4. Combine with vision embeddings and COI barcodes for final delimitation

See `DINOSARxDescriptron_workflow.pdf` (in the [Releases](https://github.com/alexrvandam/DINOSAR_v2/releases)) for a detailed illustrated workflow.

---

## Citation

If you use DINOSAR v2 in your research, please cite:

> Van Dam, A. (2026). *DINOSAR v2: Multi-modal Bayesian species delimitation for hyperdiverse insect taxa*. Museum für Naturkunde Berlin. https://doi.org/[Zenodo DOI pending]

A full methods paper describing the two-tier DINOSAR + Descriptron system is in preparation.

---

## Acknowledgements

DINOSAR v2 was developed at the **Museum für Naturkunde Berlin** within the Centre for Integrative Biodiversity Discovery. I thank Dr. Francisco Hita Garcia for taxonomic expertise on hyperdiverse ant genera and for providing training datasets for *Tetramorium* and related genera.

The DINOv3 backbone is provided by Meta AI Research ([Oquab et al. 2023](https://arxiv.org/abs/2304.07193)).

---

## License

MIT License — see [LICENSE](LICENSE) for details.
