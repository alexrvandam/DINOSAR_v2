# *DINOSAR v2 — **DINO**v3 **S**pecies **A**uto-**R**ecovery version 2: Multi-modal Bayesian species delimitation and recognition for hyperdiverse insect taxa*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)

**DINOSAR v2** (**DI**NO-based **N**eural **O**ntology for **S**pecimen **A**nalysis and **R**ecognition, version 2) is a multi-modal, Bayesian framework for automated species delimitation in hyperdiverse insect taxa. It fuses three independent evidence streams — vision embeddings (DINOv3), morphological measurements (from Descriptron COCO JSON), and DNA barcodes (COI FASTA) — into a unified adaptive Bayesian classifier that scales to dark taxa with few labelled specimens.

DINOSAR_v2 is designed to work hand-in-hand with the [Descriptron](https://github.com/alexrvandam/Descriptron) annotation platform.

---

## Pipeline overview

The pipeline runs in two sequential stages. DNA is a required input to Stage 2; two preparation utilities are provided to get raw sequence data into the expected format.

```
╔══════════════════════════════════════════════════════════════════════╗
║                        DINOSAR v2 Pipeline                          ║
║                                                                      ║
║  [DNA PREP — run if your sequences come from BOLD or need QC]       ║
║  BOLD JSONL ──► bold_jsonl_to_coi_fasta_qc.py ──► sequences.fasta  ║
║  sequences.fasta + clusters.tsv                                      ║
║       └──► dinosar_dna_barcode_gap_gate.py ──► QC-filtered FASTA   ║
║                                                                      ║
║  ┌──────────────────────────────────────────────────────────────┐   ║
║  │ STAGE 1 — Vision embedding and species discovery             │   ║
║  │                                                              │   ║
║  │  Image directory                                             │   ║
║  │       └──► DINOSAR_v2_contrastive_species_learning_CE.py    │   ║
║  │                 └──► predictions_multiview.tsv               │   ║
║  │                 └──► memory_bank.json                        │   ║
║  └──────────────────────────────────────────────────────────────┘   ║
║                              │                                       ║
║                              ▼                                       ║
║  ┌──────────────────────────────────────────────────────────────┐   ║
║  │ STAGE 2 — Multi-modal Bayesian fusion (orchestrator)         │   ║
║  │                                                              │   ║
║  │  predictions_multiview.tsv  ◄── Stage 1                     │   ║
║  │  morphology.json            ◄── Descriptron                 │   ║
║  │  sequences_qc.fasta         ◄── DNA prep                    │   ║
║  │       └──► DINOSAR_v2_unified_complete_MFA_Bayesian_v1.py   │   ║
║  │                 └──► final_predictions.tsv                   │   ║
║  │                 └──► per-specimen reliability scores         │   ║
║  │                 └──► learned_priors.json                     │   ║
║  └──────────────────────────────────────────────────────────────┘   ║
╚══════════════════════════════════════════════════════════════════════╝
```

### Stage 2 fusion features

- **Adaptive Bayesian fusion** — modality weights (α) are learned from validation performance, not hand-tuned; DNA weight is hard-capped at ~30% to prevent barcode data from overwhelming vision and morphology evidence
- **MFA morphology module** — handles mixed continuous, meristic (count), and categorical character data via Multiple Factor Analysis, with PCA fallback when `prince` is unavailable
- **Per-specimen reliability gates** — each specimen's contribution from each modality is weighted by its own data quality score (feature completeness, distance to centroid, cluster separation)
- **SMOTE oversampling** — balances rare species during classifier training
- **Open-set novel species detection** — specimens outside known cluster space are flagged rather than force-assigned
- **Validation-learned priors** — α weights are derived from per-modality accuracy and cluster separation on held-out validation data

---

## Repository structure

```
DINOSAR_v2/
├── README.md
├── environment.yml
├── LICENSE
├── .gitignore
│
├── # ── Stage 1: Vision ──────────────────────────────────────────
├── DINOSAR_v2_contrastive_species_learning_CE_v23r.py
│       DINOv3 backbone + supervised contrastive learning.
│       Produces the multi-view prediction TSV consumed by Stage 2.
│
├── # ── Stage 2: Orchestrator ────────────────────────────────────
├── DINOSAR_v2_unified_complete_MFA_Bayesian_v1.py
│       End-to-end fusion of vision predictions + Descriptron
│       morphology COCO JSON + COI FASTA → species assignments.
│       This is the main script for most users.
│
├── # ── DNA input preparation ────────────────────────────────────
├── bold_jsonl_to_coi_fasta_qc.py
│       Converts BOLD Systems JSONL exports to QC-filtered COI
│       FASTA + metadata TSV. Run first if your DNA data comes
│       from a BOLD download.
│
└── dinosar_dna_barcode_gap_gate.py
        Within/between cluster p-distance QC gate. Outputs a
        PASS/FAIL table per cluster and a filtered FASTA ready
        for Stage 2. Run on any COI FASTA + cluster TSV before
        fusion if you want to screen out poor-quality clusters.
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

> **GPU note**: the `environment.yml` installs PyTorch with CUDA 12.1. If your system uses a different CUDA version, replace the PyTorch channel line with the appropriate one from [pytorch.org](https://pytorch.org/get-started/locally/). CPU-only inference is supported but substantially slower for large image sets.

### 3. Verify installation

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python DINOSAR_v2_contrastive_species_learning_CE_v23r.py --help
python DINOSAR_v2_unified_complete_MFA_Bayesian_v1.py --help
```

---

## Running the pipeline

### Step 0 (if needed): Prepare DNA input

Skip this step if you already have a COI FASTA file with headers matching your `specimen_id` values.

**From a BOLD Systems export:**

```bash
python bold_jsonl_to_coi_fasta_qc.py \
  --bold-jsonl          raw/MyGenus.jsonl \
  --out-prefix          data/mygenus_bold \
  --min-effective-len   600 \
  --max-ambig-frac      0.02 \
  --trim-trailing-ambig
# → data/mygenus_bold.fasta  (QC-filtered, keyed by specimen_id)
```

**Optional barcode gap QC before fusion:**

```bash
python dinosar_dna_barcode_gap_gate.py \
  --coi-fasta     data/mygenus_bold.fasta \
  --clusters-tsv  data/initial_clusters.tsv \
  --out-dir       data/dna_qc/
# → use the PASS-filtered FASTA as --coi_fasta in Stage 2
```

---

### Stage 1: Vision embedding and species discovery

```bash
# Discover species clusters from an image directory
python DINOSAR_v2_contrastive_species_learning_CE_v23r.py \
  --image_dir   images/ \
  --train_tsv   data/train.tsv \
  --out_dir     output/stage1/ \
  --mode        discover

# Zero-shot classification of new/test specimens
python DINOSAR_v2_contrastive_species_learning_CE_v23r.py \
  --image_dir     images/test/ \
  --memory_bank   output/stage1/memory_bank.json \
  --out_dir       output/stage1/zeroshot/ \
  --mode          classify
```

The key output is `output/stage1/predictions_multiview.tsv`, which feeds directly into Stage 2.

---

### Stage 2: Multi-modal Bayesian fusion

```bash
python DINOSAR_v2_unified_complete_MFA_Bayesian_v1.py \
  --train_tsv          data/train.tsv \
  --predictions_mv_tsv output/stage1/predictions_multiview.tsv \
  --coco_json          data/morphology_annotations.json \
  --coi_fasta          data/mygenus_bold.fasta \
  --val_known_tsv      data/val_known.tsv \
  --val_novel_tsv      data/val_novel.tsv \
  --test_tsv           data/test.tsv \
  --use_smote \
  --out_dir            output/stage2/
```

Key outputs in `output/stage2/`:

| File | Contents |
|---|---|
| `final_predictions.tsv` | Per-specimen species assignment + posterior probability |
| `learned_priors.json` | Learned α weights for vision, morphology, and DNA |
| `validation_results.tsv` | Per-specimen accuracy and per-modality reliability scores |
| `novel_specimens.tsv` | Specimens flagged as potential novel species |

---

## Input data formats

| Argument | Format | Description |
|---|---|---|
| `--train_tsv` | TSV | `specimen_id`, `species`, `view` (H/D/P), optional metadata |
| `--val_known_tsv` | TSV | Same format; species present in training set |
| `--val_novel_tsv` | TSV | Same format; species absent from training set |
| `--test_tsv` | TSV | `specimen_id`, `view`; no species column required |
| `--predictions_mv_tsv` | TSV | Output of Stage 1; per-specimen similarity scores per species |
| `--coco_json` | COCO JSON | Morphological annotations produced by Descriptron |
| `--coi_fasta` | FASTA | COI barcodes keyed by `specimen_id` |

---

## Integration with Descriptron

DINOSAR v2 consumes morphological annotation outputs from the [Descriptron](https://github.com/alexrvandam/Descriptron) platform. The standard workflow is:

1. Annotate specimens in Descriptron → export COCO JSON with keypoints and measurements
2. Run Descriptron keypoint prediction and measurement scripts to populate the COCO JSON
3. Pass the resulting COCO JSON to Stage 2 via `--coco_json`

See the workflow diagram PDF in the [Releases](https://github.com/alexrvandam/DINOSAR_v2/releases) for a full illustrated walkthrough of a real *Tetramorium* dataset.

---

## Citation

If you use DINOSAR v2 in your research, please cite:

> Van Dam, A. (2026). *DINOSAR v2 — DINOv3 Species Auto-Recovery version 2: Multi-modal Bayesian species delimitation and recognition for hyperdiverse insect taxa*. [Computer Software] https://github.com/alexrvandam/DINOSAR_v2.

A full methods paper describing the two-tier DINOSAR + Descriptron system is in preparation.

---

## Acknowledgements

DINOSAR v2 was developed at the **Museum für Naturkunde Berlin** within the Centre for Integrative Biodiversity Discovery. I thank Dr. Francisco Hita Garcia for taxonomic expertise on hyperdiverse ant genera and for providing training datasets for *Tetramorium* and related genera.

# Additional Literature and Software to Cite if you use this project:

@misc{oquab2024dinov2learningrobustvisual,
      title={DINOv2: Learning Robust Visual Features without Supervision}, 
      author={Maxime Oquab and Timothée Darcet and Théo Moutakanni and Huy Vo and Marc Szafraniec and Vasil Khalidov and Pierre Fernandez and Daniel Haziza and Francisco Massa and Alaaeldin El-Nouby and Mahmoud Assran and Nicolas Ballas and Wojciech Galuba and Russell Howes and Po-Yao Huang and Shang-Wen Li and Ishan Misra and Michael Rabbat and Vasu Sharma and Gabriel Synnaeve and Hu Xu and Hervé Jegou and Julien Mairal and Patrick Labatut and Armand Joulin and Piotr Bojanowski},
      year={2024},
      eprint={2304.07193},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2304.07193}, 
}

@misc{siméoni2025dinov3,
      title={DINOv3}, 
      author={Oriane Siméoni and Huy V. Vo and Maximilian Seitzer and Federico Baldassarre and Maxime Oquab and Cijo Jose and Vasil Khalidov and Marc Szafraniec and Seungeun Yi and Michaël Ramamonjisoa and Francisco Massa and Daniel Haziza and Luca Wehrstedt and Jianyuan Wang and Timothée Darcet and Théo Moutakanni and Leonel Sentana and Claire Roberts and Andrea Vedaldi and Jamie Tolan and John Brandt and Camille Couprie and Julien Mairal and Hervé Jégou and Patrick Labatut and Piotr Bojanowski},
      year={2025},
      eprint={2508.10104},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.10104}, 
}

Gatis, D. (2025). rembg (Version 2.0.66) [Computer software]. https://github.com/danielgatis/rembg

---

## License

MIT License — see [LICENSE](LICENSE) for details.
