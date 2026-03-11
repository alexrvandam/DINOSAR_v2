# DINOSAR v2 — DINOv3 Species Auto-Recovery version 2

## `DINOSAR_v2_contrastive_species_learning_CE_v24.py`

&#x20;&#x20;

**DINOSAR v2** — **DINO**v3 **S**pecies **A**uto-**R**ecovery version 2 — is a DINOv3-based contrastive learning framework for species discovery, zero-shot classification, and open-set recognition in hyperdiverse insect taxa.

This repository version centers on the script:

```bash
DINOSAR_v2_contrastive_species_learning_CE_v24.py
```

Version 24 extends the earlier contrastive DINOSAR workflow with **optional multi-modal auxiliary learning**, allowing image embeddings to be trained jointly with:

- **continuous morphological traits**
- **meristic count data**
- **categorical/binary traits**
- **DNA barcodes (COI)**
- **morphometric feature vectors**

All auxiliary modalities are optional. Specimens missing one or more data types are still used normally for image-based training, while auxiliary losses are applied only where matching data are available.

DINOSAR v2 is designed to work alongside the [Descriptron](https://github.com/alexrvandam/Descriptron) ecosystem for image annotation, morphology extraction, and downstream taxonomic workflows.

---

## What v24 does

`DINOSAR_v2_contrastive_species_learning_CE_v24.py` supports:

- **DINOv3-based feature extraction** using local checkpoints
- **Supervised contrastive learning** for species discrimination
- **Specimen-level or species-level contrastive supervision**
- **Open-set evaluation** for novel species recognition
- **Incremental species discovery** using a memory bank
- **Zero-shot prediction** for new images against known species clusters
- **Multi-view zero-shot prediction** at the specimen level
- **Optional classifier-head training** with cross-entropy
- **Optional auxiliary trait prediction** from visual features
- **Optional DNA encoder training** on COI barcodes
- **Optional morphometric encoder training**
- **Cross-modal alignment losses** between vision, DNA, and morphometric embeddings
- **Foreground preprocessing** with background removal and cropping
- **Optional COCO mask loading** for specimen or structure-specific masking
- **Patch/attention visualizations** for qualitative inspection
- **Export of split TSVs, metrics, plots, and run metadata**

---

## v24 architecture overview

```text
Images
  │
  ▼
DINOv3 backbone
  │
  ├── Projection head ──► contrastive embedding space
  │                         ├── memory bank
  │                         ├── species discovery
  │                         ├── zero-shot prediction
  │                         └── open-set evaluation
  │
  ├── Optional classifier head ──► species CE loss
  │
  ├── Optional trait head ───────►
  │         ├── continuous traits (MSE)
  │         ├── meristic counts (Poisson NLL)
  │         └── categorical traits (BCE)
  │
  ├── Optional DNA encoder ──────► DNA species CE
  │
  └── Optional morph encoder ────► morphometric embedding
                                    │
                                    └── cross-modal alignment
                                         (vision ↔ DNA, vision ↔ morph)
```

---

## Key new features in v24

### 1. Multi-modal auxiliary learning

v24 can train the image backbone jointly with auxiliary supervision from non-image data:

- **Trait TSV**:
  - continuous numeric traits
  - meristic counts
  - categorical/binary characters
- **DNA FASTA**:
  - COI barcode sequences keyed by `specimen_id`
- **Morphometric TSV**:
  - vectorized morphometric or measurement features keyed by `specimen_id`

These inputs are optional and can be mixed freely.

### 2. Open-set recognition and calibration

v24 includes explicit support for **novel-species-aware evaluation**, including:

- species-holdout validation
- known vs. novel split generation
- retrieval-based open-set evaluation
- novelty threshold calibration
- ROC/AUC reporting
- threshold recommendations at target false-accept rates

### 3. Multi-view zero-shot prediction

You can classify a specimen using multiple images at once by passing a JSON mapping:

```json
{
  "specimen_001": ["img1.jpg", "img2.jpg", "img3.jpg"],
  "specimen_002": ["img4.jpg", "img5.jpg"]
}
```

DINOSAR aggregates the views to produce specimen-level predictions.

### 4. Safer memory-bank handling

Prediction mode supports:

- loading an existing memory bank
- saving to a new timestamped memory bank by default
- preventing accidental overwrite
- optional read-only inference mode

### 5. COCO mask support

The dataset loader can use a COCO JSON file containing masks for:

- full foreground
- a named body region
- a specific sclerite or structure

This makes it easier to constrain training to biologically relevant regions.

---

## Repository structure

```text
DINOSAR_v2/
├── README.md
├── environment.yml
├── LICENSE
├── .gitignore
└── DINOSAR_v2_contrastive_species_learning_CE_v24.py
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

### 3. Verify installation

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python DINOSAR_v2_contrastive_species_learning_CE_v24.py --help
```

> **GPU note:** GPU training is strongly recommended for large image datasets. CPU execution is possible, but much slower.

---

## Input data

DINOSAR v24 accepts either:

### Option A: directory-structured images

A directory where images are organized by species:

```text
data/
├── species_A/
│   ├── specimen1_H.jpg
│   ├── specimen1_D.jpg
│   └── specimen2_H.jpg
├── species_B/
│   ├── specimen3_H.jpg
│   └── specimen4_D.jpg
```

### Option B: flat image directory + metadata file

A flat image directory plus a metadata file in `.csv`, `.tsv`, or `.json` format containing species/specimen/view assignments.

Typical metadata columns are:

- `image_path`
- `species`
- `specimen_id`
- `view_id`

---

## Optional auxiliary inputs

### 1. Trait TSV

A tab-delimited file keyed by specimen ID.

Recommended conventions:

- columns starting with `cat_`, `has_`, or `is_` → categorical/binary
- columns starting with `count_` or `n_` → meristic counts
- other numeric columns → continuous traits

Example:

```tsv
specimen_id	pronotum_width	count_spines	cat_wing_spot
sp1	1.24	3	1
sp2	1.10	2	0
```

### 2. DNA FASTA

FASTA headers should begin with the `specimen_id`:

```fasta
>sp1
ATGCGT...
>sp2
ATGCGC...
```

The first token in the header is used as the specimen ID key.

### 3. Morphometric TSV

A tab-delimited file keyed by specimen ID containing vectorized morphometric features:

```tsv
specimen_id	pc1	pc2	pc3	centroid_size
sp1	0.23	-1.11	0.58	2.43
sp2	0.10	-0.92	0.44	2.10
```

### 4. COCO mask JSON

Optional COCO JSON file with masks for either:

- full foreground
- a chosen category such as `head`, `propodeum`, etc.

---

## Typical workflows

### 1. Train a basic image-only contrastive DINOSAR model

```bash
python DINOSAR_v2_contrastive_species_learning_CE_v24.py \
  --data-dir data/images \
  --output-dir output/run_v24 \
  --dinov3-local-ckpt /path/to/dinov3_checkpoint.pth \
  --epochs 50 \
  --batch-size 16
```

### 2. Train with metadata file

```bash
python DINOSAR_v2_contrastive_species_learning_CE_v24.py \
  --data-dir data/images \
  --metadata-file data/specimens.tsv \
  --output-dir output/run_v24 \
  --dinov3-local-ckpt /path/to/dinov3_checkpoint.pth
```

### 3. Train with optional classifier head

```bash
python DINOSAR_v2_contrastive_species_learning_CE_v24.py \
  --data-dir data/images \
  --metadata-file data/specimens.tsv \
  --output-dir output/run_v24_ce \
  --dinov3-local-ckpt /path/to/dinov3_checkpoint.pth \
  --use-classifier-head \
  --ce-weight 1.0
```

### 4. Train with traits, DNA, and morphometrics

```bash
python DINOSAR_v2_contrastive_species_learning_CE_v24.py \
  --data-dir data/images \
  --metadata-file data/specimens.tsv \
  --output-dir output/run_v24_multimodal \
  --dinov3-local-ckpt /path/to/dinov3_checkpoint.pth \
  --trait-tsv data/traits.tsv \
  --dna-fasta data/coi.fasta \
  --morph-tsv data/morphometrics.tsv \
  --trait-weight 0.5 \
  --dna-weight 0.5 \
  --xmodal-weight 0.3
```

### 5. Open-set evaluation with species holdout

```bash
python DINOSAR_v2_contrastive_species_learning_CE_v24.py \
  --data-dir data/images \
  --metadata-file data/specimens.tsv \
  --output-dir output/run_v24_openset \
  --dinov3-local-ckpt /path/to/dinov3_checkpoint.pth \
  --species-holdout-fraction 0.2 \
  --open-set-eval \
  --split-strategy per_species \
  --exclude-singletons \
  --singletons-as-novel \
  --export-split-tsvs
```

### 6. Zero-shot prediction on new images

```bash
python DINOSAR_v2_contrastive_species_learning_CE_v24.py \
  --data-dir data/images \
  --output-dir output/run_v24 \
  --dinov3-local-ckpt /path/to/dinov3_checkpoint.pth \
  --load-checkpoint output/run_v24/best_model.pt \
  --predict new_images/specimen1_H.jpg new_images/specimen1_D.jpg \
  --memory-bank-in output/run_v24/memory_bank.json
```

### 7. Multi-view specimen-level zero-shot prediction

```bash
python DINOSAR_v2_contrastive_species_learning_CE_v24.py \
  --data-dir data/images \
  --output-dir output/run_v24 \
  --dinov3-local-ckpt /path/to/dinov3_checkpoint.pth \
  --load-checkpoint output/run_v24/best_model.pt \
  --predict-multi-view-json data/specimen_to_images.json \
  --memory-bank-in output/run_v24/memory_bank.json \
  --store-all-similarities
```

### 8. Use COCO masks

```bash
python DINOSAR_v2_contrastive_species_learning_CE_v24.py \
  --data-dir data/images \
  --metadata-file data/specimens.tsv \
  --output-dir output/run_v24_masked \
  --dinov3-local-ckpt /path/to/dinov3_checkpoint.pth \
  --coco-mask-file data/masks.json \
  --coco-mask-category foreground
```

---

## Important command-line options

### Core data/model options

- `--data-dir` : root image directory
- `--metadata-file` : CSV/TSV/JSON metadata file
- `--output-dir` : output folder
- `--dinov3-model` : DINOv3 architecture
- `--dinov3-local-ckpt` : local checkpoint path
- `--projection-dim` : embedding dimension

### Backbone control

- `--freeze-backbone` / `--no-freeze-backbone`
- `--use-cls-token` / `--no-use-cls-token`
- `--unfreeze-last-n-blocks`
- `--backbone-lr`

### Training control

- `--epochs`
- `--batch-size`
- `--lr`
- `--weight-decay`
- `--loss` (`supcon` or `triplet`)
- `--supcon-label-mode` (`species`, `specimen`, `both`)
- `--supcon-weight`
- `--use-classifier-head`
- `--ce-weight`

### Preprocessing

- `--no-remove-bg`
- `--no-crop`
- `--target-size`
- `--mask-erode-px`
- `--coco-mask-file`
- `--coco-mask-category`

### Open-set / validation

- `--species-holdout-fraction`
- `--open-set-eval`
- `--open-set-topk`
- `--open-set-far-targets`
- `--open-set-eval-every`
- `--split-strategy`
- `--per-species-val-fraction`
- `--min-specimens-for-val`
- `--min-train-specimens-per-species`
- `--exclude-singletons`
- `--singletons-as-novel`
- `--export-split-tsvs`
- `--memory-bank-scope`

### Auxiliary modality options

- `--trait-tsv`
- `--trait-specimen-id-col`
- `--trait-weight`
- `--dna-fasta`
- `--dna-max-seq-len`
- `--dna-weight`
- `--morph-tsv`
- `--morph-specimen-id-col`
- `--xmodal-weight`
- `--xmodal-temperature`

### Prediction / memory bank options

- `--predict`
- `--predict-multi-view-json`
- `--similarity-threshold`
- `--memory-bank-in`
- `--memory-bank-out`
- `--overwrite-memory-bank`
- `--no-save-memory-bank`
- `--predictions-tsv`
- `--predictions-mv-tsv`
- `--store-all-similarities`

---

## Main outputs

Depending on the mode and options used, DINOSAR v24 writes outputs such as:

| File                                                                    | Description                                  |
| ----------------------------------------------------------------------- | -------------------------------------------- |
| `run_metadata.yaml`                                                     | Full run configuration and hardware metadata |
| `training_progress.png`                                                 | Training summary plot                        |
| `memory_bank.json`                                                      | Species memory bank built from embeddings    |
| `zero_shot_predictions.tsv`                                             | Per-image zero-shot predictions              |
| `zero_shot_predictions.json`                                            | Per-image zero-shot prediction details       |
| `zero_shot_multi_view_predictions.tsv`                                  | Specimen-level multi-view predictions        |
| `zero_shot_multi_view_predictions.json`                                 | Detailed specimen-level predictions          |
| `view_metrics_epoch_*.json`                                             | Per-view evaluation summaries                |
| `species_clusters_tsne.png`                                             | t-SNE of discovered species clusters         |
| `specimen_tsne_by_species.png`                                          | t-SNE colored by species                     |
| `open_set_metrics_*.json`                                               | Open-set evaluation results                  |
| `open_set_novelty_roc_epoch_*.png`                                      | Known-vs-novel ROC plots                     |
| `verification_roc_epoch_*.png`                                          | Pairwise verification ROC                    |
| `open_set_debug_epoch_*.tsv`                                            | Open-set debug table                         |
| `conflicted_specimens.tsv`                                              | Difficult or ambiguous specimen assignments  |
| `train.tsv` / `val_known.tsv` / `val_novel.tsv` / `test_singletons.tsv` | Exported dataset splits if enabled           |

If attention/mask visualization is enabled, additional overlays, masks, heatmaps, and `.npz` arrays are also written.

---

## Integration with Descriptron

DINOSAR v24 is designed to complement the [Descriptron](https://github.com/alexrvandam/Descriptron) platform.

Typical use cases include:

1. annotate specimens in Descriptron
2. export masks, keypoints, or measurements
3. derive trait or morphometric TSV files
4. train DINOSAR with image embeddings plus optional morphology-derived supervision

This makes it possible to combine image-based species recognition with structured phenotypic information in a single workflow.

---

## Notes on missing data

A major design feature of v24 is that **auxiliary modalities are optional on a per-specimen basis**:

- specimens without traits still contribute to image training
- specimens without DNA still contribute to image training
- specimens without morphometrics still contribute to image training

Only the losses corresponding to available data are applied.

This makes the framework practical for real museum and biodiversity datasets where data completeness varies across specimens.

---

## Citation

If you use this software in your research, please cite the GitHub repository and Zenodo archive for the version you used.

Replace the placeholder below with your final Zenodo DOI/version-specific citation:

```text
Van Dam, A. R. (2026). DINOSAR v2: DINOv3 Species Auto-Recovery version 2, version 0.24.0 [Computer software]. GitHub. https://github.com/alexrvandam/DINOSAR_v2
```

---

## Acknowledgements

DINOSAR v2 was developed at the **Museum für Naturkunde Berlin** within the **Centre for Integrative Biodiversity Discovery**.

Special thanks to collaborators and specimen providers contributing to the development and testing of automated workflows for hyperdiverse insect taxa.

---

## Additional software and literature to cite

Please also cite key dependencies and related methods where appropriate, including:

- DINOv2
- DINOv3
- PyTorch
- rembg
- Descriptron

```text
# *DINOSAR v2 — **DINO**v3 **S**pecies **A**uto-**R**ecovery version 2: Multi-modal Bayesian species delimitation and recognition for hyperdiverse insect taxa*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)

**DINOSAR v2** — **DINO**v3 **S**pecies **A**uto-**R**ecovery version 2, is a multi-modal, Bayesian framework for automated species delimitation in hyperdiverse insect taxa. It fuses three independent evidence streams — vision embeddings (DINOv3), morphological measurements (from Descriptron COCO JSON), and DNA barcodes (COI FASTA) — into a unified adaptive Bayesian classifier that scales to dark taxa with few labelled specimens.

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

```text
Van Dam, A. R. (2026). *DINOSAR v2 — DINOv3 Species Auto-Recovery version 2: Multi-modal Bayesian species delimitation and recognition for hyperdiverse insect taxa*. [Computer Software] Accessed from: https://github.com/alexrvandam/DINOSAR_v2.DINOSAR. https://doi.org/10.5281/zenodo.18935714
```

# Additional Literature and Software to Cite if you use this project:

```text
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
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---


