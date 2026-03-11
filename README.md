# DINOSAR v2 — DINOv3 Species Auto-Recovery version 2

## `DINOSAR_v2_contrastive_species_learning_CE_v24.py`

&#x20;&#x20;

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)

**DINOSAR v2** — **DINO**v3 **S**pecies **A**uto-**R**ecovery version 2 — a DINOv3-based contrastive learning framework for species discovery, zero-shot classification, and open-set recognition in hyperdiverse insect taxa.

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


## Acknowledgements

DINOSAR v2 was developed at the **Museum für Naturkunde Berlin** within the **Centre for Integrative Biodiversity Discovery**.

Special thanks to collaborators and specimen providers contributing to the development and testing of automated workflows for hyperdiverse insect taxa.

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


