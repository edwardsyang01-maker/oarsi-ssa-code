# OARSI-SSA code

Code for the paper **"Automated OARSI grading of knee osteoarthritis radiographs with site-specific attention"**. Code author: Cunhao Yang.

The model grades the Kellgren-Lawrence (KL) score of a single knee from a radiograph crop. It uses a DINOv3 ViT-L/16 encoder at 448 x 448, auxiliary supervision with OARSI osteophyte and joint-space-narrowing grades, and patch attention pooling with one query per anatomical site. The experiment is a 2 x 2 factorial design plus one exploratory group:

| Group | Patch attention pooling | OARSI auxiliary labels |
|---|---|---|
| P0 | no | no |
| P1 | yes (KL query only) | no |
| P2 | no | yes, on the CLS token |
| P3 | yes (KL + 6 site queries) | yes, on the site queries |
| P4 | as P3 | as P3, without swapping medial/lateral labels on random flips |

Seeds 17, 23 and 29. The frozen experiment plan and its amendments are in `outputs/downstream/oarsi_patchattn_1p6/source_v*/FROZEN_PLAN*.md`.

## Data

Images and labels come from the Osteoarthritis Initiative (OAI; ClinicalTrials.gov NCT00080171) and are available to registered researchers from the NIMH Data Archive (https://nda.nih.gov/oai). They are not included here. The code uses the bilateral PA fixed-flexion knee radiographs (DICOM), the central reading tables `kXR_SQ_BU` (KL and OARSI grades) and, for reader reliability, `kXR_SQ_Rel_BU`.

Patient-level train/validation/test splits, manifests, predictions and model weights are not included.

## Repository layout and pipeline

Paths keep the layout of the original project. Run the steps in this order from the project root.

| Step | Location | Entry points |
|---|---|---|
| 1. OAI metadata, PA candidates, patient split, KL labels, baseline QC | `scripts/`, `downstream/design/`, `downstream/preprocess/` | `import_metadata.py`, `build_xray_manifests.py`, `build_ssl_patient_split.py`, `build_three_task_candidates_v3.py`, `prepare_baseline_qc_v1.py` |
| 2. DICOM decoding and gray mapping | `src/drai/data/` | `dicom_gray.py`, `full_pa.py` |
| 3. Knee detection with the YOLOv2 of Chen et al. and 1.3x crops | `experiments/chen_yolo_crop_compare_v1/`, `experiments/chen_yolo_fullcrop_v1/`, `experiments/chen_yolo_repair_v1/`, `experiments/chen_yolo_448_v1/` | `fetch_assets.py`, `code/pipeline.py`, `finalize_usable.py`, `export_448.py` |
| 4. Ruler cleaning and 1.6x crops at 448 | `experiments/chen_yolo_expand*_v1/`, `experiments/chen_yolo_2x_clean_v1/`, `experiments/chen_yolo_1p6x_clean_v1/` | `pipeline16.py`, `release16.py` (publishes the train/val manifests) |
| 5. OARSI site labels | `experiments/oarsi_aux_labels_v1/` | `build_oarsi_labels_v2.py` |
| 6. Training | base trainer: `outputs/downstream/oa_kl_vitl_l5_reviewed_v3_training_v1/seed17/source_v1/`; paper models: `outputs/downstream/oarsi_patchattn_1p6/source_v1` (P0, seed 17), `source_v2` (P1-P4, seed 17), `source_v3_seed23`, `source_v3_seed29` (extra seeds) | `runner.py --group P0 ... P4`, configured by `spec.json`; `source_v4_*` and `source_v5_*` hold only the specs of the supplementary runs (training code identical to `source_v3`) |
| 7. Test-set preparation and inference | `outputs/downstream/oarsi_patchattn_1p6/test_eval_v1/` | `prep_code_v1/`, `inference_v1/`, `stats_v1/t5.py` |
| 8. Analyses | `outputs/downstream/oarsi_patchattn_1p6/` | `analysis_v1/code/wp1_primary.py` (primary comparison), `wp2_secondary.py`, `wp3_attention.py`, `wp4_reader_reliability.py`, `wp8_analysis/`, `wp8_x5/`, `pvalues_v1/`, `pvalues_flip_v2/`, `knee_bootstrap_v1/`, `threshold_v1/`, `results_v16_knee_v1/` |
| 9. Figures and tables | `outputs/downstream/oarsi_patchattn_1p6/manuscript_v24_build/scripts/` | `run_all.R` (reads the aggregate outputs of step 8 from `data/`) |

## Environment

- Python 3.12 with torch 2.11.0+cu128, transformers 4.57.6 and DeepSpeed 0.19.8. The full pinned list is in `requirements-lock.txt`; DeepSpeed is installed separately.
- Training used two GPUs with BF16 and DeepSpeed ZeRO-3.
- Figures and tables: R 4.4.3 with ggplot2 4.0.4, patchwork 1.3.2, officer 0.7.9 and flextable 0.10.5.

## Pretrained weights

- DINOv3 ViT-L/16 (`facebook/dinov3-vitl16-pretrain-lvd1689m`), under the DINOv3 License. Place it in `checkpoints/external/dinov3_vitl16_lvd1689m/`.
- YOLOv2 knee detector of Chen et al.: `experiments/chen_yolo_crop_compare_v1/code/fetch_assets.py` downloads the weights from https://data.mendeley.com/datasets/56rmx5bjcr/1 and the source code of https://github.com/PingjunChen/GradingKneeOA at commit dc19c2462111c3c5b27066e738fc11ddf57a6407. That code is not redistributed here.

## Notes

- Paths specific to the original servers were replaced with placeholders. Set them for your environment before running:

| Placeholder | Meaning |
|---|---|
| `/path/to/drai` | project root |
| `/path/to/workspace` | working directory outside the project |
| `/path/to/nas` | read-only share holding the OAI images |
| `/path/to/nas/drai_checkpoint_backups` | checkpoint backup directory |
| `<server>` | server host name |
| `<N>` | hardware counts and sizes |

- Many scripts check the SHA256 of their inputs and of their own source files against values recorded during the original runs. These checks fail on other systems or after edits; adjust them to your data.
- Docstrings were removed and only short comments on non-obvious code were kept. Messages, reports and documents were translated from Chinese to English.

## License

The code in this repository is released under the MIT License (see `LICENSE`). OAI data, the DINOv3 weights and the YOLOv2 detector of Chen et al. are not covered by this license and remain under their own terms.
