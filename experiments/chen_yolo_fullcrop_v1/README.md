# Full cropping with the author YOLOv2

On 2026-11-11 the user chose: "Crop with the author YOLOv2 model first, no retraining for now, and keep going." This stage only runs inference with fixed weights, engineering acceptance and preparation of review material.

## Scope and data

- Uses the frozen train set of 17,133 films / 3,542 patients and val set of 2,131 films / 435 patients, with the original patient split and same-side KL labels.
- Test images and labels stay sealed; no detector retraining or classifier training.
- Weights SHA256: `daa658b52e41ea7ad1493fc97ed34480d5dfb152e47c0bd296b187af2ceae95a`.
- Input, detection thresholds and cropping are the same as in stage 2: 0-255 float32 detector input; score 0.12, NMS 0.3, box expansion 1.3; no flipping; output `[1,224,224]` float32 with a bool valid mask.
- Image left linked to the right knee and image right to the left knee is the current PA position assumption, explicitly marked in the lists as pending per-case review; correct label linkage is not presented as manual confirmation of image orientation.
- Original NAS images are read-only and all derived material stays in the server data directory. New case images and patient lists are not copied to a local machine.

## Execution and acceptance

The per-case function of the new version only extends the allowed splits; its gray mapping, localization and cropping are identical to the earlier 304-case version. A stratified small batch, a regression on known anomalous cases and an explicit resume check come first; then a separate systemd user job runs full CPU inference. Each case commits its result and image SHA atomically; resuming requires an explicit `--resume`, and failed cases are not silently rerun.

After the full run, the SHA, shape/dtype, value range and label linkage of every ROI are read back, the boxes and output bytes of all original 304 cases are checked, and an internal server review book and an empty manual review template are generated. The number of detections with the fixed author model does not indicate localization accuracy or better grading performance.

## Outputs

- `outputs/yolo_candidates_train.csv` / `yolo_candidates_val.csv`: all YOLO candidates saved successfully.
- `outputs/yolo_no_auto_flags_train.csv` / `yolo_no_auto_flags_val.csv`: the subset without automatic ROI flags, still not clinical approval.
- `outputs/geometry_fallback_train.csv` / `geometry_fallback_val.csv`: separate old-method candidates where YOLO missed, not mixed into the main YOLO list.
- `outputs/issues_private.csv`: anomalous or missing knee sides; `human_review_template.csv`: manual review template, saved separately once filled in.
- `outputs/review_server_only/index.html`: review book blinded to KL labels, kept on the server only.
- `outputs/summary.json`, `verification.json`, `completion.json`: counts, engineering acceptance and background completion receipts.

All `training_ready` and `clinical_quality_approved` stay false. No automatic flag, successful saving and manual approval are recorded separately.

## Methodological limits

The public detector training directory of the author overlaps with some of our validation/test patients; without the full training log of these weights it cannot be concluded that KL labels leaked or that grading metrics are inflated. The current results are used for cropping and data preparation; the full pipeline is not claimed to be independently validated on entirely unseen patients. Retraining is one option to strengthen evaluation independence later; by the user's decision it is not done in this round.

Server experiment: `/path/to/drai/experiments/chen_yolo_fullcrop_v1`.

Server float ROI: `/path/to/drai/data/processed/chen_yolo_fullcrop_v1`.
