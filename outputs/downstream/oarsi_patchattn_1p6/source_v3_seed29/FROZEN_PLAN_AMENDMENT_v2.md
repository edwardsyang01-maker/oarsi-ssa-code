# Frozen plan amendment v2: where lambda is calibrated (2026-11-14)

The original frozen plan `FROZEN_PLAN.md` (SHA b72ae918…0717) is unchanged. This amendment only replaces the location where gradients are measured in the "lambda calibration" section.

## Why

The v1 preflight (attempt01) locked lambda at P2 = 0.541 and P3 = 10.14 (P4 reuses P3).

v1 compared the gradient norms of the main loss and the OARSI loss on the outputs of all 789 tokens.

- **P2 is not affected**: both losses read only CLS and are compared at the same location, so the result is reliable.
- **P3 is overestimated**: OARSI reads the attention-pooled result. At initialization attention is nearly uniform, so the gradient is spread over 784 patches and the per-token norm is small. On the shared parameters of the last 6 layers, however, these components add up coherently. Lambda may therefore be overestimated by about an order of magnitude (reasoning, not yet measured).

**User decision (2026-11-14)**: "Start P0 now, fix λ".

## What changed

- **Measurement location**: the **trainable encoder parameters shared** by the main loss and the OARSI loss, i.e. the last 6 Transformer blocks plus the final LayerNorm. This is exactly where the two gradients are added during training.
- **Computation**:
  - Capture the input of the first trainable block and recompute the last 6 blocks on an FP32 copy.
  - Take the gradients of both losses with respect to the tokens, normalized over the whole batch as in v1.
  - Accumulate the gradients on the shared parameters with vector-Jacobian products in groups of 8 images.
  - A unit test shows this equals a single backward pass over the whole batch.
- **Lambda formula unchanged**: lambda = 0.25 x median(||g_main|| / ||g_oarsi||), over 1,024 fixed training images, 16 batches per GPU, median over both GPUs combined.
- **Reference record**: the v1-style ratio and the relative error of the FP32 recomputation (required < 0.05) are also recorded.
- **Prespecified check before automatic queuing**: 0.1 <= lambda_P3 / lambda_P2 <= 10 is required. Outside that range the run stops and waits for the user, to guard against a new measurement error.

## What did not change

- Model, loss, OARSI labels, augmentation, random flip and swap, learning rates, epochs, gates and the epoch selection rule are unchanged.
- Seed 17 only; test is still not read.

## Run plan

- **P0**: lambda = 0, independent of calibration. Trained with source_v1, output attempt01/train_P0, used as the baseline for the v2 groups.
- **P3, P1, P2, P4**: recalibrated with source_v2 and trained in this order after preflight, output attempt02_lambda_v2.
- **P1 also uses v2**: P1 also has lambda = 0 but uses v2 to keep the later queue uniform. The training-path code of v1 and v2 is identical; only the calibration function and version binding differ.
- **The primary comparison is still P3 - P0** with unchanged gates. Because P0 and P3 use different source versions (differing only in calibration), this is noted in the report.
