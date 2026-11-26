# Frozen plan amendment v5 (WP8 X2, X3): lambda sensitivity of P3 and extra seeds for P4 (2026-11-17)

The original frozen plan `FROZEN_PLAN.md` (SHA b72ae918…0717), amendment v2 (lambda calibration location), amendment v3 (extra seeds for P0 and P3) and amendment v4 (X1: extra seeds for P1 and P2) are unchanged. This amendment only adds two post hoc supplementary experiments.

## Basis

- **User instructions** (2026-11-17): in a side session the user said "add WP8" and selected X1, X2, X3 and X5 in the question box; in this session the user said "start training X2 and X3 directly once their preflights pass, no need to ask me again".
- **Timing**: both are supplementary experiments decided after seeing the validation results for seeds 17, 23 and 29 and the test results. The paper must state that they are post hoc; they do not change the WP1 primary comparison, the A/B/C/D grading or the main test conclusions.

## X2: lambda sensitivity of P3 (seed 17)

- **Purpose**: show that the P3 result does not depend on the exact value of lambda.
- **Groups**: P3, seed 17, two separate queues:
  - lambda x0.5: lambda_P3 = 0.1474783234533083, output `attempt05_x2_lam050`;
  - lambda x2: lambda_P3 = 0.5899132938132332, output `attempt05_x2_lam200`.
- **Where lambda comes from**: the lambda_P3 = 0.2949566469066166 locked for seed 17 in source_v2 (lock file `attempt02_lambda_v2/lambda_lock.json`, SHA d584ba13…d5c0) multiplied by 0.5 or 2. Both multiplications are exact in floating point. The new values are written to `wp8_x2x3/lambda_lock_x2_lamNNN.json` and bound by path and SHA256 through `lambda_fixed` in the spec. The P2 value stays unchanged in the lock file, but P2 is not trained.
- **Training code**: **byte-identical** to `source_v3_seed23`, `source_v3_seed29`, `source_v4_x1_seed23` and `source_v4_x1_seed29`; only the spec differs: seed 17, `run_order` set to `["P3"]`, and `lambda_fixed` pointing to the new lock file. With seed=17 this code is bit-identical to source_v2 (covered by existing unit tests), so lambda is the only difference from the original seed 17 P3 (`attempt02_lambda_v2/train_P3`).
- **Lambda warm-up** (0.25, 0.5, 0.75, 1.0 x lambda over the first epochs) is unchanged.

## X3: extra seeds 23 and 29 for P4

- **Purpose**: check whether the P4 result holds for other seeds.
- **Groups**: P4, seeds 23 and 29, outputs `attempt05_x3_seed23` and `attempt05_x3_seed29`.
- **Training code**: same as above, byte-identical; the spec is the `source_v3_seedNN` spec with only `run_order` changed to `["P4"]`, the output directory and the notes.
- **Lambda**: P4 reuses lambda_P3 locked for seed 17 (lock file unchanged) and is not recalibrated per seed, as in amendments v3 and v4.

## What did not change

- Model, loss, OARSI labels, augmentation, random flip and swap (P4 does not swap), learning rates, epochs (20 epochs / 10,080 steps), epoch selection rule and gates are unchanged.
- Test is not read, and no test inference is run for the new models (test was already used in WP6).

## Execution

- Each queue: CPU unit tests -> GPU preflight (synthetic save and cold resume, bit-identical on both GPUs) -> authorization -> training. Preflight needs idle GPUs, so it starts after X1 finishes normally.
- The four queues run in the order X2 lambda x0.5, X2 lambda x2, X3 seed 23, X3 seed 29. Training starts only when all four preflights pass; any failure stops the run, with no automatic retry.

## How the results are used

- **X2**: report, for the three P3 runs with lambda x0.5, x1 (the original seed 17 P3) and x2, the selected-epoch QWK, the E11-20 plateau QWK, the KL>=2 AUC and the cross-two-grade error rate relative to seed 17 P0; intervals use the same patient-clustered bootstrap as WP1 (default_rng(20261115)). One seed only; descriptive.
- **X3**: report P4 over three seeds together with seed 17 P4 (relative to P3 and P0), and flip consistency. Descriptive only.
- Both use validation only, have no gates and do not change the grading in section 6.2.
