# Frozen plan amendment v3: adding seeds 23 and 29 for P0 and P3 (2026-11-15)

The original frozen plan `FROZEN_PLAN.md` (SHA b72ae918…0717) and amendment v2 (lambda calibration location) are unchanged. This amendment only relaxes the "seed 17 only" rule.

## Basis

- **Preliminary seed 17 results** (3,962 validation knees, computed by hand from the training records; the formal analysis script had not been run): P3 - P0 selected-epoch QWK +0.0080, mean over epochs 11-20 +0.0100, AUC +0.0016, cross-two-grade error +0.0036. All four prespecified gates passed.
- **User instruction** (2026-11-15): "add two seeds". Scope: P0 and P3, adding seed 23 and seed 29 for each, 4 runs in total, each 20 epochs / 10,080 steps.

## What changed

- **Seed**: `seed` in the spec is 23 or 29. It sets the classifier-head initialization, data order and augmentation random numbers.
- **Initialization of new components**: previously hard-coded as 170300 / 170301 / 170310+k / 170320, now `seed*10000 + 300 / 301 / 310+k / 320`. For seed=17 this is bit-identical to the old values; otherwise the P3 runs of all three seeds would share the same initial pooling and OARSI heads, underestimating between-seed variation.
- **Lambda is not recalibrated**: lambda is a hyperparameter and reuses the value locked for seed 17 in source_v2, P3 = 0.2949566469066166 (lock file SHA d584ba13…d5c0). Recalibrating per seed would give each seed a different configuration and mix hyperparameter differences into between-seed differences.
- **Run order**: seed 23 P0 -> P3, then seed 29 P0 -> P3. P0 and P3 use the same source version (source_v3).

## What did not change

- Model, loss, OARSI labels, augmentation, random flip and swap, learning rates, epochs, epoch selection rule and gates are unchanged.
- Test is still not read. P1, P2 and P4 get no extra seeds.

## How the results are used

- P3 - P0 is computed for each of the three seeds; the primary estimate is the arithmetic mean of the three differences, and the direction for each seed is reported. It is not the performance of an ensemble.
- The gates were set before the seed 17 results were seen; seeds 23 and 29 were added after seeing them, and the report must state this order.
- For seed 17, P0 used source_v1 and P3 used source_v2; their training-path code is identical and they differ only in the calibration function and version binding.
- Patient-clustered paired intervals and the test evaluation are outside this amendment and are decided separately.
