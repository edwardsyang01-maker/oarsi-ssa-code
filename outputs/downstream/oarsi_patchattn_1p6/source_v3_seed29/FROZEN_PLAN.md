# OARSI auxiliary task + patch attention pooling: experiment plan on 1.6x images (frozen v1, 2026-11-13)

Status: **frozen (user, 2026-11-13: "Freeze and start")**. Formal training still needs separate user authorization. Test is not read. Any change after freezing must be saved as a new version with the reason stated.

## 1. Questions

In group A (2.0x), 81% of errors fall among KL0/1/2. Of the cross-two-grade errors, 68% are KL2 predicted as KL0. This experiment tests two things, and their combination:

1. **OARSI site labels (what to look at)**: does auxiliary supervision with osteophyte and joint-space-narrowing grades improve KL grading?
2. **Patch attention pooling (where to look)**: does selecting important regions from the 784 local tokens, in addition to CLS, improve KL grading?

When combined, each site has its own query vector, and the label of that site supervises where it looks.

## 2. Why 1.6x, and the constraints that follow

- **Resolution vs. coverage**: with a fixed 448 input, a 2.0x crop gives each pixel a larger area, so small osteophytes are harder to see; 1.3x may cut off context at the edges. 1.6x lies in between. This is a rationale, not a verified result.
- **A new 1.6x baseline is required**: 1.6x images have never been trained on, so group A at 2.0x cannot serve as the control. All comparisons in this plan are within 1.6x (P0 vs P1/P2/P3).
- **Data**: the full usable 1.6x list `chen_yolo_usable448_1p6x_clean_v1_20261112`.
  - Train 32,256 knees / 3,537 patients; validation 3,962 knees / 435 patients.
  - Train KL0-4: 12,837 / 6,173 / 7,931 / 4,176 / 1,139.
  - Visits V00-V10, all fields present (checked read-only).
  - 32,256 / 64 = 504 steps per epoch; 20 epochs, 10,080 steps in total.

## 3. Label preparation (step 0)

Use the existing `build_oarsi_labels.py` with the 1.6x source list, as a separate version (v1p6), writing to `data/manifests/chen_yolo_1p6x_oarsi_aux_v1_20261113/`. Rules are the same as for the 2.0x version:

- **Row selection**: reading project 15 for V00-V06, project 37 for V08/V10 (project 42 recoded as 37); project 43 is not used. Each knee must have exactly one row, and KL must equal the list label; otherwise an error is raised.
- **6 targets**:
  - Osteophytes at 4 sites: medial femur OSFM, lateral femur OSFL, medial tibia OSTM, lateral tibia OSTL, each grade 0-3.
  - Joint space narrowing at 2 sites: medial JSM, lateral JSL, each grade 0-3.
- **Missing values**: missing osteophyte grades stay empty and are skipped with a mask, not filled with 0; missingness is not a model input.
- **Fractional codes**: the integer part is used and the original value kept. Verified on 2026-11-13 on page 7 of `OAI/OAICompleteData_SAS/kXR_SQ_BU_Descrip.pdf` (2016-04-29 version), read-only on the NAS: at follow-up, x.2/x.4/x.6/x.8 mean the OARSI grade is unchanged from the previous time point but the reader saw definite worsening within the grade. The decimal does not encode the degree of narrowing; the integer part is the OARSI grade.
- **Why osteophytes are missing** (same document): osteophytes and other features were read only if the participant had KL>=2 in either knee at any time point; otherwise they are coded as the special missing value .A ("no data expected").
- **Expected coverage**: similar to the 2.0x version. Narrowing about 100%; all four osteophyte sites about 54%, of which about 17% at KL0, 41% at KL1 and 97% at KL2-4. Actual numbers after generation take precedence.

## 4. Model

Backbone, preprocessing and frozen layers are identical to group A: DINOv3 ViT-L/16, input 448, first 18 layers frozen, only the last 6 blocks, the final norm and the heads trained.

```
image [B,1,448,448] -> encoder -> last_hidden_state [B, 1+R+784, 1024]
  CLS     = token 0                         [B,1024]
  patch   = last 784 tokens                 [B,784,1024]   R = number of register tokens, read from the config and asserted in preflight
```

- **Attention pooling module**: one multi-head cross-attention layer, 8 heads, dimension 1024.
  - Queries: learnable Q of shape [K,1024], K = 1 (KL only) or 7 (KL + 6 sites).
  - Keys: LayerNorm(patch) + learnable 2-D position embedding [784,1024].
  - Values: LayerNorm(patch).
  - Output: pooled features [B,K,1024].
  - About 4.2M new parameters (attention) plus 0.8M (position embedding).
- **Left/right orientation (decided by the user on 2026-11-13: flip the input image)**: by the list convention, image left is the right knee and image right is the left knee; the films are not flipped. On PA films, the medial side of the right knee is on the right of its crop, and the medial side of the left knee is on the left of its crop.
  - Method: for knees with `clinical_side=L`, the image [1,448,448] is flipped along the width before input. After flipping, every knee is shown in right-knee orientation with the medial side on the image right. All of P0-P3 do this, for both training and validation.
  - This differs from the input convention of earlier experiments. P0 is a new 1.6x baseline and is not compared directly with group A at 2.0x.
  - Risk: the list marks `individual_anatomical_side_verified=false`, so a few sides may be wrong. Attention maps are spot-checked after the experiment.
- **Training augmentation and random horizontal flip (decided by the user on 2026-11-13)**: all augmentations of the group A recipe are kept: random horizontal flip (probability 0.5), brightness/contrast/gamma +-10%, rotation +-5 degrees, translation +-3%, and mild blur with probability 15%.
  - Order: deterministic flip of left knees first, then random augmentation.
  - P0-P3: when an image is randomly flipped, its OARSI medial/lateral labels are swapped at the same time (OSFM<->OSFL, OSTM<->OSTL, JSM<->JSL). The KL label is unchanged.
  - P4 (see section 6): labels are **not swapped** after a random flip, to test whether the model can tell medial from lateral from anatomy.
  - Validation has no random augmentation, only the deterministic flip of left knees.
- **KL head**:
  - P0 / P2: Linear(CLS), output [B,5], as in group A.
  - P1 / P3: Linear(concat(CLS, KL pooled feature)), input [B,2048], output [B,5].
  - The pooled half of the weights is initialized to 0 and the CLS half with the same seed as P0. At step 0 the outputs of P1/P3 therefore equal P0 exactly, and group differences come only from training.
- **OARSI heads** (6, each Linear(1024->4)):
  - P2: on CLS (auxiliary task without pooling).
  - P3: each on the pooled feature of its site (for example the medial tibial osteophyte query -> OSTM head).

## 5. Loss

```
L = mean_b{ w[y_b] * (CE_b + 0.5*RPS_b) }                     identical to group A
  + lambda * mean_b{ sum_t m_bt * CE(oarsi_t) / max(1, sum_t m_bt) }   only in P2/P3; m_bt=1 if the knee has a reading for target t
```

- The OARSI cross-entropy has no class weights at first, to avoid changing too much at once.
- **Choosing lambda**: calibrated in preflight following project practice. With 1,024 fixed training images, forward only, take the norm ratio of the auxiliary gradient to the main gradient at full weight on CLS and the pooling input, set its median to 0.25, then lock it. 0.25 is larger than the 0.1 used for RNC because OARSI is directly related to KL and the risk of conflict is lower.
- The first 4 epochs ramp up to full weight (0.25 / 0.5 / 0.75 / 1.0 x lambda) to reduce interference with the main task early on.

## 6. Groups (2x2 factorial)

| Group | Patch attention pooling | OARSI auxiliary | Question |
|---|---|---|---|
| P0 | no | no | 1.6x baseline (group A recipe) |
| P1 | yes (KL query only) | no | does pooling alone help |
| P2 | no | yes (on CLS) | do site labels alone help |
| P3 | yes (7 queries) | yes (on site pooling) | both combined; main hypothesis |
| P4 | as P3 | as P3, but medial/lateral labels are **not swapped** on random flips | can the model learn medial vs lateral itself (additional exploratory group, added by the user on 2026-11-13) |

P4 is compared only with P3; the only difference is whether labels are swapped after a random flip. P4 is not part of the main gates in section 8.

Apart from the differences above, everything is fixed as in group A: AdamW, encoder peak learning rate 1e-5 and heads 1e-3, layer-wise learning-rate decay 0.75, weight decay 0.05, 1 warmup epoch, 20 epochs, global batch 64, BF16 + ZeRO-3 on two GPUs; class weights, loss and augmentation unchanged; data order set by the seed. New modules (pooling layer, position embedding, OARSI heads) use the head learning rate of 1e-3.

## 7. Seeds and staged execution

**Decided by the user on 2026-11-13: seed 17 only for now**, five groups P0-P4, about 5 x 6 hours = 30 GPU hours (estimate; preflight measurements take precedence). Order: P0, P3, P1, P2, P4, main comparison first. Can start only after the current B2/C2 queue ends (expected early tomorrow morning).

Consequence (must be stated in the conclusions): QWK varies by about 0.005-0.01 between epochs, so single-seed results are exploratory evidence only, not confirmatory. The "3-seed mean" below is the single seed 17 value in this round, and the patient-clustered bootstrap reflects only validation sampling error, not training randomness. Whether to add seeds 23/29 is decided later; if added, it must be stated that the decision came after seeing the seed 17 results.

## 8. Evaluation and decision

**Primary metric**: validation QWK, summarized in two ways per run:
1. Selected epoch: by the original rule, the epoch with the highest raw QWK, the earlier one on ties.
2. Plateau mean: mean QWK over epochs 11-20. It does not depend on the luck of epoch selection and serves as a robustness reference.

**Primary comparison**: P3 - P0, mean over 3 seeds; in addition, the validation predictions of all seeds are pooled and a patient-clustered bootstrap (10,000 replicates) gives the 95% interval of the difference.

**Gates** (taken from the earlier RNC experiment, as exploratory gates):
- QWK gain (strict version set by the user on 2026-11-13): selected-epoch QWK difference >= 0.003 **and** plateau (epochs 11-20) QWK difference >= 0.003;
- KL>=2 AUC drops by no more than 0.005;
- cross-two-grade error rate rises by no more than 0.01.

**Secondary metrics** (reported only, not gates): KL1 F1, precision and recall; number of KL2->KL0; confusion matrix; MAE; ECE; validation QWK of each OARSI head (to check that the auxiliary task is learned).

**Attribution** (from the factorial design):
- pooling main effect = (P1 - P0 + P3 - P2) / 2
- auxiliary main effect = (P2 - P0 + P3 - P1) / 2
- interaction = P3 - P2 - P1 + P0

With 3 seeds these are descriptive only, without significance claims.

**Attention map check** (qualitative): draw the attention of the 7 queries on 32 fixed validation images, and check whether site queries fall near the corresponding bone margins or joint space, and whether left and right knees are mirror-consistent. Focus on KL2->KL0 cases: does the model look at osteophyte locations?

**Evaluation of P4 (exploratory, reported only)**:
- **Validation performance of each OARSI head**: compare QWK of the medial and lateral heads with P3. Validation has no random flip, so both groups have labels in the correct anatomical orientation.
- **Flip consistency test**: flip each validation image horizontally once more and predict again.
  - If P4 has learned anatomical orientation, predictions of the "medial" head should hardly change.
  - P3 learns "a fixed side of the image", so medial and lateral predictions should swap.
  - Quantified as the fraction of predictions that agree between the original and the flipped image.
- **Ruling out a shortcut**: medial narrowing is much more common than lateral, so the model might learn only that the narrower side is medial. Knees with a higher lateral than medial grade are therefore counted separately, to see whether P4 separates medial and lateral correctly on them.
- The KL QWK of P4 is also recorded but is not used for decisions.

**Not allowed**: reading test; reselecting epochs by OARSI metrics or AUC; changing gates, lambda or group definitions after seeing results.

## 9. Preflight (all must pass before authorization is requested)

1. Generate the 1.6x OARSI labels and check: same keys and order as the list, matching KL, grades within 0-3, coverage reported by KL.
2. Unit tests:
   - output shape of each pooling layer;
   - OARSI loss is 0 when the mask is all 0;
   - left-knee grid flip is correct, and two flips give back the original;
   - at step 0 the logits of P1/P3 are bit-identical to P0;
   - the number of register tokens matches the config and exactly 784 patch tokens are taken.
3. Calibrate and lock lambda: both GPUs agree, and the model, optimizer and RNG states are unchanged by calibration.
4. Save on synthetic data for all four groups -> bit-identical resume in a new process; record peak memory and time per step.
5. Disk: about 10 GB of checkpoints per run (two same-disk copies + the last 3 + the best). The data directory has 19 TB free, which is enough. The two copies are on the same disk and are not independent disaster recovery.

## 10. Risks and limitations

- **Osteophyte labels are sparsest where it matters most**: most KL0/1 knees have no osteophyte reading, so auxiliary supervision comes mainly from KL>=2 knees. Its direct help at the KL0/1 boundary may be limited; narrowing labels are the main auxiliary signal there.
- **Correlated noise**: OARSI and KL come from the same reading, so label errors occur together. The auxiliary task cannot resolve label ambiguity.
- **Sides not verified per image**: if the side of some knees is wrong, site queries learn the wrong location.
- **1.6x and 2.0x results are not directly comparable**: the difference between P0 and group A mixes the crop factor and the cleaning pipeline; it is a reference at most.
- **All conclusions are on the development set**: validation is used repeatedly for epoch selection, so final conclusions need an independent evaluation.

## 11. Decisions

Decided (user, 2026-11-13):

- Left/right orientation: flip the input image.
- Lambda calibration ratio: 0.25.
- Seeds: seed 17 only for now.
- Gates: strict version; both the selected-epoch difference and the plateau difference must be >= 0.003.

- Random horizontal flip: P0-P3 keep the random flip and swap the OARSI medial/lateral labels; P4 is added, without swapping after flips, to test whether the model learns medial vs lateral itself.
- Fractional narrowing codes: take the integer part (verified against the official documentation).
- All defaults adopted: no class weights in the OARSI cross-entropy; learning rate 1e-3 for new modules; lambda ramps to full weight over the first 4 epochs; epoch selection rule unchanged.

**All design questions are settled.** The discussion record below is kept for reference:

1. **Random horizontal flip augmentation** (found on 2026-11-13): the group A recipe includes `horizontal_flip_probability=0.5`, plus brightness/contrast/gamma +-10%, rotation +-5 degrees, translation +-3% and mild Gaussian blur with probability 15%. Random flips swap medial and lateral.
   - Option a (recommended): keep the random flip. For flipped samples, swap the OARSI medial/lateral labels at the same time (OSFM<->OSFL, OSTM<->OSTL, JSM<->JSL). The augmentation of all four groups then equals the group A recipe, and site queries learn "the compartment on a fixed side of the image". Validation has no random flip; after the deterministic flip of left knees, the image right is always medial.
   - Option b: turn off the random horizontal flip in all four groups and keep the other augmentations. Simpler, but changes the augmentation recipe.
2. **Lambda calibration ratio**: 0.25.
3. **Seed commitment**: four groups x 3 seeds (about 72 hours), or 3 seeds for P0 and P3 only and seed 17 for P1/P2 (about 40 hours)?
4. **Gate version**: the recommended version is selected-epoch difference >= 0.003 and a positive plateau difference. A stricter version (both differences >= 0.003) and a looser one (selected epoch only) were also considered.
5. **Fractional narrowing codes** (about 3% of knees): **resolved**. The official documentation confirms the decimal means worsening within the same grade and the integer part is the OARSI grade, so taking the integer part is correct and no masking is needed.

Defaults (adopted unless objected to): no class weights in the OARSI cross-entropy; learning rate 1e-3 for new modules; lambda ramps to full weight over the first 4 epochs; epoch selection rule unchanged.
