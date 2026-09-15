# Predictions, registered before the run

Rules alone do not stop someone designing a test whose answer they already know.
So each run's expected outcome is written here **before** its results exist, with
a confidence and a reason, and the outcome is filled in afterwards without
editing the prediction. Over time this says whether the confident statements made
around this project are worth anything.

A prediction is not a hypothesis and carries no weight in any verdict. It is a
record of what was expected.

## step7b_sides.py — split-half replication of the IR52b routing

Registered 2026-09-16, rules at `467d670`, before the run.

Inputs and outputs were audited first (`step7b_sides.py --lookup`), and the audit
is the reason some of these confidences are not higher:

- every stimulus set is clean: 48 per wing half, 65 per leg half, no duplicate
  bodies, no missing `rootSide`, only the intended cell types;
- no stimulus body appears in any readout set;
- every stimulus moves the network: 6,700–14,500 neurons past 1e-4, peak
  responses around 2, convergence in 62 iterations;
- `rootSide` is missing for all 1,258 male-specific, all 156 pC1 and all 138
  dsx_high neurons, so laterality is not askable and the rules do not ask it;
- **the ppk23 and ppk25 leg draws overlap by 52–55 of 65**, because 65 is drawn
  from 79. Their ten draws are nearly one set, so the median over draws carries
  little independent information and R5 is weaker for those channels than its
  form suggests. IR52b's leg set is exactly 65, so it is drawn whole.

| rule | prediction | confidence | reason |
|---|---|---|---|
| R1 wing left half | pass | 0.75 | step 7's whole-wing IR52b sat at +0.766 against random seeds near 0.21–0.26; halving the driven set should not move a normalised share much |
| R2 wing right half | pass | 0.75 | same, and the two halves are 48 bodies each of one cell type |
| R3 dissociation within each half | pass | 0.85 | the whole-wing gap was 3.5x (0.766 vs 0.220 and 0.214), wide enough to survive a split |
| R4 the halves agree | pass | 0.65 | no reason for asymmetry, but the shuffled spread this is measured against has not been seen |
| R5 leg replication | pass | 0.55 | the hypothesis came from leg-containing data, but at a different stimulus size, and the share measure is not comparable across sizes |
| SPLIT-HALF REPLICATED | pass | 0.65 | R1 and R2 and R3 together |

Also registered, not a verdict: **ppk23's leg share will come out below its wing
share** in step 7's own leg columns, confidence 0.55. Step 6 saw -0.257 for a
mixed 17-body draw and step 7 saw +0.220 for 96 wing bodies, but those two are
not comparable — stimulus size changed with composition, and a ReLU network's
normalised share does not hold still across stimulus size. Step 7's leg columns
settle it at one size.

Outcome: _pending_

## In-flight runs whose outcomes were predicted in conversation

Recorded here because the predictions were made aloud while the runs were going,
which is the case where a record is worth most.

| run | prediction | confidence | reason |
|---|---|---|---|
| step 2b, T1 (loss falls to 0.8x its start) | fail | 0.95 | by step 28 of 40 the loss had moved 0.1901 to 0.1845, 2.9%, with the gradient at 4e-3 |
| step 2b, T2 (held-out gain, lower CI bound above 0) | fail | 0.7 | almost nothing moved, so the readout should barely change |
| step 3, S1 (trained balanced accuracy differs by more than 0.01) | fail | 0.7 | same reason: step 2b's parameters barely left their start |
| step 3, S0 (transcription matches score_shiu within 4 of 149) | pass | 0.8 | the untrained parameter set reduces to score_shiu's g = 0.9, b = 0.1 model exactly |

Outcome: _pending_
