# PREDICTIONS — registered before the run

Rules alone do not stop someone designing a test whose answer they already know.
So each run's expected outcome is written here **before** its results exist, with
a name and a reason, and the outcome is filled in afterwards without editing the
prediction. Over time this says whether the confident statements made around this
work are worth anything.

From 2026-09-16 a prediction carries **no confidence number** — see `METHOD.md`.
PR-1 and PR-2 keep theirs, because a prediction is not edited after the fact.

A prediction is not a hypothesis and carries no weight in any verdict. It is a
record of what was expected. The runs themselves, their committed rules and their
results live in `~/repos/malecns-operator`; only the predictions live here.

---

## PR-1 — step7b split-half replication of the IR52b routing

Registered 2026-09-16, against rules committed at `malecns-operator@53a075e`,
before the run.

Inputs and outputs were audited first (`step7b_sides.py --lookup`), and the audit
is why some of these confidences are not higher:

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

**Outcome, 2026-09-16: all six correct.** R1 +0.787 against a null topping out at
+0.638; R2 +0.748 against +0.462; R3 holds in both halves; R4 with the halves
0.039 apart against a shuffled spread of 0.433; R5 on leg as well. Six calls, six
right, but six correlated calls on one run is not a calibration record — R5 at
0.55 came out right without that saying the confidence was wrong. The value of
this table only appears once some of it is wrong.

Also registered, not a verdict: **ppk23's leg share will come out below its wing
share** in step 7's own leg columns, confidence 0.55. Step 6 saw -0.257 for a
mixed 17-body draw and step 7 saw +0.220 for 96 wing bodies, but those two are
not comparable — stimulus size changed together with composition, and a ReLU
network's normalised share does not hold still across stimulus size. Step 7's leg
columns settle it at one size.

Outcome, 2026-09-16, from step 7's own leg columns at one size: **correct.**
ppk23's leg share is -0.130 against its wing share of +0.220. Worth noting
against the confidence of 0.55: the raw signs differ, but both sit *below* their
shuffled nulls, so the position that carries the meaning was the same on both
sensilla and only the unnormalisable raw number flipped. The prediction was right
for a reason narrower than the one it was made on.

The step7b verdicts themselves are still _pending_.

---

## PR-2 — in-flight runs predicted aloud

Recorded because these predictions were made in conversation while the runs were
going, which is the case where a record is worth most: a prediction stated during
a pre-registered run is exactly what erodes the discipline it is meant to protect.

| run | prediction | confidence | reason |
|---|---|---|---|
| step 2b, T1 (loss falls to 0.8x its start) | fail | 0.95 | by step 30 of 40 the loss had moved 0.1901 to 0.1840, 3.2%, with the gradient at 5e-3 |
| step 2b, T2 (held-out gain, lower CI bound above 0) | fail | 0.7 | almost nothing moved, so the readout should barely change |
| step 3, S1 (trained balanced accuracy differs by more than 0.01) | fail | 0.7 | same reason: step 2b's parameters barely left their start |
| step 3, S0 (transcription matches score_shiu within 4 of 149) | pass | 0.8 | the untrained parameter set reduces to score_shiu's g = 0.9, b = 0.1 model exactly |

**Outcome, 2026-09-16 01:40, step 2b baseline (40/40):**

- **T1 fail — correct at 0.95.** The cross loss went 0.1901 to 0.1816, 4.5% against
  the 20% the rule asked for.
- **T2 pass — WRONG, and it was called at 0.7.** Held-out balanced accuracy went
  0.619 to 0.656, a paired bootstrap gain of +0.037 with a 95% interval of
  [+0.012, +0.070], which excludes zero. Per-class recall rose on three of four:
  sugar 0.175 to 0.225, water 0.725 to 0.800, bitter 0.575 to 0.600, ir94e
  unchanged at 1.000.

The reasoning behind the T2 call was "almost nothing moved, so the readout should
barely change", and it conflated two different things: how far the loss fell and
whether what it learned generalised. A 4.5% drop in a cross-fitted loss was
enough to move held-out accuracy measurably. Thirty-four gains, trained on a loss
that never sees the held-out neurons, bought +0.037 — small, real, and not what
was expected.

This is the first wrong entry in this file, and it is the first entry that makes
the file worth keeping.

**Outcome, 2026-09-16, step 2b shuffled (40/40):** T1 fail, T2 fail, and
**T3 gain needs the wiring: True.** The shuffled network trained to no held-out
gain at all — balanced accuracy 0.262 to 0.244, interval [-0.055, +0.017] — while
the real network gained +0.037. T3 was not registered here, so there is nothing
to score; it is recorded because it is the control that makes step 2b's one
positive result mean anything.

**Outcome, 2026-09-16, step 3 (rules `malecns-operator@39f32b4`):**

- **S0 pass — correct at 0.8, and exactly.** The transcription agrees with
  `score_shiu.json`'s g = 0.9 baseline on **149 of 149**, not the 145 the rule
  allowed.
- **S1 fail — WRONG, and it was called at 0.7.** Balanced accuracy on the 149
  outcomes went **0.669 to 0.715, +0.046**, against a rule asking for more than
  0.01.

The reason given for the S1 call was "step 2b's parameters barely left their
start". They did not. The loss barely moved; the parameters moved a great deal —
alpha went from a flat 0.9 to [0.951, 0.570, 0.970, 0.531, 0.963], s from 1.000
to 1.339, b from 0.100 to 0.263. **This is the same conflation that made T2
wrong, made one entry later, in the same file, about the same run.** Twice now
"the loss hardly moved" has been used as a proxy for "nothing changed", and both
times what it was standing in for went the other way.

---

## LK-8 — lookup for step 8, the time axis on the escape pathway

Run 2026-09-16, before any step 8 rule is written. No step 8 comparison is
measured here. Recorded because the numbers below decide what a step 8 rule is
allowed to ask, and three of them say a whole class of rule is unreachable.

### The category holds what its name says

Input composition, whole MaleCNS, by share of the postsynaptic cell's input
synapses:

- **DNp01** (2 bodies, 43,478 in-synapses): **LC4 0.146, LPLC2 0.112** are the
  first and second inputs, ahead of DNp70 at 0.033. The published circuit is
  LC4 and LPLC2 onto the giant fibre, and that is what the data has.
- **LPLC2** (185 bodies): T5a–d **0.184** and T4a–d **0.127** together, all eight
  subtypes present with 0.029–0.053 each. The four-armed cross that radial
  motion opponency is built on has its substrate here.
- **LC4** (126 bodies): T2 0.147, TmY3 0.130, Tm4 0.099, Tm2 0.065. **No T4 or
  T5 above threshold.** LC4 is lobula, not lobula plate, so its velocity
  component does not arrive through direction-selective cells.
- **T4a–d** (6,861): Mi1 0.257, Tm3 0.117, Mi9 0.099, Mi4 0.053, CT1 0.051 — the
  published ON pathway. **T5a–d** (6,719): Tm9 0.165, Tm2 0.150, Tm1 0.108,
  CT1 0.093, Tm4 0.086 — the published OFF pathway.

### Scale and feasibility

Hex coordinates exist on 15 medulla-columnar types only (C2, C3, L1, L2, L3, L5,
Mi1, Mi4, Mi9, T1, Tm1, Tm2, Tm20, Tm4, Tm9), 23,720 bodies, 892 distinct
columns. **T4, T5, LPLC2, LC4 and DNp01 carry none.** T4 draws 0.430 and T5
0.467 of its input from hex-carrying bodies, so their position is inferable from
their inputs; LPLC2 draws 0.049 and LC4 0.117, so theirs is not, directly.

`rootSide` is absent on LPLC2, LC4, DNp01, T4 and T5 alike. **`somaSide` is
present** on all of them — LPLC2 L 94 / R 91, LC4 L 71 / R 55, DNp01 L 1 / R 1 —
so laterality is askable here, unlike step 7b.

### The null, from step 4's own readouts

Step 4 already read LC4 and LPLC2 at each disc radius. Baseline against the ten
shuffled seeds, right-eye disc, r = 2 to 18:

| readout | baseline r2 → r18 | shuffled max over seeds |
|---|---|---|
| DNp01_R | 0.0002 → 0.0061 | 0.0001 → 0.0044, peaking 0.0044 at r = 8 |
| LC4 | 0.0191 → 0.0477, flat from r = 6 | 0.0056 → 0.0220 |
| LPLC2 | 0.0051 → 0.0198 | 0.0052 → **0.0395** |
| DNp01_L, PSI, TTMn | 0.0000 → 0.0002 | — |

Three consequences, each of which rules out a rule someone would otherwise write:

1. **LPLC2's response level never clears its own shuffled null** — the shuffled
   maximum exceeds baseline at every radius above r = 2. Any step 8 rule asking
   LPLC2 to respond more strongly than shuffled is unreachable before it is run,
   which is step 4's E1 mistake in a new place.
2. **LC4 saturates by r = 6** (0.0461, 0.0470, 0.0471, 0.0471, 0.0477). A
   velocity rule read off LC4's steady level at large radii is reading a ceiling.
3. **DNp01's whole range is 0.0002 to 0.0061** and DNp01_L, PSI and TTMn sit at
   or below 0.0002. No absolute threshold belongs anywhere on this pathway; step
   8 takes rank tests against shuffled seeds and orderings, per `METHOD.md`.

### The known item, from the literature

Ache & von Reyn et al., *Current Biology* 2019: the giant fibre's looming
response is reproduced by **a linear function of angular velocity, supplied by
LC4, summed with a Gaussian function of angular size, supplied by LPLC2**, and
LPLC2 supplies the entire size component. This fixes what step 8's two arms are
and gives each a published shape to be compared against, not a threshold.

Two cautions attach to it. Currier & Clandinin, *Cell* 2025, compared 43 fly cell
types against connectomic prediction and found **receptive field size among the
worst-predicted properties** while orientation tuning was among the best — and
the LPLC2 arm is a size measurement. And Duan et al., NeurIPS 2025, already put
the "recover dynamics from structure with minimal parameter tuning" question to
the head-direction circuit, tuning gains, thresholds and time constants at the
cell-type level, so step 8's framing is not new; its circuit, its dataset and
its direction — counting how little suffices rather than fitting until it works —
are what is left.

---

## PR-3 — step 8, a time axis added to the escape pathway

Registered 2026-09-16 after LK-8 and **before any step 8 rule is committed**, so
each claim below is about a rule that does not exist yet and must be re-read
against the rule when it lands. No confidence numbers, per `METHOD.md`.

The model gains a leak, `tau dh/dt = -h + ReLU(g W h + b + u(t))`, and is run
forward only. Nothing is trained, so there is no BPTT and no stored activation.

**Claims about the machinery**

- Holding the input fixed and running long, the dynamic model reproduces step 4's
  steady-state numbers for DNp01, LC4 and LPLC2 to the tolerance the fixed-point
  solver itself converges at. If it does not, the two are not the same model and
  nothing downstream is interpretable.
- The cost claim holds: a 100-step sequence costs under 2 seconds on OR 1%, and
  the whole of step 8 runs in under an hour on the 8 GB machine.

**Claims about the size arm**

- LPLC2's size tuning survives the time axis as an ordering — response still rises
  monotonically with disc radius — and still does **not** clear its shuffled null
  in level, exactly as in LK-8. The dynamics add nothing to the size arm.
- The Gaussian shape the literature reports is **not** recovered. Step 4 found a
  saturating rise, not a peak with a falling flank, and a leak does not create a
  falling flank.

**Claims about the velocity arm — this is the one worth registering**

- With a single uniform tau, DNp01's response **does** differ between an expanding
  and a contracting disc. This is the trivial direction: a causal leak is not
  time-reversal symmetric and path lengths through the network already differ, so
  some asymmetry is arithmetic, not biology. Step 4's docstring says looming and
  receding are identical *by construction*; that is true of a model with no time
  axis and stops being true the moment one is added, and a step 8 rule that
  registers "no asymmetry" would be scoring the arithmetic.
- With a single uniform tau, DNp01's response is **not** a monotone function of
  angular velocity at fixed final size. A leak is a low-pass filter and velocity
  coding is a differentiation; the wiring would have to supply it as delayed
  inhibition through a longer path, and LK-8 shows LC4's inputs are T2, TmY3 and
  Tm4 rather than anything direction-selective.
- Therefore the velocity arm **requires tau heterogeneity**, and the number that
  comes out of step 8 is how much: the smallest spread across cell types at which
  DNp01 separates fast from slow approach above every shuffled seed. **If that
  sweep finds no value at which it separates, that is the result** and it is
  reported as such.
- The four T4 and four T5 subtypes will **not** need to be told their preferred
  directions by hand. If they do, step 8 has stopped being a wiring result.

**Outcome, 2026-09-16, step 8 at uniform tau** (rules committed at
`malecns-operator@3c6df62` before the run; D6's sweep is still running):

- **D0 same model — correct, and exactly.** Max absolute difference from step 4's
  baseline ladders is **0.00e+00** on all three of DNp01_R, LC4 and LPLC2. The
  leaky model at lambda = 1 is step 4's fixed-point iteration bit for bit.
  Worth recording separately: the *peak* over frames at static r18 is 0.0080
  against the steady state's 0.0061, a 31% transient overshoot. D0 was written
  against the peak first and was changed to read the final frame before the rules
  were committed; unchanged, it would have failed a model that is provably right.
- **D1 asymmetry is wiring — WRONG.** It was registered as expected to fail and it
  **passed**: baseline's loom-minus-recede gap at DNp01_R is 0.00154 against a
  shuffled null of 0.00000–0.00009.
- **D2 velocity tuned — correct.** Spearman of speed with DNp01_R is -0.600,
  short of 0.9, and the raw numbers are stronger than the verdict: loom responses
  are **0.00641, 0.00643, 0.00643, 0.00643, 0.00643** across speeds 20 to 130.
  Approach speed changes the giant fibre's response by 2e-5 over a sixfold range.
- **D3 velocity is wiring — correct.** Baseline's |rho| of 0.600 sits *below* four
  of the five shuffled seeds, which reach 0.707.
- **D4 size arm survives — correct.** Spearman of radius with DNp01_R is 1.000,
  the same as step 4's E2 without a time axis.
- **D5 Gaussian size — correct.** LPLC2 rises monotonically to r18 (0.0057,
  0.0133, 0.0182, 0.0202, 0.0220, 0.0228, 0.0261). No interior peak, no falling
  flank.
- **WIRING CARRIES VELOCITY: False.**

### The reading of D1, which is not what D1 says

D1 passed and its verdict stands. What it measured is not looming selectivity.

Recede's schedule shows the full r18 disc at frame 0 to a network at rest; loom's
walks up to r18 over 20 to 130 frames. So recede drives the largest onset
transient available and loom never does, and the gap is that onset, not a
direction preference: **both loom and recede are flat to five decimals across a
sixfold speed range** (recede 0.00797 at every speed), which a real velocity or
direction signal could not be. The shuffled null failed to catch it because the
shuffled networks barely respond at all — 0.0002 to 0.0011 — so any transient
structure in baseline clears them. A rank test against a null that cannot produce
the artefact does not control for the artefact.

This is the same species of mistake as step 4's E1 and step 6's P5: the lookup
asked whether the *readout level* was reachable and whether the *category* held
what it claimed, and did not ask whether the **stimulus schedules were matched**.
Two schedules that differ in onset as well as in direction cannot separate them.

Per `METHOD.md`, D1 is not re-tuned now that its run has been seen. The sharper
question needs its own rule, committed before its own run: loom and recede matched
on onset — both preceded by the same held frame, or both starting from the same
adapted state — with the speed-flatness above as the thing to beat.

### D6 — NONE IN RANGE, for a reason that is not about tau

No sigma in 0.25 to 2.0, at either tau seed, passed D2 and D3 together. The
verdict is **NONE IN RANGE**, and read plainly it would say that tau heterogeneity
does not buy the velocity arm. It does not say that, because **this sweep never
asked about heterogeneity.**

Stability was held by setting `lam0 = 1 / max(m)`, so raising the spread lowers
the global lambda with it:

| sigma | lam0 | baseline rho | shuffled max | D2 | D3 |
|---|---|---|---|---|---|
| 0.25 | 0.373 | +1.000 | 1.000 | pass | fail |
| 0.5 | 0.139 | +1.000 | 1.000 | pass | fail |
| 1.0 | 0.019 | +1.000 | 1.000 | pass | fail |
| 1.5 | 0.003 | +1.000 | 1.000 | pass | fail |
| 2.0 | 0.0004 | +0.707 / +1.000 | 1.000 | fail / pass | fail |

At sigma = 1.0 the global time constant is already 50 frames against a 200-frame
window, and at sigma = 2.0 it is 2,600. The network stops settling, so a slower
approach simply spends longer at the final radius and peaks higher — which is why
baseline reaches rho = +1.000 and why **the shuffled seeds reach 1.000 too**.
Both are measuring the same settling artefact, and D3 is the reason that did not
get called velocity coding.

Two things follow, both recorded rather than fixed:

- **The sweep confounded spread with global slowing.** Normalising the mean
  lambda rather than the maximum, and lengthening the window until every
  condition settles, is what asks the intended question. That belongs to a new
  rule committed before its own run, not to a re-tuning of D6.
- **D3 has a ceiling.** It demands baseline strictly above every shuffled seed on
  a statistic bounded at 1.000, and once both saturate it cannot pass however
  large the real effect is. A rank test on a bounded statistic needs the
  statistic not to saturate; this one did.

What survives from D6 as evidence is narrow and worth keeping: **at every tau
spread tried, the shuffled networks matched the baseline's speed correlation
exactly.** Nothing in this sweep separated wiring from its own degree-preserving
null.

---

## PR-4 — step 3's remaining verdicts, registered before the run

PR-2 registered S0 and S1 while step 2b was still running and left S2, S3 and S4
unregistered. They are registered here before `step3_rescore.py` is run, against
rules committed at `malecns-operator@39f32b4`. No confidence numbers, per
`METHOD.md`.

What is already known and is not being predicted: step 2b's baseline run moved
the cross loss 4.5% and bought +0.037 held-out balanced accuracy on the taste
task, and its shuffled run bought nothing (-0.018, interval spanning zero). The
149 Shiu outcomes were never in either training set.

- **S2 it moved it the right way** — fails. Thirty-four gains tuned on a four-way
  taste readout have no route to improving agreement with 149 silencing and
  activation outcomes across the whole fly, and step 2b's parameters barely left
  their start. If S1 finds any movement at all it is as likely to be downward.
- **S3 the wiring earned it** — the comparison is empty rather than negative.
  The trained and shuffled-trained gains are both near their shared starting
  point, so whichever scores higher will do so by an amount too small to mean
  anything, and S3 will be decided by noise in either direction. Registering this
  as pass or fail would be registering a coin toss; what is being claimed is that
  **the gap will be smaller than the 4-of-149 gap S0 tolerates as transcription
  noise.**
- **S4 compaction still holds up** — passes. The untrained run already found OR 1%
  changing 4 of 149 answers against random pruning's 10.6 +/- 2.8, and parameters
  that moved this little cannot undo a structural margin that wide.

**Outcome, 2026-09-16. Two of the three are wrong, and the third is wrong
about its size.**

| verdict | registered | result |
|---|---|---|
| S2 it moved it the right way | fails | **passes** — trained 0.715 against untrained 0.669 |
| S3 the wiring earned it | empty, gap under 4/149 | **passes, and the gap is not small** — trained 0.715 against shuffled-trained 0.659 |
| S4 compaction still holds up | passes | **passes** — OR 1% changes 5 answers, the ten random seeds change 9 to 19 |

**TRAINING IMPROVED BIOLOGICAL SIMILARITY: True.**

S2 was called a fail on the reasoning that "thirty-four gains tuned on a four-way
taste readout have no route to improving agreement with 149 silencing and
activation outcomes across the whole fly". There was a route.

S3 was the worse call. It was written to avoid registering a coin toss, and the
caution was aimed at the right hazard — D3's ceiling, seen the same day — but it
was applied to a comparison that was not near any ceiling. The gap is **0.056 in
balanced accuracy, twice the 4-of-149 tolerance it was claimed to fall under**,
and it points the way that matters: gains learned on scrambled wiring score
**0.659, below the untrained 0.669**, so training on a shuffled network made
agreement with the fly *worse*. The real network's gains improved it. Declining
to predict a direction is not free; here it substituted for looking at what the
shuffled control had already shown in step 2b, where the same scrambled network
had gained nothing.

**Correction to this reading, made the same day from the saved answers.** The
paragraph that stood here said this run establishes that taste-fitted parameters
improve agreement with the real fly and only do so on the real wiring. The
verdicts say that; the effect size does not support the word *establishes*.

Exact McNemar on the 149 paired outcomes: untrained to trained changes **6
answers**, 5 right and 1 wrong, **p = 0.219**. Shuffled-trained to trained changes
7, p = 0.125. Untrained to shuffled-trained changes **one answer**, so the
shuffled control shows those gains doing almost nothing here rather than actively
hurting. The +0.046 balanced accuracy is 27 "yes" answers becoming 33.

The direction is consistent across accuracy, balanced accuracy and both controls,
and every margin is inside what six coin flips could produce. The honest claim is
**suggestive, pending a larger or better-resolved test set**.

This also indicts S1's rule, not just its prediction. "Balanced accuracy differs
by more than 0.01" was written without a lookup on what 149 paired outcomes can
resolve, and 0.01 is far below the noise floor. S1 was registered as a fail and
came out a pass, and the pass was cheaper than it looked. The prediction was still
wrong for the reason recorded above — the parameters did move — but the rule it
was wrong about could not have detected the difference either way.

S0 at 149 of 149 is unaffected and remains the solid part: the scoring is the
same scoring Result 1 used.

---

## LK-9 — lookup for step 9, delayed match with an outside memory

Run 2026-09-17 before step 9's rules were written, from `scripts/lookups/lk9_lookup.py`; no arm
was trained and no delay-task accuracy was read. What it found, and what each
finding changed in the rules committed at `malecns-operator@f9b2cd4`:

- **Neuron overlap is a shortcut.** Two samples of one class drawn from one pool
  share 34-46% of their active neurons (Jaccard); held-out water has 5 neurons
  and 10 possible subsets, so 9.6% of same-class pairs are the identical subset.
  Different classes share none. A and B now come from disjoint pools in every set.
- **Resolution.** A lower bound above 0.5 needs 93 trials at accuracy 0.6; a
  paired gap of 0.05 needs 290-934 at 80% power for discordant shares 0.1-0.3.
  Test B is 1000 trials.
- **Cost, 8 GB M1, batch 160.** Forward 36 s (62 iterations), forward and
  backward 52 s, warm start from the previous frame 19 s. The open loop caches
  features once per sample; the closed loop would cost about 4 min per training
  step and is not run in the pilot.
- **The shuffled null is weak.** At g 0.9, 97.5% of DNs respond against 31.4%
  for the real wiring, converging in 10 iterations against 62. Lowering gain to
  0.2 brings the share to 0.43 but shrinks the median response below the real
  wiring's, so no gain in 0.2-0.9 matches both. V3 is flagged as easy.
- **Injection sets for a closed loop**, not used yet: the 121 taste GRNs move 87
  DNs past 1e-4 at amplitude 0.1; their 183 strongest downstream partners move 329.
- **V0 tolerance.** `fpmodel` at step 2b's start and the plain iteration differ by
  6.3e-5, because sigmoid(10) is not 1. The pilot's V0 checks the encoder's
  responsive share against LK-9's 0.314 instead.

## PR-5 — step 9 taste pilot, registered before the run

Against rules committed at `malecns-operator@f9b2cd4`. The smoke run printed
V0's share (0.3135) and training losses on 24 trials, no scores.

- **V0 encoder as measured** — passes; the smoke run already showed it.
- **V1 learnable** — passes. Train and test A draw from the same neuron halves,
  and 347 DN features carry class identity on seen neurons (step 2a test A 1.000).
- **V2 generalises** — passes. Step 2a read single tastes from held-out neurons at
  0.712 four-way; comparing a remembered class to one read that well should clear
  0.5 with 1000 trials, though well below test A.
- **V3 wiring earns it** — passes, for the weak reason LK-9 gives: step 2a's
  shuffled encoder read held-out tastes at chance.
- **V4 beats memory alone** — passes. MEM-ONLY's test B inputs are neurons whose
  input weights never received a gradient, so its B frame is noise to it and it
  should sit near 0.5.
- **V5 holds over delay** — fails. Nothing in training penalises drift past four
  blank frames, and a 16-unit GRU fed zeros for eight steps will move toward its
  own resting state rather than hold A.
- **WIRING HELPS MEMORY** — passes.

**Outcome, 2026-09-17. Every verdict passes; V5 was called wrong, and V3 and V4
pass for a reason neither rule names.**

| verdict | registered | result |
|---|---|---|
| V0 encoder as measured | passes | **passes** — 0.3135 |
| V1 learnable | passes | **passes** — REAL test A at D 4: 0.909, lower bound 0.888 |
| V2 generalises | passes | **passes** — REAL test B at D 4: 0.816, lower bound 0.797 |
| V3 wiring earns it | passes | **passes** — SHUF test B 0.497 |
| V4 beats memory alone | passes | **passes** — MEM-ONLY test B 0.505, paired lower bound +0.284 |
| V5 holds over delay | fails | **passes — wrong.** REAL test B at D 8: 0.807, lower bound 0.788 |
| WIRING HELPS MEMORY | passes | **passes** |

V5 was wrong outright: the 16-unit GRU lost 0.009 between D 4 and D 8 on held-out
neurons and 0.007 on seen ones. Nothing penalised drift past four blank frames,
and it did not drift.

**V3 and V4 are the step 8 D1 mistake again.** SHUF and MEM-ONLY score at chance
on **test A as well** (0.499 and 0.480 at D 4), the set drawn from the neurons
they were trained on, while their training loss went to 7e-5. They memorised 800
trials instead of learning a class comparison, so both arms were broken
in-distribution. V4's rule says the connectome wins because MEM-ONLY's test B
inputs were never active; the run shows MEM-ONLY failed before held-out neurons
came into it. V3's rule compares wiring; the run shows a memory that overfits
1,309 near-uniform shuffled features. Step 2a's shuffled encoder read single
tastes at test A 1.000, so the wiring was not what failed.

What stands from the pilot: a 16-unit memory on the real DN readout holds a taste
across eight blank frames and compares it with one arriving on neurons it never
saw (0.807). What does not: any claim that the wiring beats the memory alone or
shuffled wiring. The verdicts are not re-run; the olfactory step gets a design
change instead, registered before its own run — every arm must first be
learnable in-distribution (a test A gate), or its comparison is void.

---

## LK-10 — lookup for step 10: which DoOR entries can be used at all

Run 2026-09-17 from `scripts/lookups/lk10_mapping.py` and `lk10_lookup.py` on DoOR.data v2.0.1; table arithmetic only.

- **Mapping.** 47 of MaleCNS's 53 ORN types reach a DoOR unit with data. Unmapped:
  DA1 (Or67d mapped, but the normalized matrix holds nothing for it), VA7m, VM6l/m/v.
  DL2d and DL2v share one recording (ac3A). MaleCNS VC3 and VC5 are hemibrain
  VC3l and VC3m; DoOR's VC5 (Ir41a) matches the current name.
- **Complete rectangle.** 23 units x 112 odorants with no missing entry (1,282
  ORNs), which is Hallem & Carlson 2006's panel less Or33b (maps to DM5+DM3).
  Dropping to 15 units gains one odorant.
- **Imputation, masked 10% x 30 on 201 odorants.** RMSE: zero 0.202, unit mean
  0.139, rank 2 0.132, rank 4 0.148, rank 8 0.203, against an observed SD of 0.155.
- **Missingness is selective.** Within Hallem 2006, odorants other studies also
  measured respond more strongly (+0.26 z, permutation p = 0.0002); Marshall 2010
  the same (+0.47, p = 0.0002). The masked RMSEs above are therefore optimistic.

Decision: step 10 uses Hallem 2006's raw panel alone (23 x 110, spikes/s), no
imputation.

## LK-11 — lookup for step 10: scale, readout, null, cost

Run 2026-09-17 after step 9's pilot finished. No arm trained, no odor identity read.

- **Splits.** 33 of 110 odorants held out, stratified over 10 chemical classes;
  smallest ORN pool after the 30% / half / half split is 5 (VA5).
- **The DN readout does not carry odor.** The rule as drafted — the weakest SCALE
  at which the median odor moves as many DNs as the median taste sample (171) —
  has no solution: SCALE 100, with inputs 3.5x taste's peak, moves 38; SCALE 300
  moves 12. The rule stopped the lookup as written.
- **Revised before any accuracy existed.** SCALE is set on input only: the
  odor panel's median L2 norm matched to taste's 3.43, giving **SCALE = 163**.
  Background amplitude **BG = 0.5** by the drafted rule (median norm 2.38, below
  3.43 — the rule passes it, though background is 70% of the odor's norm).
- **Readouts at SCALE 163, 160 samples.**

| readout | size | real: n >= 1e-3 / n >= 1e-2 / median | shuffled 1000, 1001: n >= 1e-3 / median |
|---|---|---|---|
| DN | 1,314 | 9 / 0 / 7.5e-6 | 754, 728 / 1.2e-3 |
| MBON | 97 | 74 / 0 / 2.7e-3 | 66, 61 / 1.4e-3, 1.2e-3 |
| LH output (LHAV/LHAD/LHPV/LHPD) | 1,929 | 1,648 / 493 / 4.8e-3 | 894, 877 / 8.6e-4, 8.2e-4 |

  LH output clears the 50-neuron floor, and unlike DN its shuffled null is not
  the near-uniform saturated state: shuffled LH responses are about 5.7x
  smaller, not larger. MBON barely separates from shuffled.
- **V0 as drafted is useless on LH.** The share past 1e-4 is 0.9979 for the real
  wiring and for both shuffled seeds (1,925 of 1,929 in every case). V0 needs a
  magnitude, not a share.
- **Cost.** Encoder 36 s per 160 samples real, 6 s shuffled; 6,400 samples per
  network gives 24 min real and 4 min per shuffled seed, 45 min with five.

## PR-6 — step 10, delayed match on odors, registered before the run

Against rules committed at `malecns-operator@2cee1bd`. The smoke run printed V0's
value (4.8183e-3, LK-11's exactly) and losses on 24 trials, no scores. Step 9's
pilot numbers are exploratory and are not the reason for any call below.

- **V0 encoder as measured** — passes; the smoke run showed it.
- **G for REAL, and V1** — passes. Train and test A share odors and neurons, and
  1,648 LH output neurons move past 1e-3.
- **G for every SHUF seed** — passes. Shuffled LH output still moves 877-894
  neurons past 1e-3, and a fixed projection with weight decay leaves a 16-unit
  memory less room to memorise 800 trials than step 9 did.
- **G for MEM-ONLY** — passes, for the same reason: on test A its B frame lands on
  the h2 neurons it was trained on.
- **V2 generalises** — passes, but well below the taste pilot. With 77 training
  odors the memory cannot learn odor identities; it has to learn a similarity
  comparison, and roughly five same-odor trials per odor is thin for that.
- **V3 wiring earns it** — **fails**: at least one shuffled seed matches or beats
  REAL on test B. A random recurrent network preserves distances between inputs,
  and shuffling keeps every neuron's degree, so each LH neuron still pools many
  ORNs. Unseen neurons of one type land on overlapping targets in both.
- **V4 beats memory alone** — passes. MEM-ONLY's test B frame B falls on ORNs
  whose projected directions never co-occurred with the same odor in training.
- **V5 holds over delay** — passes. Nothing in this design differs from the
  pilot's memory on that point except the projection.
- **V6 new odors alone** — passes, above V2's accuracy.
- **WIRING HELPS MEMORY** — fails, through V3.
- Reading, not a verdict: **POOLED scores above REAL** on test B.

**Outcome, 2026-09-17. REAL works on new odors and new neurons, and the two
comparisons the question was about are VOID: every null arm failed the gate.**

| verdict | registered | result |
|---|---|---|
| V0 encoder as measured | passes | **passes** — 4.8183e-3 |
| G REAL, V1 | passes | **passes** — test A 0.775, lower bound 0.744 |
| G every SHUF seed | passes | **wrong.** All five fail: test A 0.486-0.523, lower bounds 0.449-0.488 |
| G MEM-ONLY | passes | **wrong.** Fails: test A at D 4 0.513, lower bound 0.480 |
| V2 generalises | passes, well below the pilot | **passes** — test B 0.724, lower bound 0.705 (pilot 0.816) |
| V3 wiring earns it | fails | **VOID** — no shuffled seed is a valid arm |
| V4 beats memory alone | passes | **VOID** — MEM-ONLY is not a valid arm |
| V5 holds over delay | passes | **passes** — D 8 0.713, lower bound 0.693 |
| V6 new odors alone | passes, above V2 | **passes; "above V2" wrong** — test C 0.724 against test B 0.724 |
| WIRING HELPS MEMORY | fails | **VOID** |
| reading: POOLED above REAL | yes | **yes** — test B 0.765, test A 0.840 |

The step 9 fix did not fix it. A fixed 64-dimensional projection and weight
decay left every null arm memorising: training loss 2e-5 to 1.5e-4 with test A at
chance, exactly the pilot's failure. The gate did its job — V3 and V4 are VOID
instead of the pilot's cheap passes — but the question they carried is
unanswered.

Two predictions about validity were wrong for the same reason: the call assumed
a projection and weight decay limit memorisation, and was made without any
lookup of whether they do. What stands: a 16-unit memory reading LH output of
the real wiring tells same from different odors on odors and receptor neurons it
never saw (0.724), holds it over eight blank frames (0.713), and new neurons cost
it nothing beyond new odors (test C 0.724). The oracle that is handed the 23
receptor types does better (0.765). Whether the wiring, rather than any
convergent encoder, is what makes this possible is not decided by this run.

**Exploratory, after the verdicts, not evidence for any of them.** Step 10's
exact trials rebuilt and encoded again, with no memory: the cosine similarity
between frame A and frame B LH features, same-odor trials against different-odor
trials, as an AUC. Real wiring: train 0.956, test A 0.962, **test B 0.947** (raw;
0.924-0.945 centred on the train mean). Shuffled seed 1000: 0.489-0.514 raw,
0.527-0.602 centred; any two of its samples sit near orthogonal (median cosine
0.02 whether same odor or not). Two readings. The shuffled arms' failure is mostly
an absence of same-odor information in their features, not only memorisation.
And the trained memory is the bottleneck on the real wiring as well: a fixed
similarity with no training separates new odors on new neurons far better (0.947
AUC) than the 16-unit memory did (0.724 accuracy). A confirmatory test needs its
own rules; one shuffled seed and no calibration slice do not make this one.

*Transcription correction, 2026-09-17:* the outcome table first gave MEM-ONLY's
test A as 0.490, copied from the wrong line of the log. The run's JSON has 0.513;
the gate used the lower bound, 0.480, and fails either way.

---

## LK-12 — lookup for step 11: the two side fields, the balance, and the offset

Run before any rule was written, with `step11_side.py --lookup` and two scratch
probes. The probe numbers are quoted as measurements of feasibility and are not
re-used as results; the run repeats everything on held-out odors and neurons.

- **The two ends need different side fields.** `somaSide` is null for all 804
  panel ORNs; `rootSide` is null for all 1,929 LH output neurons. Step 7 met the
  first half of this and recorded its Q4 as not evaluable. Step 11 takes ORN side
  from `rootSide` and readout side from `somaSide`, and asserts both at load.
- **The panel is laterally unbalanced**: 410 L, 611 R, 261 neither — those 261
  are `rootSide == "unknown"`, not null, and are 261 of the connectome's 414
  `unknown` values. An unbalanced panel tilts a left-minus-right readout before
  any odor arrives, so each receptor type is cut to `min(nL, nR)` per side:
  **1,282 -> 804 ORNs, 402 per side.** The cut is not even across types. Or47b
  loses 130 -> 30, Or88a 132 -> 46, Or65a 103 -> 34, and **Or49b falls to 3 per
  side**, which is small enough that its contribution is close to noise.
- **The readout leans one way regardless of the odor**, and the lean is the size
  of the signal. Over 400 calibration trials the statistic `d` sat at -5.3e-2 to
  -6.7e-2 at every contrast, including the shallowest, against its own spread of
  2.9e-2 to 7.4e-2. A raw sign test reads that lean as evidence: in the probe,
  raw accuracy 0.885 at 1.0/0.0 against 0.955 once centered.
- **The offset depends on which ORNs carry the odor**, -5.3e-2 to -5.8e-2 on the
  train set against -5.8e-2 to -6.7e-2 on the held-out set, about 10% apart. So
  it is estimated per arm, per contrast AND per ORN set, on calibration trials
  sharing no odor with any test set.
- **A paired statistic is not usable.** Presenting the same odor with the
  gradient both ways scored 1.000 for REAL at every contrast, because resetting
  the noise draws leaves the L/R swap as the only difference between the two
  presentations. That measures sensitivity, not a task, and no animal gets the
  same stimulus twice. Recorded, not used.
- **Scale and feasibility.** On the full 402-per-side pool, centered accuracy
  fell 0.955, 0.922, 0.840, 0.632 over contrasts 1.0/0.0, 1.0/0.25, 1.0/0.5,
  1.0/0.8, with two shuffled seeds at 0.505-0.560 throughout. **1.0/0.5 is the
  primary**: 1.0/0.0 is a unilateral stimulus rather than a gradient, and 1.0/0.8
  sits near the floor.
- **Shuffled wiring is weak, not dead**: LH median max|x| 7.3e-4 against REAL's
  4.4e-3, the ratio LK-11 found. It responds; it does not lateralise.
- **Cost**: 340 s for 1,600 stimuli on the real wiring, 60 s per shuffled seed,
  668 s for the lookup itself, 2.04 GB peak.

**What this lookup cannot settle.** Whether a low ceiling at shallow gradients is
the wiring or the input code. Most *Drosophila* ORNs project bilaterally to both
antennal lobes, so side information is pooled by construction. The ORACLE arm
bounds the laterality the stimulus itself carries; it does not separate these two.

---

## PR-7 — step 11, one-shot lateralisation, registered before the run

Registered 2026-09-20, against rules committed in `step11_side.py`'s docstring,
before the run. No arm trains. The only fitted quantity in the whole run is one
centering scalar per arm, contrast and ORN set.

| rule | prediction | reason |
|---|---|---|
| H harness works | pass | ORACLE reads the antennae directly; if it fails, the wiring is not why |
| V1 works in-dist | pass | the probe measured 0.840 on this distribution before the held-out split |
| V2 generalises | pass | the statistic sums 1,929 neurons and does not depend on which ORNs carry the odor |
| V3 wiring earns it | pass | two shuffled seeds sat at 0.505-0.560 against 0.840; five clearing that by chance is not credible |
| V4 the sides earn it | pass | a random 967/962 split of LH has no reason to track the stimulated antenna |
| V5 new odors alone | pass | held-out neurons are the harder axis, and V2 already assumes them |
| V6 contrast orders | pass | the probe's four contrasts were already monotone |
| THE WIRING LATERALISES | pass | V2, V3 and V4 are each predicted to pass |

**Registered with a number, not just a direction.** The probe's 0.840 at the
primary contrast used all 402 ORNs per side and centered on the test set's own
mean. The run holds out 30% of the ORNs and centers honestly, so REAL's test B
accuracy is predicted to land **between 0.62 and 0.75**, below the probe. An
outcome above 0.80 would mean the held-out split is not doing what it should and
should be treated as a warning, not a win.

Every rule is predicted to pass, which is worth recording as a weakness: this run
is closer to a measurement than a test, because the probe already saw the effect
at three of four contrasts. What the run adds is the two held-out axes, honest
centering, five null seeds instead of two, and a control the probe did not have.

**The outcome this registration exists to catch: if V4 fails while V3 passes**,
the result is not about the wiring's laterality but about the `somaSide` labels,
and the reading changes completely.

### PR-7 outcome

Run 2026-09-20 against the rules committed at `malecns-operator@1f9c70c`, before
the run. `data/step11_side.json`, `data/log_step11_side.txt`, 5,085 s, 1.88 GB.

**Every rule passed, as predicted, and the registered number was right.** REAL's
test B accuracy at the primary contrast came in at **0.715** (lower bound 0.686),
inside PR-7's registered band of 0.62 to 0.75 and below the probe's 0.840 as
predicted. Nothing in the pipeline was trained.

| set | REAL | ten null seeds | ORACLE |
|---|---|---|---|
| test A, seen odors and neurons | 0.805 (0.765) | 0.415-0.580 | 1.000 |
| test C, new odors | 0.766 (0.739) | 0.408-0.590 | 1.000 |
| test B, new odors AND new neurons | **0.715 (0.686)** | 0.441-0.620 | 1.000 |

REAL test B over contrasts, decaying as the gradient shallows: 0.862 (0.841) at
1.0/0.0, 0.805 (0.780) at 1.0/0.25, 0.715 (0.686) at 1.0/0.5, 0.591 (0.559) at
1.0/0.8. The centering offset moved only from -6.7e-2 to -5.8e-2 across all four,
as LK-12 measured.

**What went right that PR-7 flagged as the thing to watch.** V4 passed: all five
readout-permutation seeds sat at 0.482-0.620 against REAL's 0.715. The effect
tracks the anatomical `somaSide` labels, not any partition of 1,929 neurons into
967 and 962. Had V4 failed while V3 passed, the reading would have been about
label bookkeeping instead of wiring.

**Three limits, none of them repaired after the fact.**

- **The rank test is weak even though the margin is not.** Five shuffled seeds
  and five permutation seeds give a one-sided p of about 1/6 each, the same
  convention step 10 used. The bootstrap margin is wide — 0.686 against a
  highest null of 0.620 — but the formal p from the permutation is 0.167, not a
  conventional threshold. Step 7 bought p ~ 0.048 with 20 rewirings; 20 seeds
  here would cost about 2.8 hours and has NOT been run. The result is reported
  at the strength five seeds buy.
- **ORACLE scored 1.000 everywhere**, so the stimulus carries the answer
  perfectly and the ceiling is 1.0. REAL's 0.715 therefore measures how much side
  information the wiring PRESERVES, not something it computes. Roughly a third of
  what was available is lost between the antennae and lateral horn output.
- **The bilateral-projection confound stands**, exactly as LK-12 said it would.
  Most *Drosophila* ORNs project to both antennal lobes, so the decay from 0.862
  to 0.591 as the gradient shallows may be a fact about the connectome's pooling
  rather than about this model. This run cannot separate them.

**Why this one is different from steps 9 and 10.** No arm trained, so no arm
could fail to train, and no comparison went VOID. The null arms sat at chance
because chance is their answer. It is the first task in this project where the
real wiring beats its nulls with nothing fitted anywhere except one centering
scalar per arm.

---

## LK-13 — lookup for step 12: does olfaction have anything for a time axis

`owner-doubt:` the assistant stated that adding a time axis "was tried and
failed", citing step 8. Asked whether step 8 had covered OLFACTION — it had not;
its script mentions odor zero times. The claim was a carry-over from the visual
escape pathway, and step 12 exists because it was challenged.

Run before any rule was written. Step 8's numbers are NOT reused: the point of
this lookup is that step 8 failed on a pathway that was already silent, and
whether olfaction is in the same state had never been measured.

Two shuffled seeds, uniform tau, triangle ramps at s = 10, 25, 60.

| | REAL | SHUF-1000 | SHUF-1001 |
|---|---|---|---|
| lam = 1 reproduces the fixed point | 1.5e-8 | 3.0e-8 | 3.0e-8 |
| mean LH \|x\| at full concentration | 3.6-4.5e-3 | 4.0-4.5e-4 | 4.2-4.8e-4 |
| HYST at s = 10 / 25 / 60 | +0.166 / +0.105 / +0.056 | -0.024 / +0.019 / +0.018 | -0.026 / +0.017 / +0.017 |
| peak \|x\| | 0.140 | 0.121 | 0.109 |

- **The signal is alive, unlike step 8's.** About 9x above the nulls, consistent
  with LK-11's 5.7x. LK-8 had found DNp01_R's whole range at 0.0002-0.0061 and
  LPLC2 never clearing its own null at any radius; step 8 was unanswerable before
  it began. This is not that case.
- **The rate signal orders**: faster ramp, more hysteresis, on the real wiring
  only. Both shuffled seeds sit flat near zero with no ordering.
- **Peak is blind here, and this is the measurement that mattered most.** 0.140
  against 0.121 and 0.109 is not a separation. The whole effect lives in the
  matched-concentration comparison, so a step 8-style peak statistic would have
  found nothing. Registered as E4, expected to fail, rather than left in a
  footnote.
- **Step 8's D6 confound is reproducible and now fixed.** Setting LAM0 = min(m)
  slows the whole network as the spread grows, which is exactly why step 8 could
  say nothing about tau. Step 12 fixes LAM0 at a constant 0.1353 sized for the
  largest sigma and reuses it at every sigma, so mean log lam is identical and
  spread is the only thing varying. z is clipped to +-2 because 11,751 cell types
  otherwise drive m_min to ~exp(-4), which would leave the network unable to
  follow the fastest ramp.
- **A per-type tau is nearly a per-neuron tau.** 11,751 types over 166,700
  neurons. This is structured noise and not a claim about cell-type time
  constants in the fly. Recorded so no reader takes it for one.

**What a pass will and will not license.** A normalised hysteresis above shuffled
says the real wiring integrates more slowly than its degree-matched shuffles — a
graph-level property, stronger and more coherent recurrence. It does NOT say the
wiring computes a derivative, that anything downstream reads one, or that the fly
uses it. The claim available is that the material for a rate signal survives the
wiring and does not survive shuffling.

---

## PR-8 — step 12, a time axis on olfaction, registered before the run

Registered 2026-09-20, against rules committed in `step12_odor_dynamics.py`'s
docstring, before the run. Nothing is trained and the statistic has zero
parameters — not even step 11's one centering scalar.

| rule | prediction | reason |
|---|---|---|
| E0 same model | pass | LK-13 measured 1.5e-8 against a 1e-4 bar |
| E1 signal above floor | pass | 9x separation in the lookup, on 8 odors; 60 will not reverse it |
| E2 rate is read | pass | HYST +0.166 against -0.024 and -0.026 at the fastest ramp |
| E3 scales with rate | pass | the lookup was already monotone, 0.166 / 0.105 / 0.056 |
| **E4 peak is not it** | **FAIL** | 0.140 against 0.121 and 0.109. A pass would mean the lookup's peak reading was wrong and E2's margin needs re-reading, not that the result is stronger |
| E5 holds over tau | **uncertain** | never measured. LAM0 is held constant now, so sigma changes only the spread, and whether a near-per-neuron spread preserves the effect is genuinely unknown |
| THE WIRING CARRIES RATE | pass | E1, E2 and E3 each predicted to pass |

**The one rule whose answer I do not know is E5**, and it is the only one worth
running for its own sake. E1 to E3 are confirmations of a lookup on more odors,
five seeds instead of two, and rules fixed in advance. E4 is a registered
negative whose job is to keep the peak statistic's blindness on the record.

RATEACC was not measured in the lookup at all — only HYST was — so its level is
unregistered. Predicted above 0.5 and above every shuffled seed, with no number
claimed.

**What would change the reading.** If E3 fails while E2 passes, the effect is a
fixed lag rather than a rate signal, and the phrase "carries rate" must not be
used. If E5 fails at sigma 1.0, the effect depends on near-uniform time constants
and that dependence belongs in any sentence written about it.

---

## PR-9 — step 12 re-registered after smoke found PR-8's rules unrunnable

Registered 2026-09-20, before the run. **PR-8 is not edited and is not
withdrawn.** Its E2 stands as a registered mistake caught by a smoke test rather
than by a reviewer, which is the whole point of registering.

**What smoke found in PR-8's rules, before any result existed.**

1. **E2 was written on a saturated statistic — step 8's D3 defect, reproduced.**
   RATEACC scored exactly 1.000 for the real wiring AND for a shuffled seed at
   every ramp speed. At matched concentration the lagging limb is higher
   essentially always, so the fraction carries nothing once any lag exists.
   PR-8's E2 "failed" only because 1.000 > 1.000 is false. This project already
   recorded step 8's D3 as "a strict win on a statistic both sides had saturated
   at 1.000", and the same defect was written into PR-8 hours after that line was
   read. RATEACC is now a reading, printed so the saturation stays visible, and
   every verdict is on HYST.
2. **The sigma fix destroyed the discrimination it was meant to make askable.**
   The imposed per-neuron leak is identical in every arm, so a large leak adds
   lag the wiring did not produce and swamps the lag it did.

   | lambda | REAL HYST | SHUF HYST | ratio |
   |---|---|---|---|
   | 1.0 | +0.166 | +0.018 | 9x |
   | 0.1353 | +0.855 | +0.723 | 1.18x |

   Primary verdicts now run at lam = 1, where the only lag is the network's own
   recurrence. **The tension is structural and is stated in the rules rather than
   hidden:** stability needs max lam = lam0 / min(m) <= 1, so sigma > 0 FORCES
   lam0 < 1. The tau question can only be asked where discrimination is already
   compressed, so E5 is declared LOW POWER by construction and no strong reading
   may be taken from it either way.
3. **E1 measured the timescale, not the level.** At a fast ramp the network has
   not settled, so a ramp frame cannot report a response level. E1 now measures
   at the fixed point with the odor held.

**A number that moved, recorded rather than quietly adopted.** LK-13's 9x used a
blank baseline taken at the final frame. The committed rule subtracts the blank
column at the MATCHING frame, step 8's convention, and under it the six-odor
smoke gives 0.468 against 0.197, about 2.4x. Same direction, same ordering,
smaller ratio. The verdicts are rank tests so the ratio does not enter them, but
LK-13's 9x should not be quoted for the committed statistic.

| rule | prediction | reason |
|---|---|---|
| E0 same model | pass | 1.49e-8 against a 1e-4 bar, unchanged |
| E1 signal above floor | pass | 5.55e-3 against 6.62e-4 at the fixed point, 8.4x on six odors |
| E2 rate is read | pass | 0.468 against 0.197 at the fastest ramp |
| E3 scales with rate | pass | 0.468 / 0.231 / 0.107, monotone on six odors |
| **E4 peak is not it** | **FAIL** | unchanged from PR-8. Its job is to keep the peak statistic's blindness on the record |
| E5 holds over tau | **uncertain, low power** | never measured, and now known to be asked in a compressed regime |
| THE WIRING CARRIES RATE | pass | E1, E2 and E3 each predicted to pass |

**Still the only rule whose answer is unknown is E5**, and it is now known to be
a weak test rather than merely an untried one. E1 to E3 are confirmations on 60
odors and five seeds of what six odors and one seed already showed.

**What would change the reading**, unchanged from PR-8: if E3 fails while E2
passes, the effect is a fixed lag and the phrase "carries rate" must not be used.
Added here: the per-odor HYST spread at the fastest ramp was sd 0.0605 around a
mean of 0.468 on six odors. **If that spread stays small on 60, the honest
description is one global lag, not an odor-specific rate signal**, and the
reading must say so even though every verdict passed.

### PR-9 outcome

Run 2026-09-20 against the rules committed at `malecns-operator@a7f4743`, before
the run. `data/step12_odor_dynamics.json`, `data/log_step12_odor_dynamics.txt`,
2,821 s, 1.97 GB. 60 odors, five shuffled seeds.

**Every rule came out as predicted, including E4, which was registered to fail.**

| rule | predicted | outcome |
|---|---|---|
| E0 same model | pass | PASS, 2.98e-8 against a 1e-4 bar |
| E1 signal above floor | pass | PASS, 4.59e-3 against 5.47-5.89e-4, about 8x |
| E2 rate is read | pass | PASS, HYST 0.460 against 0.191-0.197 |
| E3 scales with rate | pass | PASS, 0.460 / 0.224 / 0.101, monotone |
| E4 peak is not it | **FAIL** | **FAILED**, as registered. Peak 0.1617 is not above the seeds |
| E5 holds over tau | uncertain, low power | **PASS at every sigma** |
| THE WIRING CARRIES RATE | pass | PASS |

**E5, the only rule whose answer was unknown, passed and did not weaken.** At the
constant lam0 = 0.1353, REAL scored 0.852, 0.839, 0.799 at sigma 0, 0.5, 1.0
against shuffled maxima of 0.723, 0.706, 0.657. The gap is 0.129, 0.133, 0.142 —
it widens slightly as the spread grows. So the effect does not depend on
near-uniform time constants. This is the first thing step 8's D6 was meant to ask
and could not, and holding lam0 constant is what made it askable.

**THE REGISTERED CAVEAT TRIGGERED, AND IT GOVERNS THE WORDING.** PR-9 said: if
the per-odor HYST spread stays small on 60 odors, the honest description is one
global lag, not an odor-specific rate signal, **even though every verdict
passed**. It stayed small — sd 0.0519 around a mean of 0.4599, about 11%, against
a real-to-shuffled gap of 0.268. The odor-to-odor variation is five times smaller
than the wiring effect.

So the result is: **the real wiring integrates more slowly than its
degree-matched shuffles, by roughly 2.4x, and it does so almost uniformly across
odors.** It is a property of the graph, not an odor-specific rate code. The
phrase "carries rate" is true in the narrow sense the rules defined and would be
misleading in any wider one. What a downstream neuron could read from this is
"something is increasing", not "this odor is increasing".

**The rank test badly understates the separation, and both numbers belong in any
quotation.** Five shuffled seeds give one-sided p ~ 1/6 by the convention steps
10 and 11 used. But the null is nearly deterministic: the five seeds spanned
0.1910 to 0.1967, a spread of 0.0057, against a gap to REAL of 0.268 — about
47 null spreads. The honest statement is a large, tightly-bounded separation
reported at the weak significance five seeds buy. More seeds are cheap and were
not run, the same parked item step 11 carries.

**Readings.** RATEACC was 1.0000 in every arm at every speed, saturated exactly
as PR-9 said it would be, which is why no verdict used it. The 33 odors from step
10's held-out set scored 0.4505 against 0.4599 for all 60 — no difference, as
expected where nothing is fitted.

---

## PR-10 — steps 11 and 12 re-run at 20 null seeds, registered before the run

Registered 2026-09-20, against rules committed in `step11_side.py` and
`step12_odor_dynamics.py`, before the run.

**Nothing about either question changes.** The verdicts, statistics, splits,
stimuli and arms are identical; `--seeds=N` moves only the strength of the rank
test, from one-sided p ~ 1/6 at five seeds to ~1/21 at twenty, which is the
convention step 7 used. Results are written to separate files so the five-seed
runs stand unaltered beside them.

**Why this is worth three hours of a laptop.** Both results carry a wide
separation reported at weak significance, and in both the null turned out to be
nearly deterministic:

| | REAL | five null seeds | null spread | gap |
|---|---|---|---|---|
| step 11, test B | 0.715 | 0.441-0.620 (ten arms) | — | ~0.10 to nearest |
| step 12, HYST s10 | 0.460 | 0.1910-0.1967 | 0.0057 | 0.268 |

Step 12's gap is about 47 null spreads wide and still reports p ~ 1/6. That is
the rank test's floor at five seeds, not a fact about the data.

| prediction | reason |
|---|---|
| step 11 V3 and V4 hold at 20 seeds | the ten existing null arms span 0.441-0.620 against 0.715; fifteen more clearing 0.715 by chance is not credible |
| step 12 E2 and E5 hold at 20 seeds | the five existing seeds span 0.0057 against a gap of 0.268 |
| **at least one new null seed will land above the current five-seed maximum** | five draws do not bound a distribution. Step 8's LK-8 found a one-seed "worst" was not a property of the control at all, and the skill's own validation section says a random control must be drawn over several seeds and never one |
| no verdict flips | the margins are far outside the observed null spread in both steps |

**What would be informative rather than merely confirmatory.** If a new seed
lands near or above REAL in either step, the five-seed result was luck and both
must be re-read. That is the outcome this run exists to expose, and it is why the
third prediction above is registered explicitly rather than assumed away.

**Not claimed.** p ~ 0.048 is still not strong evidence, and twenty seeds do not
make step 12's global lag odor-specific, do not raise step 11's ceiling above the
input's own laterality, and do not separate the bilateral-projection confound.
Those limits are unchanged.

### PR-10 outcome — STOPPED BEFORE COMPLETION, by the owner's decision

`owner-doubt:` not a doubt about a premise but about a priority — the assistant
had ordered confirmatory seed runs ahead of the work the owner actually asked
for, and the owner said so. Recorded here because the ordering was the
assistant's recommendation and the correction came from outside it.

Started 2026-09-20 and stopped the same day, part way, on the owner's call that
confirmatory work was the wrong use of the machine. **No verdict from PR-10 is
claimed and no JSON was written.** It is recorded here rather than dropped,
because a registered prediction that disappears is worse than one that fails.

**What completed: the readout-permutation arm, all 20 seeds.** Test B at the
primary contrast, sorted:

```
0.368 0.473 0.473 0.475 0.482 0.485 0.486 0.490 0.495 0.497
0.502 0.508 0.513 0.519 0.536 0.539 0.540 0.551 0.552 0.620
```

Median 0.500, which is exactly what a random 967/962 relabelling of the readout
should give. **PR-10's third prediction was wrong, in the harmless direction:**
no new seed exceeded the five-seed maximum of 0.620, so the original five had
already found the tail. REAL's 0.715 stands 0.095 above the most extreme of
twenty draws and about 0.215 above the median. V4 is therefore better supported
than it was, though at a rank strength that was never computed.

**What did not complete:** the shuffled arm reached 3 of 20 seeds (1000, 1001,
1002 — unchanged from the five-seed run, as they must be, same seeds and same
stimuli). Step 12's 20-seed run never started. **V3, E2 and E5 remain at the
five-seed strength, one-sided p ~ 1/6**, exactly as steps 11 and 12 report them.

**The parked item is therefore still parked**, now with one third of it done and
the permutation half answered. Unparking cost is about 2.5 hours for step 11's
remaining 17 shuffled seeds and 2.5 hours for step 12.

**Why it was stopped, recorded as method rather than as apology.** The runs would
have moved p from ~1/6 to ~1/21 and changed no reading, no limit and no next
step. Nothing downstream depended on them. The machine went instead to a question
whose answer was unknown. That is the right trade and it is worth having on the
record as a precedent: a confirmatory run with no dependent work is not
automatically worth its compute.

---

## LK-15 — lookup for step 13: what already exists, and whether it is untrained

`owner-doubt:` two challenges, both before any code was written. First, whether
the dynamics work had prior art — it has, extensively, including work this repo
already cites. Second, whether the existing connectome games are really
untrained as their READMEs imply. They are not trained, and that turned out to
be the wrong thing to have been checking.

**Prior art: there is no methodological novelty here, and the record should say
so plainly.**

| what we do | prior art |
|---|---|
| steady-state / fixed-point response on a connectome | standard — the connectome as a normalised weighted adjacency matrix, response defined as the steady state of a discrete-time linear system |
| leaky dynamics on a connectome (step 12) | Shiu et al. 2024, **which this repo already cites and scores against** in `score_shiu.py`. Also Loihi 2, Brian2 and NEST ports |
| implicit gradients at a fixed point (step 1, `fpmodel.py`) | Deep Equilibrium Models, Bai/Kolter/Koltun 2019. The repo's trainer is a DEQ on a connectome |
| wiring fixed, only gains trained (steps 2–3) | `train-your-fly/connectome`; Lappalainen et al., Nat Neuro 2025 |
| closed-loop connectome agent (step 13) | Vaxenburg et al.; NeuroFly; whole-brain graph locomotion control — **all of which TRAIN a controller** |
| bilateral antennal comparison for odor steering | established biology; PNAS 2026 finds gradient cues informative in SMOOTH plumes, odor-motion cues in complex ones |

The last row changes a rule: step 13 models the **smooth-gradient regime only**,
which is where the bilateral mechanism is the documented one. Walking flies in
complex plumes use encounter timing, not spatial gradients, and step 13 says
nothing about that case.

**The games: about 40 exist, and odor navigation is among them.** Two awesome-
lists index them. MaleCNS has been wired to Doom, Mario 64, Minecraft, Flappy
Bird, CARLA, Pong, chess, blackjack and FNAF. Odor-guided navigation
specifically exists at least four times, including a 139k-neuron browser fly
following fermenting apples.

**What the code audit found, which is not what the question asked.** None of the
three examined trains weights; all use connectome weights with Shiu's LIF. The
contamination is hand-inserted structure, and it sits exactly where this project
took its measurements:

- **flyverse-core** states that its plume instrument "provides bilateral odor
  concentration differences as artificial steering input, **but this bypasses
  native circuit computation**." Its three path gains are "mechanistic but
  magnitude-uncalibrated", one applied to 101,619 edges it calls mislabeled.
- **fly-with-a-real-brain** — the most transparent of them — records that "the
  laterality is the model's and **the sign convention is ours**", chosen because
  the connectome's laterality was ambiguous, and that "a foraging drive is added.
  Nothing in the connectome makes a fly walk about on its own."
- **flybrain** ships a "documented motor adapter" and "weight calibration", and
  labels courtship and egg-laying as modeled programs.
- **None of the three has a shuffled, randomised or otherwise controlled
  network.** Their documentation is honest; a viewer of the demo cannot see any
  of it.

**An independent corroboration of LK-11, worth recording.** flyverse reports
that DNa02 is "held below a 7.0 mV gap" with lateral excitatory inputs
near-silent, **preventing directed odor-guided turning**. LK-11 found the same
wall from the other side: at SCALE 163 only 9 DNs pass 1e-3 on the real wiring
against 754 and 728 on shuffled seeds, which is why step 11 reads lateral horn
output instead. Two projects that do not know of each other hit the same
obstacle; one bypassed the circuit, the other moved the readout and wrote down
that this is not navigation.

**What this leaves step 13 as.** Not a novel method and not a new genre. The
distinguishing claims are narrow and checkable: the steering sign is **measured**
(0.715 on held-out odors and neurons) rather than chosen; the odor never
bypasses the circuit; and the shuffled arms run beside the real one in the same
view.

**A choice step 13 does make, declared rather than buried.** "Turn toward the
more stimulated side" is an assumption about attraction versus aversion. It is a
behavioural assumption, not a neural sign, it is identical in every arm, and it
is named here so it is not mistaken for something the wiring supplied.

---

## LK-16 — lookup for step 13: geometry, the start heading, and a floor that walks away

`owner-doubt:` two challenges to the design, both before the run. First, that a
shorter start distance would make the result more visible — it does the
opposite. Second, that episodes starting already pointed at the source should be
excluded — correct, though the effect is smaller than either of us expected, and
the measurement is recorded rather than the intuition.

**Start distance: closer is WORSE, and the assistant had it backwards.** The
suggestion was r0 = 2 mm because arrival rises to 0.84. But the chance floor
rises with it, to 0.295. Measured in the geometry model at p = 2:

| r0 | real wiring | chance floor | gap | ratio |
|---|---|---|---|---|
| 2 mm | 0.841 | 0.290 | 0.551 | 2.90x |
| 3 mm | 0.789 | 0.240 | **0.549** | 3.29x |
| 5 mm | 0.682 | 0.164 | 0.518 | 4.17x |
| 10 mm | 0.442 | 0.077 | 0.364 | 5.70x |

The absolute gap peaks near 3 mm and is flat from 2 to 5; the RATIO rises with
distance throughout. At 2 mm a random walker arrives nearly a third of the time,
which is the worst possible reading for a demonstration whose whole point is that
the wiring matters. The declared primary of 5 mm is near optimal on both measures
and is kept unchanged.

**Start heading: the exclusion is right, and small.** Rejecting any episode whose
initial heading lies within THETA of facing the source, at r0 = 5 mm:

| THETA | real | floor | ratio |
|---|---|---|---|
| 0 deg (none) | 0.682 | 0.164 | 4.17x |
| 30 deg | 0.673 | 0.154 | 4.35x |
| 60 deg | 0.654 | 0.148 | 4.41x |
| **90 deg** | **0.631** | **0.137** | **4.62x** |

Free arrivals were real but not dominant: 30 degrees of turn per frame washes the
initial heading out within 20-30 frames. **THETA_MIN = 90 degrees is adopted
anyway.** It costs the real wiring 0.05 and buys a sentence that needs no
defending — not one fly began pointed at the smell. Knowing an effect is small
because it was measured is not the same as never having looked.

**A statistic that only works because it was fixed.** Smoke's first N4 compared
path length among ARRIVALS and ranked shuffled wiring as "straighter" (1.30
against 1.75) because an arm that rarely arrives only arrives when it started
lucky — the statistic rewarded failing. Replaced by PROGRESS, (start distance -
final distance) / path length, defined for every episode. It immediately showed
what the biased version hid:

| arm | arrival | progress | final distance |
|---|---|---|---|
| REAL | 0.417 | +0.0079 | 4.21 mm |
| SHUF-1000 | 0.125 | +0.0054 | 4.46 mm |
| RANDOM-WALK | 0.250 | **-0.1057** | **15.57 mm** |
| ORACLE | 1.000 | +0.9315 | 0.41 mm |

**The random walker's 0.250 arrival was entirely luck: it ends up three times
further away than it started.** This is the third statistic in this project to
measure something other than what it appeared to — after step 8's D3 and step
12's RATEACC — and the second caught by a smoke test rather than by review.

**In-loop accuracy matches the static prediction.** Step 11's curve reads the
contrast at 5 mm as per-step accuracy ~0.551; the network measured 0.541 running
inside the loop. Two different code paths, a fixed point and 400 frames of
dynamics, agreeing to the second decimal.

---

## PR-11 — step 13, closing the loop, registered before the run

Registered 2026-09-21, against rules committed in `step13_surge.py`, before the
run. Nothing is trained. One centering scalar per arm, estimated on static
presentations outside the loop. Held-out odors and held-out ORNs only — the
closed-loop form of step 11's test B condition.

| rule | prediction | reason |
|---|---|---|
| N0 harness works | pass | ORACLE steers on the true concentrations; smoke gave 1.000 |
| N1 better than luck | pass | smoke 0.417 against 0.250, and the random walker's progress is NEGATIVE |
| N2 wiring earns it | **uncertain** | smoke had one seed at n=24. REAL 0.417 against SHUF 0.125 is encouraging and is four coin flips |
| N3 not one geometry | **uncertain** | never measured at 2 or 10 mm with the network |
| N4 makes progress | **uncertain** | REAL +0.0079 against SHUF +0.0054 at n=24 is nothing |
| THE WIRING WALKS TO THE SMELL | lean pass | N1 is near-certain, N2 and N4 are not |

**Unlike PR-8 and PR-9, most of this is genuinely unknown.** Steps 11 and 12
confirmed lookups on more data. This one asks whether a per-step accuracy of
0.541 survives 400 frames of compounding, and the smoke's single shuffled seed
does not answer it.

**The registered mechanism, and what would confirm it.** The geometry model
predicts 0.682 arrival at 5 mm from step 11's own accuracy curve; the network
managed 0.417 while its per-step accuracy matched the prediction almost exactly
(0.541 measured against 0.551 predicted). **Right decisions, worse outcome.**

The hypothesis is that the geometry model treats each frame as an independent
coin flip, while the network carries state: the same odor and the same internal
activity persist, so its errors are CORRELATED ACROSS TIME. A run of wrong turns
costs far more than the same number scattered. If this is right it is the price
of step 12's finding — a network that integrates 2.4x more slowly than its
shuffles holds onto a wrong heading longer too, and the rate signal and the
sluggish turn are the same property seen twice.

**What would falsify it:** if REAL's arrival at 200 episodes comes in near 0.68,
the gap was small-sample noise and no correlation story is needed. Registered so
that a pleasing mechanism cannot be adopted after the fact.

**Not claimed even on a full pass.** That the fly navigates as a real fly does —
this is the smooth-gradient regime only, and walking flies in complex plumes use
encounter timing. That the wiring supplies the approach behaviour — "turn toward
the more stimulated side" is declared and identical in every arm. That the result
is strong — five seeds buy one-sided p ~ 1/6, the same weakness steps 11 and 12
carry and the same parked fix.

---

## LK-17 — lookup for a parked step 14: the compass, and a paper aimed at this project

`owner-doubt:` asked for prior art on the ring-attractor idea before any of it was
built, the same challenge that produced LK-15. It found the question half
answered already, and a second paper that tests the premise steps 11 to 13 rest
on. Recorded now; nothing is run.

**Why a compass is the missing stage at all.** Steps 11 to 13 wire odor straight
to steering, which is chemotaxis — a bacterium, not a fly. The real path is
odor -> LH/MB valence -> FC goal direction -> PFL3 comparing goal against the EPG
heading -> DNa02. **Odor does not specify a direction; it updates a goal, and
steering comes from the difference between a compass and that goal.** All of it
is in MaleCNS: EPG 50, PEN 42, PEG 18, Delta7 42, PFL1/2/3 50, FC 274, ER 282,
DNa02 2.

This also explains flyverse's failure from LK-15. It tried to steer through PFL3
and DNa02 and found them silent, then bypassed the circuit. **PFL3 with no EPG
input has nothing to compare**; the circuit was not broken, half its input was
missing.

**The ring attractor is largely done, by Janelia.** Biswas, Stanoev, Romani and
Fitzgerald (2026) built a model satisfying connectome constraints and continuous
heading representation together. Two findings bear on a step 14:

- **Raw connectome weights are not enough as-is**: "synaptic variations among and
  around the fly connectomes can be compensated by **cell-type-specific rescaling
  of synaptic weights**", possibly achieved by neuromodulation. Theory agrees —
  ring attractors need finely tuned connectivity, and reconstructions carry
  asymmetries from measurement noise and biological variation.
- **But the requirement is weaker than assumed.** They found a new class of ring
  attractor with **weaker symmetry requirements**, where continuous heading comes
  from combining symmetric and anti-symmetric activity. So "the raw wiring cannot
  do it" is NOT the settled answer, and a plain test is still worth running.

They report three realizations across four fly connectomes. **Whether MaleCNS is
among them is unchecked, and whether any of it survives THIS pipeline's PREDICTED
signs — glutamate inhibitory, 3,177 unclear neurons defaulted to +1 — has not
been asked by anyone.** That assumption carries steps 11 to 13 as well.

**A paper aimed squarely at this project, and why it does not land.**
"Topological Sensitivity in Connectome-Constrained Neural Networks" builds a
control ladder on the fly connectome and finds the connectome's advantage
COLLAPSES under a proper null:

| null | connectome advantage |
|---|---|
| naive random graph | loss +0.184, activity +1.206 |
| **degree-preserving rewiring** | **loss +0.0003, activity -0.011 (reversed)** |

Its conclusion is that apparent topology advantages "depend critically on
**initialization and null-model design** rather than reflecting genuine
topological superiority."

**Its confound cannot occur here, and the reason is structural, not rhetorical.**
That result is about TRAINED networks — loss, checkpoints, initialization. Steps
11 to 13 train nothing and have no initialization. And the null it prescribes is
the one already in use: presynaptic partner permutation preserving degrees and
counts, plus step 7's 20 degree-preserving rewirings. Step 11 scored 0.715
against ten such nulls spanning 0.441 to 0.620, under rules committed before the
run. **The check that paper says is required was already passed, and passed
before the paper was read.**

**What a step 14 should therefore ask**, since "does a bump form" is mostly
answered: **does the compass stand up under THIS pipeline's assumptions, and if
it falls, is topology or sign to blame?** Two nulls side by side —
degree-preserving rewiring (breaks topology, keeps signs) against sign flipping
(keeps topology, breaks signs). Whichever destroys the bump more is the answer.

It tests the assumption steps 11 to 13 all stand on, it needs step 12's dynamics
and could not have been asked before it, the subnetwork is under 1,000 neurons,
and **a failure is a result** — "predicted signs will not hold a compass" applies
to every whole-brain simulation built on this data, not only to this one.

### LK-17 addendum — the precondition, checked

LK-17 made a step 14 conditional on whether Biswas et al. had already used
MaleCNS. Checked 2026-09-21, before anything was built.

**They did.** The v2 preprint covers four connectomes containing the central
complex — Male CNS, hemibrain, flywire-fafb and banc — finding three shared
realizations in each. The v1 preprint used hemibrain alone.

**But they never ran unscaled weights.** To turn synapse counts into strengths
they introduce **four fitted scale factors** (gamma_ab), because the
proportionality constant varies with pre- and post-synaptic type, and those
parameters are described as essential to finding viable ring attractors at all.

**That is exactly the assumption this pipeline makes and they do not.** Here a
weight is `count / in_total_full[post]` times a predicted sign, with nothing
fitted — the rule the skill fixes and steps 11 to 13 all inherit. So the open
question is not "does the MaleCNS compass work", which is answered, but **"does
it work without the four scale factors", which is the version this project
actually runs.** A step 14 is complementary rather than redundant, and it is
narrower and better posed than LK-17 first framed it.

Related, noted and not yet read: "Hidden symmetries in network connectivity
support ring attractor dynamics in the fly's neural compass" (2026), and "How the
fly holds a single goal: normalization, not selection, in Drosophila FC2" for the
goal-direction stage that would follow a compass.

---

## LK-18 — lookup for step 14: the contraction problem, and a ring that is not yet identified

`owner-doubt:` on being told the compass might work because direction-finding
did, the naive test was found to be impossible by construction, not merely
unlikely. Asked to take the spectral route instead. Measured the same night; no
rules exist and nothing below is a result.

**The naive test cannot be run at all, and the reason is our own stability
choice.** The skill fixes `g < 1` so the iteration is a contraction — rows of
|W| sum to at most 1 and ReLU is 1-Lipschitz. **A contraction has exactly one
fixed point, by Banach.** A ring attractor needs a CONTINUUM of stable states,
one per heading. At g = 0.9 any bump melts into the single fixed point whatever
the wiring looks like. Asking "does the bump persist" at the settings steps 11 to
13 use would have produced a confident NO that says nothing about the connectome.

**The property that makes steps 11 to 13 work is the property that forbids a ring
attractor.** A unique fixed point is why "the network's response to this
stimulus" is well defined at all. This tension is worth keeping in view: the
model was built for questions of the form "is this response larger than that
one", and a compass is not that kind of question.

**So the test moved to the spectrum**, where no dynamics and no g are needed: a
ring attractor leaves a signature in the connectivity itself. Subnetwork: EPG 46,
EPGt 4, Delta7 42, PEN_a 20, PEN_b 22, PEG 18 = **152 neurons, 7,390 internal
edges, density 0.320.**

**The first null was broken, and would have passed anything.** Permuting
presynaptic partners across all 166,700 neurons leaves almost no edge landing
back inside a 152-neuron subnetwork: shuffled leading eigenvalues came out at
0.006 to 0.053 against the real 0.602. **That null deletes the subnetwork rather
than its ring structure.** This is the fourth statistic in this project to
measure something other than what it appeared to, after step 8's D3, step 12's
RATEACC and step 13's first N4.

**With a proper null — degree-preserving double-edge swaps WITHIN the subnetwork,
in- and out-degree exactly preserved:**

| | annulus | cos-Fourier | leading \|eigenvalue\| |
|---|---|---|---|
| REAL | **4.321** | 0.029 | **0.6015** |
| 10 rewired seeds | 1.280–1.812 | 0.000–0.026 | 0.1618–0.1753 |

**There is structure.** The real wiring beats all ten seeds on both the leading
eigenvalue (3.4x) and how ring-like its spectral embedding is (2.4x), against a
null spanning only 0.162 to 0.175.

**It has NOT been shown to be the head-direction ring, and the difference
matters.** The label-based test — is connection strength a cosine function of
angular difference between glomeruli — gave 0.029 against a null of 0.026. Both
are zero. The likely cause is crude label parsing rather than biology:
**Delta7 instances span three glomeruli** (`Delta7(PB15)_L1L9R8_R`) and the
parser took only the first, and PEN and PEG carry their own offsets. A proper PB
glomerulus to heading mapping is a precondition for any step 14, and until it
exists "the wiring has more structure than its rewirings" is all that is
measured — which is true of many subnetworks and is not a compass.

### LK-18 addendum 2 — the direction, settled from the literature

`owner-doubt:` asked whether the figure could simply be fetched and read. It
could, and two figures were the wrong ones, and a third was read two
incompatible ways at the available resolution. The text settled it where the
picture could not.

Hulse et al.'s central-complex connectome paper states both halves plainly:

> "The 16 EPG wedges in the EB **alternate** so that half go to the right PB
> while half go to the left." ... "the bump on the right side will be shifted
> **22.5°** with respect to the bump on the left"

> "PEN_a neurons on the **left** side of the PB send projections to the EB that
> are **counterclockwise** shifted ... on the **right** side ... **clockwise**
> shifted."

**So a mirror-symmetric bridge with indices counting outward from the midline
requires the SAME index-offset sign on both hemispheres to produce opposite
rotations.** That is what the lookup measured — PEN_a L -0.742 / R -0.896,
PEN_b L -0.512 / R -0.843.

**And that reading is EXPLORATORY and is not evidence.** The numbers were seen
before the direction was fixed, which is exactly the case METHOD.md's
exploratory/confirmatory rule governs: the numbers that generated a hypothesis
are never later cited as evidence for it. They are also not impressive on their
own — REAL's |L-R| of 0.154 is second smallest of eleven, not an outlier.
Recorded as a reading with no verdict attached.

**The confirmatory test therefore goes on a claim the data has not touched**:
that EPG wedges ALTERNATE hemispheres around the EB. Committed in
`step14_compass.py` before the run.

---

## PR-12 — step 14, does the compass ring alternate hemispheres, registered before the run

Registered 2026-09-21, against rules committed in `step14_compass.py`. Nothing is
trained and the statistic has no parameters. The direction of the test comes from
Hulse et al., quoted above, not from the data.

| rule | prediction | reason |
|---|---|---|
| C0 pool is balanced | pass | 23 EPG per side, seen in the lookup |
| C1 alternation | **pass** | the alternation is stated as established anatomy; if the connectome does not show it, either the parsing or the similarity measure is wrong |
| C2 wiring earns it | **uncertain** | rewiring preserves degrees and reuses the same weights, so the null keeps total drive. Whether nearest-neighbour identity survives 20 sweeps of swapping is genuinely unknown |
| C3 holds at k = 3 | **uncertain** | k = 3 spans more than one EB tile and may dilute the alternation |
| THE RING ALTERNATES | lean pass | C1 near-certain, C2 not |

**The real question is C2, not C1.** Alternation is documented anatomy; the test
is whether it is recoverable from raw count-fraction weights with predicted signs
and nothing fitted. **Biswas et al. needed four fitted scale factors to get a
working ring attractor out of these connectomes** (LK-17 addendum). This asks a
smaller thing — a structural signature, not working dynamics — but asks it of the
unfitted weights that steps 11 to 13 actually use.

**A failure of C2 would be informative and is not hedged away here.** It would
mean the alternation is present in the labels but not recoverable from the
weighting scheme, which is a statement about this pipeline's count-fraction rule,
and it would apply to every whole-brain model built the same way.

**Not claimed on a pass.** That the compass works, that a bump would persist — it
cannot at g < 1, by Banach — or that the predicted signs are correct. This is one
structural signature, measured once, against ten seeds, at one-sided p ~ 1/11.

### PR-12 outcome — C1 FAILED, and the pre-registration is what saves the reading

Run 2026-09-21 against rules committed at `malecns-operator@0884fbb`, before the
run. `data/step14_compass.json`, 10 s.

| rule | predicted | outcome |
|---|---|---|
| C0 pool is balanced | pass | PASS, 23 EPG per side |
| C1 alternation | **pass** | **FAILED.** ALT(k=1) = 0.320, below chance |
| C2 wiring earns it | uncertain | FAILED, below all ten seeds (0.380–0.720) |
| C3 holds at k = 3 | uncertain | FAILED, 0.460 against 0.420–0.567 |
| THE RING ALTERNATES | lean pass | FAILED |

**EPG nearest neighbours are 68% IPSILATERAL**, the opposite of what alternation
predicts, and below every rewired seed rather than at chance.

**PR-12 said in advance how to read this:** "if the connectome does not show it,
either the parsing or the similarity measure is wrong." Alternation is documented
anatomy in Hulse et al.; a single similarity measure built at 3 a.m. is not
evidence against it. **Without that sentence registered beforehand, the available
move would have been to announce that MaleCNS lacks the alternation** — a claim
against established anatomy, resting on one unvalidated statistic.

**The failure is structured, not noise.** 0.320 sits below the entire null range,
so the measure is picking up something real — most likely hemisphere membership
in the PB rather than position in the EB. The incoming PEN/PEG profile appears
more sensitive to which half of the bridge a neuron belongs to than to which EB
tile it sits in, which makes it the wrong instrument for this question.

**What a correct version needs:** an EB-position measure that does not inherit PB
hemisphere as a confound — the PEN tile structure used explicitly, or EB wedge
labels from a field not yet examined. Not attempted tonight.

**This is the first registered prediction in this project to come out wrong on
its headline rule.** PR-7 through PR-9 held, PR-10 was stopped, PR-11's mechanism
survived. `PREDICTIONS.md` opens by saying the record "only starts being worth
something once part of it is wrong." That part now exists.

### PR-11 outcome — four of five rules pass, the composite FAILS, and the effect is in the tail

Run 2026-09-21 against rules committed at `malecns-operator@68fc40c`, before the
run. `data/step13_surge.json`, 18,754 s, 2.35 GB. 200 episodes per arm per
condition, five conditions, no episode starting within 90° of facing the source.

| rule | predicted | outcome |
|---|---|---|
| N0 harness works | pass | PASS, ORACLE 1.000 at every condition |
| N1 better than luck | pass | PASS, 0.135 against RANDOM-WALK's 0.120 |
| N2 wiring earns it | uncertain | **PASS**, above all five shuffled seeds |
| N3 not one geometry | uncertain | **PASS**, holds at 2, 5 and 10 mm |
| N4 makes progress | uncertain | **FAILED**, -0.0046 against -0.0044 to -0.0056 |
| THE WIRING WALKS TO THE SMELL | lean pass | **FAILED**, because N4 gates it |

**N2 is clean and consistent.** Arrival at the primary condition: REAL 0.135
against shuffled 0.005, 0.020, 0.025, 0.030, 0.045 — a 3x to 27x gap, and the
ratio holds at 4–5x across all five geometries.

**Shuffled wiring is WORSE THAN RANDOM TURNING.** RANDOM-WALK arrives 0.120 of
the time; every shuffled seed lands between 0.005 and 0.045. A fly steered by
scrambled wiring does worse than one that ignores its brain. Random turning is
neutral; shuffled wiring actively missteers.

**Why N4 failed, stated rather than explained away.** Median progress is
essentially zero for REAL (-0.0046) and for every shuffled seed (-0.0044 to
-0.0056). The typical episode of either arm gets nowhere. **The wiring's
advantage lives entirely in the tail** — how often it arrives — not in how far
the median fly travels toward the source. N4 was defined on the median before the
run and it is not re-tuned now. The composite verdict fails.

**A finding neither PR-11 nor the geometry model anticipated.** In-loop per-step
accuracy is nearly IDENTICAL across arms: REAL 0.532, shuffled 0.526–0.532. Yet
arrival differs 4.5-fold. **Being right equally often does not produce equal
outcomes** — when a decision is correct matters as much as how often. Neither the
static curve of step 11 nor LK-14's independent-coin-flip model can express this,
and it is the clearest thing this run adds.

**PR-11's registered mechanism survived its falsification test.** The geometry
model predicted 0.682 from step 11's own accuracy curve; REAL delivered 0.135
while its per-step accuracy matched prediction (0.532 measured against 0.551).
PR-11 registered that a result near 0.68 would mean small-sample noise and no
correlation story was needed. It came in at a fifth of that, so the story stands
— though the equal-accuracy finding above suggests correlation is not the whole
of it either.

**What this means for step 11.** A static 0.715 on held-out odors and neurons does
NOT translate into navigation. That is a negative result about the transfer, not
about step 11, and it is the question none of the forty existing connectome
projects asked.
