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
