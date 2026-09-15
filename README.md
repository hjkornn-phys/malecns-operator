# malecns-operator

Network-model work on the MaleCNS v1.0 connectome: fetch, filter, compact,
solve, compare against baseline, and train a steady-state rate model. Every
number below was measured on one 8 GB Apple M1 laptop.

Each question below is decided by rules written into the script's docstring and
committed BEFORE the run that answers it, so a failed rule stays failed. Where a
result is negative or undecidable, it is reported as such.

**What holds so far**

- The compacted network (`OR 1%`, half the baseline's edges) keeps the
  baseline's readout responses far better than random pruning of equal size —
  [Pipeline](#pipeline).
- Training through the fixed point works: implicit gradients are correct and a
  second-order optimizer recovers known parameters — [Step 1](#step-1-does-training-through-the-fixed-point-work).
- The untrained wiring already carries taste identity to descending neurons,
  including from neurons never used in training; shuffled wiring does not —
  [Step 2a](#step-2a-can-taste-be-read-from-the-untrained-network).

**What does not, and what is still open**

- Similarity to the real fly is NOT established: on 149 experimental outcomes
  the untrained model scores near chance —
  [Result 1](#result-1-similarity-to-the-real-fly-could-not-be-established).
- Whether compaction preserves *behaviour* is undecided. Three task sets have
  now tried and none could separate compaction from random pruning
  ([Result 2](#result-2-agreement-with-the-baseline), Step 2a rule R4,
  [Step 2a-mix](#step-2a-mix-a-harder-task-and-it-decided-nothing)).
- Generalisation to held-out neurons holds for single tastes and breaks for
  mixtures, where the real wiring is no better than shuffled wiring.
- A dark patch reaches the giant fiber with the right size tuning, sidedness and
  spatial pooling, but too weakly to clear the response threshold, and never
  reaches the jump muscle — [Step 4](#step-4-a-dark-patch-reaches-the-giant-fiber-and-stops-there).
  The obvious culprit, the electrical synapses this connectome does not carry,
  turned out not to be it: supplying them changes nothing, because the giant
  fiber's own response is already too small and this rate model has no spike to
  amplify it — [Step 5](#step-5-the-missing-gap-junctions-were-not-the-bottleneck).
- `step2b_taste.py` trains the gains on the Step 2a task. Its rules are
  committed; its results are not in yet.

## Setup

```sh
uv sync                 # base: numpy, pandas, pyarrow, scipy (Python 3.13)
uv sync --extra torch   # plus PyTorch with MPS; needed for steps 1 and 2
```

Every script reads and writes the current directory, so all of them run from
`data/`:

```sh
cd data
uv run --project .. python ../scripts/<script>.py                 # pipeline, tasks
uv run --project .. --extra torch python ../scripts/<script>.py   # steps 1 and 2
```

## Setting a threshold

Every rule here needs numbers — a response threshold, a correlation floor, a
grouping of channels — and choosing them by feel has now cost two runs, so a
**lookup pass comes before the rule is written**, and the numbers it returns go
in the docstring beside the threshold they justify.

A lookup may measure:

- **the null** — the shuffled and random-control distributions, which fix the
  noise floor and so what "above chance" can mean here;
- **scale and feasibility** — whether a stimulus of the intended size moves the
  intended readout at all, and what range that readout spans;
- **items whose answer is already known** from the literature or from an earlier
  scored task, which calibrate sign and magnitude.

A lookup may **not** measure the comparison the verdict is about. Setting the
threshold from the effect under test is how a rule stops being a test. When only
the effect itself would answer the question, the answer is a calibration slice
held out from scoring, not a peek.

What this would have caught:

- **Step 4, E1.** The 0.01 response threshold was inherited from `score_shiu.py`
  without checking it against this readout. Shuffled DNp01 responses span
  0.001–0.004, so no variant of this network could have reached 0.01 at the giant
  fiber: the rule was unreachable before it was run, and looking at the null
  alone would have said so.
- **Step 6, P5.** Bitter was grouped with sugar and water as "food" and required
  to score above the pheromone channels on MN9. Bitter is aversive and suppresses
  proboscis extension — which this repo had already measured, in Shiu task 5 — so
  it goes negative and fails the rule for the correct biological reason. A lookup
  on that known item would have split the grouping.

Both failures stand as recorded. A rule is not re-tuned once its run has been
seen; a sharper question gets a new rule, committed before its own run.

## Terms

| term | meaning |
|---|---|
| baseline | the network at `weight >= 3`, 10.5 M edges — what every variant is compared against |
| OR 1% | the compacted network: `weight >= 3` and at least 1% of either endpoint's synapse budget, 5.5 M edges |
| random control | baseline edges subsampled at random to OR 1%'s size — does the rule matter, or only the size? |
| shuffled | presynaptic partners permuted across baseline edges: same degrees and counts, different wiring |
| readout neurons | descending, motor, efferent and endocrine neurons — the network's output side |
| test A / test B | classification on neurons also used in training / only on held-out neurons |

## Compute

The pipeline scripts use `pyarrow`, `pandas`, `numpy` and `scipy`, sparse on
CPU (~77 s, ~1.4 GB peak for the full comparison).

PyTorch is optional and only worth it where it wins:
- the cell-type-collapsed network (~11.7k², dense, fits in memory) on MPS;
- anything needing gradients (fitting gains or weights).

Measured here (8 GB M1, `scripts/bench_solve.py`, ms per iteration, 29 columns):

| backend | baseline 10.5 M edges | OR 1% 5.5 M edges |
|---|---|---|
| SciPy CSR, CPU | 209 | 116 |
| torch CSR, CPU | 119 | 63 |
| torch CSR, MPS | unsupported | unsupported |
| torch `index_add_`, MPS | 276 | 146 |

Neuron level: torch CSR on CPU (identical to SciPy, 1.8x faster). MPS only for
dense cell-type work.

## Pipeline

The first four scripts build the neuron-to-neuron network and measure it:

- `fetch.py` downloads the three MaleCNS tables anonymously.
- `retention.py` caches neuron edges and measures how many edges and synapses
  survive absolute-count and input-fraction thresholds.
- `retention_or.py` measures the OR rule at `weight >= 3`: an edge is kept when
  it is at least a given fraction of EITHER the target's total input or the
  source's total output. "OR 1%" is that rule at 1%, the compacted network.
- `compare_steady.py` solves a steady-state rate model on the `weight >= 3`
  baseline, the compacted networks and random edge subsets of the same size,
  and compares their responses on readout neurons.

```sh
uv run --project .. python ../scripts/fetch.py          # ~1.1 GB, ~2 min
uv run --project .. python ../scripts/retention.py
uv run --project .. python ../scripts/retention_or.py
uv run --project .. python ../scripts/compare_steady.py 0.5 0.9
```

Measured: 25,582,938 neuron edges; baseline `weight >= 3` 10,520,431 edges;
OR 1% 5,487,781 edges (61.9% of synapse mass). Readout r against the baseline
at g = 0.9: OR 1% median 0.968 / worst 0.931; one random control of the same
size 0.779 / 0.563. 77 s at 1.41 GB.

## Ground-truth tasks (Shiu et al. 2024)

Downloads go to `data/shiu/` (see each script's docstring for sources):

```sh
uv run --project .. python ../scripts/shiu_tasks.py                        # -> shiu_tasks.json
uv run --project .. python ../scripts/lb3_split.py                         # sugar/water check
uv run --project .. python ../scripts/score_shiu.py 0.9 0.5 --seeds=10     # -> score_shiu.json, ~7 min
```

**Task set.** 149 yes/no outcomes from experiments in Shiu et al. 2024 (Nature
634:210, supplementary tables 2, 3, 5, 10): which neurons respond to sugar,
water or Johnston's organ input, which are required for proboscis extension
(PE), which of 106 SEZ types trigger PE when activated, and two bitter/Ir94e
interactions. Shiu's FlyWire neurons were mapped to MaleCNS through FlyWire
`cell_type` → MaleCNS `flywireType` and `synonyms`; 99/106 screen types
matched. aBN2 has no match and its task is skipped.

**Sugar and water.** FlyWire types both GRN sets as LB3. Stimuli use MaleCNS
subtypes per Tastekin et al. (bioRxiv 10.1101/2025.08.25.671814): water = LB3a
(ppk28, 17 bodies), sugar = LB3b + LB3c (Gr64f, 34 bodies); LB3d (salt) and LB4b
are excluded. The assignment rests on morphology. A connectivity fingerprint
trained on Shiu's FlyWire GRNs (leave-one-out 0.895, p = 0.002) calls all 17
LB3a bodies water, robust to removing the scored neurons as features; it does
not separate sugar from LB3d.

**Rules**, fixed in `score_shiu.py` before the first run: responds = max
response > 0.01 (1e-4..1e-1 also reported); required = silencing lowers the
readout by > 20%; sufficient = MN9 response > 0.01, truth = optogenetic PE rate
> 0.

### Result 1: similarity to the real fly could not be established

The untrained model does not reproduce the experiments, so these tasks cannot
say whether compaction keeps the fly's behaviour.

| g = 0.9 | accuracy | balanced accuracy |
|---|---|---|
| answer "no" to everything | 0.671 | 0.500 |
| baseline `weight >= 3` | 0.758 | 0.669 |
| OR 1% (compacted) | 0.758 | 0.664 |
| random, OR 1% size, 10 seeds | 0.760 ± 0.014 | 0.658 ± 0.021 |

- Without the 106-type screen (50 outcomes) every network scores 0.50–0.54;
  "yes" to everything scores 0.72.
- Silencing one type rarely moves MN9 by 20%: the baseline calls 2/10 sugar
  neurons required (truth 7/10) and 1/10 water neurons (truth 6/10).
- The screen finds 2 of 13 PE-positive types.
- g = 0.5 and every threshold give the same picture.

Likely causes: weights are count fractions with chosen g and b, nothing is
fitted, and the ReLU rate model has no firing threshold, which Shiu's spiking
model does. Sign is predicted from neurotransmitter.

### Result 2: agreement with the baseline

What the tasks can measure is whether a network gives the baseline's answers.

| | changed answers (of 149), g = 0.9 | g = 0.5 | agreement excl. 86 screen negatives, g = 0.9 |
|---|---|---|---|
| OR 1% | 4 | 4 | 0.937 |
| random, 10 seeds | 10.6 ± 2.8 (7–15) | 8.0 ± 3.0 (4–12) | 0.848 ± 0.044 |
| seeds changing ≤ 4 | 0/10 | 1/10 | |

- OR 1% keeps the baseline's answers better than random pruning of the same
  size, but the margin is modest, and at g = 0.5 one seed ties it. The single
  seed 0 (15 and 12 changes) had overstated the gap.
- OR 1%'s changes are borderline: three sit within 0.002 of the 0.01 threshold
  (diatom, aDN1, aBN1 from JO-F), and the bitter/Ir94e pair flips near the 20%
  cutoff at both gains.
- Agreement with the baseline is not agreement with the fly (Result 1).

## Step 1: does training through the fixed point work?

The long-term plan is to train this model (then classification tasks). Step 1
checks the training mechanism on its own terms, with targets we generated
ourselves. It says nothing about similarity to the real fly.

**Model** (`scripts/fpmodel.py`):

    h* = ReLU(gamma[superclass(post)] * (W0 @ (alpha[transmitter(pre)] * h*)) + s * u + b)

`W0` is the signed count-fraction matrix, `sign(pre) * weight / total input of
post`. `alpha` (5 transmitter classes) and `gamma` (per superclass) pass through
sigmoids so every gain is below 0.99 and the iteration always contracts. A
global gain `g` is absorbed into `alpha * gamma`, and a threshold `theta` would
only enter as `b - theta`, so neither is a separate parameter. Gradients use
implicit differentiation at the fixed point (adjoint iteration), with no
unrolling.

```sh
uv run --project .. --extra torch python ../scripts/step1_gradcheck.py         # 1a, ~1.5 min
uv run --project .. --extra torch python ../scripts/step1_recovery.py few sub  # 1b, Adam
uv run --project .. --extra torch python ../scripts/step1_identify.py few sub  # 1c, spectrum + Levenberg-Marquardt
uv run --project .. --extra torch python ../scripts/step1_identify.py tens sub 25
uv run --project .. --extra torch python ../scripts/step1_identify.py few full 15
```

### 1a. Gradients are correct

| check | parameters | max relative error | limit |
|---|---|---|---|
| subnetwork (6k neurons), implicit vs finite differences | 25 | 2.2e-5 | 1e-4 |
| subnetwork, implicit vs backprop through the unrolled iteration | 25 | 2.8e-13 | 1e-6 |
| full baseline (10.5 M edges), implicit vs finite differences | 10 | 1.4e-8 | 1e-3 |

Full baseline, float32, 29 stimuli: forward + backward 6.9 s, 1.17 GB peak
(8 GB M1).

### 1b/1c. Known parameters are recovered, with a second-order optimizer

A teacher model with known parameters generates responses; a student starts
from perturbed parameters (raw values + N(0, 0.7)) and trains on them.
Held-out stimuli are different sensory groups. Pass: held-out loss <= 1e-3 of
its start, every alpha within 0.01, s within 1%, b within 0.005.

| run | params | optimizer | held-out loss, end / start | max alpha error | result |
|---|---|---|---|---|---|
| subnetwork | 7 | Adam, 300 steps | 1.3e-5 | 0.32 | fail |
| subnetwork | 24 (+ gamma per superclass) | Adam, 300 steps | 3.4e-4 | 0.12 | fail |
| subnetwork | 7 | LM, 5 iterations | 6.1e-26 | 2.5e-11 | pass |
| subnetwork | 24 | LM, 25 iterations | 8.2e-18 | 2.6e-8 (gamma 1.3e-7) | pass |
| full baseline (166,700 neurons), loss on 37,328 readout outputs | 7 | LM, 6 iterations | 7.9e-23 | 2.2e-8 | pass |

- Adam matches the outputs but stalls short of the parameters. The model is
  badly conditioned: `J^T J` condition number 2.2e6 for 7 parameters and
  1.25e11 for 24, with the softest directions on transmitter classes and
  superclasses that carry little synaptic weight.
- Levenberg-Marquardt from the same start recovers every parameter, including
  gamma for superclasses with a single neuron in the subnetwork. By the rule
  fixed in `step1_identify.py`: identifiable; the optimizer was the problem.
- On the full network LM also recovers all 7 parameters with the loss seen only
  on readout neurons (condition number 4.07e8, softest direction alpha for
  histamine; 825 s, 1.6 GB peak on the 8 GB M1). Adam was not run at full size.
- LM here uses finite-difference Jacobians, affordable only for a few
  parameters. Larger models (classification) will need L-BFGS, Gauss-Newton
  with implicit Jacobian-vector products, or reparameterisation; plain Adam is
  expected to stall.

## Step 2: classification

Step 1 used targets the model generated itself. Step 2 gives it a task with many
labelled samples: name the taste from the network's output. Step 2a asks what
the untrained wiring already does; Step 2b (rules committed, results not in yet)
trains the gains on the same task and split.

### Step 2a: can taste be read from the untrained network?

Before training anything, check whether the network as wired already carries
taste to its output. Rules were committed (`91bd611`) before the run.

```sh
uv run --project .. --extra torch python ../scripts/step2a_taste.py --seeds=10   # ~15 min, 1.58 GB
```

**Task.** Four taste groups from the Shiu mapping: sugar (34 bodies), water
(17), bitter (38), Ir94e (32). A sample switches on a random half of one
group, plus weak random background from the other 1,307 gustatory neurons. A
linear classifier (logistic regression) reads the 1,314 descending neurons.
30% of each group's bodies are held out. Test A reuses training bodies in new
combinations; test B uses only held-out bodies, so it asks whether the network
pools neurons of the same taste. Chance is 0.25.

| network | test A | test B (held-out bodies) |
|---|---|---|
| input only, no network | 1.000 | 0.269 |
| baseline `weight >= 3` | 1.000 | 0.712 |
| OR 1% (compacted) | 1.000 | 0.737 |
| random, OR 1% size, 10 seeds | 1.000 | 0.702 (0.500–0.831) |
| shuffled wiring, 10 seeds | 1.000 | 0.267 (0.094–0.425) |

| rule | result |
|---|---|
| R1 readout carries taste: baseline test A >= 0.5 | pass |
| R2 generalises: baseline test B >= 0.5 and above input only | pass |
| R3 wiring matters: baseline test B above every shuffled seed | pass |
| R4 compaction keeps it: OR 1% within 0.05 of baseline and above every random seed | **fail** |

- The wiring carries taste identity to descending neurons, including from
  bodies never used in training; with partners shuffled (same degrees, same
  counts) it does not.
- It is uneven. Baseline test B recall: water 1.00, bitter 0.95, Ir94e 0.75,
  **sugar 0.15** (34 of 40 held-out sugar samples called water). `viz_export.py`
  says why: across descending neurons the sugar and water responses correlate at
  r = 0.942, and 78 of the 133 descending neurons that respond to sugar also
  respond to water. Bitter is independent of sugar (r = -0.018). At this
  readout the two sweet-ish channels are close to one signal.
- R4 fails because the task cannot separate pruning rules: OR 1% scored above
  the baseline, but 9 of 10 random seeds came within 0.05, and 160 test samples
  carry about ±0.07 of sampling noise. It is not evidence that compaction hurts.
- Test A is 1.000 for every network, including shuffled ones, so it measures
  nothing here. Shuffled networks drive 98% of descending neurons against 32%
  for the baseline, so part of their failure may be saturation.

### Step 2a-mix: a harder task, and it decided nothing

Step 2a could not separate compaction from random pruning (R4). This run tried
to fix that with a harder task and six times the test samples. Rules were
committed (`1daaf27`) before the run.

```sh
uv run --project .. --extra torch python ../scripts/step2a_mix.py --seeds=10   # ~44 min, 1.51 GB
```

Each sample now mixes two tastes and the label is the dominant one, at 1.0
against 0.3–0.8. Test B grows from 160 to 1,000 samples. The deciding measure is
no longer accuracy but **disagreement**: the share of test B samples where a
network's answer differs from the baseline's, paired over the same samples.

| network | test A | test B | d vs baseline [95% CI] | disagreement [95% CI] |
|---|---|---|---|---|
| baseline `weight >= 3` | 0.970 | 0.308 | — | — |
| OR 1% | 0.965 | 0.390 | +0.082 [+0.065, +0.099] | 0.152 [0.129, 0.174] |
| random, 10 seeds | 0.940–0.960 | 0.389 mean (0.305–0.536) | — | 0.185 mean (0.080–0.324) |
| shuffled, 3 seeds | 0.930–0.938 | 0.254–0.318 | — | 0.383–0.568 |

| rule | result |
|---|---|
| M0 task is informative: baseline test B in [0.35, 0.95] | **fail** (0.308) |
| S wiring still matters: baseline above every shuffled seed | **fail** (shuffled seed 0 reaches 0.318) |
| C1 OR 1% loses no accuracy: lower CI bound of d >= -0.05 | pass |
| C2 OR 1% beats random pruning: lowest disagreement | **fail** (3 seeds disagree less) |
| **compaction kept** | **not decided** |

- M0 fails, so by its own rule this task decides nothing about compaction; C1 and
  C2 are reported but carry no weight. The threshold was not revisited after the
  result was seen.
- The failure is informative. Mixtures are read almost perfectly off neurons
  used in training (0.970) and barely above chance off held-out ones (0.308,
  chance 0.25). Single tastes generalised to held-out neurons (Step 2a, 0.712);
  "which of two tastes is stronger" does not.
- S fails here too, so on this task the real wiring is no better than shuffled
  wiring. Step 2a's R3 result stands on its own task, not on this one.
- Every pruned network scores at or above the baseline on test B, which is what
  a set of near-chance scores looks like when noise is the main signal; it is not
  evidence that pruning helps.

## Step 4: a dark patch reaches the giant fiber, and stops there

Rules were committed (`ec8b12e`) before the run. Read the docstring for what this
does NOT test: not looming, because frames are solved independently and the model
has no time axis; not movement, because there is no body.

```sh
uv run --project .. python ../scripts/step4_size.py --seeds=10   # ~3 min, 1.2 GB
```

A dark disc is grown over the right eye and drives `L2`, the OFF-pathway lamina
output, on its 893 hex-addressed columns. Photoreceptors carry no hex coordinates
in MaleCNS, so the retinotopy has to be taken from the columnar neurons.

| stage at r = 18 (651 columns) | response |
|---|---|
| LC4 | 0.0477 |
| LPLC2 | 0.0198 |
| **DNp01 (giant fiber), stimulated side** | **0.0061** |
| DNp01, opposite side | 0.0001 |
| PSI | 0.0001 |
| TTMn (jump muscle) | 0.0002 |
| DLMn (wing) | 0.0001 |

| rule | result |
|---|---|
| E1 command survives: DNp01 > 0.01 at r = 18 | **fail** (0.0061) |
| E2 size tuned: Spearman(radius, DNp01) >= 0.9 | pass (1.000) |
| E3 contiguity matters: disc beats all 10 scattered seeds at every r >= 6 | pass |
| E4 reaches the muscle: TTMn and DLMn > 0.01 | **fail** (0.0002, 0.0001) |
| E5 the wiring did it: baseline above every shuffled seed | pass (0.0061 vs 0.0044 max) |
| E6 side is right: DNp01 ipsilateral > contralateral | pass (60x) |
| **escape pathway carried** | **no** |

- The pathway is real and behaves like itself. The response rises monotonically
  with disc size, a contiguous disc beats a scattered stimulus of identical
  neuron count at every size, and the side separation is 60-fold. Shuffled
  wiring does not reproduce it, though one shuffled seed reached 0.0044.
- It is also weak, and it dies at the giant fiber. LC4 is well above the 0.01
  threshold; DNp01 is 1.6x below it; below DNp01 there is nothing left.
- The reason is in the data, not the model. `DNp01 -> PSI` is **4 edges, 16
  synapses** and `DNp01 -> TTMn` is **2 edges, 90 synapses**, against 36,733
  synapses arriving at DNp01. In the fly those are **electrical** synapses, and
  MaleCNS is a chemical-synapse connectome with no gap junctions. The escape
  command physically cannot leave the giant fiber in this dataset.
- Contiguity matters most when the disc is small (r = 6: 0.0016 vs 0.0012) and
  almost not at all when it covers most of the eye (r = 18: 0.0061 vs 0.0057),
  which is what pooling over neighbouring columns should look like.
- The threshold was not revisited after the result was seen.

## Step 5: the missing gap junctions were not the bottleneck

Step 4 stopped at the giant fiber, and the obvious suspect was the connectome
itself: in the fly, `DNp01 -> TTMn` and `DNp01 -> PSI` are ShakB-mediated
rectifying **electrical** synapses alongside their chemical ones, and MaleCNS
carries no gap junctions, so only the minor partner is in the data. This run puts
the electrical component back as a **declared modification** and measures what it
buys. Rules were committed (`e7015cd`) before the run.

```sh
uv run --project .. python ../scripts/step5_gapjunction.py --seeds=10   # ~14 min, 1.2 GB
```

No pairing is invented: the chemical edges already sit on those pairs, so the
endpoints come from the data and only the strength is assumed. `kappa` is the
share of the target's input budget the gap junction supplies, its other inputs
giving way, so the iteration stays a contraction.

| kappa | DNp01 | PSI | TTMn | DLMn |
|---|---|---|---|---|
| 0 | 0.0061 | 0.0001 | 0.0002 | 0.0001 |
| 0.2 | 0.0061 | 0.0006 | 0.0012 | 0.0001 |
| 0.4 | 0.0061 | 0.0011 | 0.0023 | 0.0001 |
| 0.8 | 0.0061 | 0.0022 | 0.0044 | 0.0001 |

| rule | result |
|---|---|
| G0 regression: kappa = 0 reproduces step 4 | pass (deviation 0.00e+00) |
| G1 reaches the muscle: some kappa <= 0.4 puts TTMn and DLMn above 0.01 | **fail** |
| **gap junctions were the bottleneck** | **no** |

G2-G5 were not evaluated, since the rules make them conditional on G1.

- The added synapse does exactly what it should: TTMn lands on `g * kappa *
  DNp01` to three decimals at every kappa. The hypothesis was implemented
  faithfully and still failed.
- It fails because the bottleneck is upstream. DNp01 itself only reaches 0.0061,
  and nothing downstream can exceed what arrives. Even at kappa = 0.8, an absurd
  assumption where the gap junction supplies four fifths of the motor neuron's
  entire drive, TTMn reaches 0.0044, under half the threshold.
- DLMn never moves at all, at any kappa. It sits two hops out, behind a PSI that
  is itself barely driven; `PSI -> DLMn` is chemical and fully present, so the
  serial losses, not a missing edge, are what silence it.
- So the honest answer to "is the connectome missing the escape pathway" is: not
  in the way we guessed. What is missing is the **spike**. The real giant fiber
  is an all-or-none amplifier, and a rate model with no threshold has no such
  step — the same gap Result 1 already blamed for the Shiu task failures.
- Reported as a failed hypothesis, not retuned. Raising kappa past 0.8 would only
  be fitting the assumption to the wanted answer.

## Step 6: pheromone routing — the question was not answered, and the design is why

Rules were committed (`f208010`) before the run. Three receptor-labelled channels
(putative ppk23 269, ppk25 257, IR52b 226, all leg and wing bristle GRNs) were
compared against sugar, water and bitter, every channel subsampled to 17 bodies
over 10 draws, scored as the share of a channel's drive landing on a readout.

```sh
uv run --project .. python ../scripts/step6_pheromone.py --seeds=10   # ~7 min, 1.3 GB
```

| readout | ppk23 | ppk25 | IR52b | sugar | water | bitter | unlabelled leg/wing |
|---|---|---|---|---|---|---|---|
| pC1 | 0.016 | 0.041 | -0.167 | -0.023 | -0.038 | -0.145 | -0.193 |
| male-specific | -0.257 | -0.003 | **0.688** | 0.056 | 0.184 | 0.143 | -0.198 |
| fru_high | 0.007 | 0.117 | 0.363 | 0.022 | 0.100 | 0.076 | -0.091 |
| MN9 | -0.020 | -0.009 | 0.010 | **3.652** | **1.190** | **-0.389** | -0.050 |

| rule | result |
|---|---|
| P1 pheromone to courtship | **fail** |
| P2 not just anatomy | pass |
| P3 male-specific routing | **fail** |
| P4 the wiring did it | **fail** (baseline gap +0.032, shuffled max +0.158) |
| P5 reverse control | **fail** |
| P6 compaction keeps it | **fail** |
| **routing reverse-engineered** | **no** |

Two design errors, both of the same kind: a conjunctive rule over a group whose
members do not behave alike.

- **P5 grouped bitter with sugar and water.** Bitter is aversive and suppresses
  proboscis extension, which this repo had already measured in Shiu task 5, so it
  lands at -0.389 on MN9 and sinks the rule. What the numbers do show, as a
  description and not as a passed test, is sugar 3.652 and water 1.190 against
  about 0.01 for every pheromone channel — the appetitive separation the gate was
  built to look for, at more than a hundredfold. Testing that needs its own rule,
  committed before its own run.
- **P1 and P3 treated ppk23, ppk25 and IR52b as one category.** They are not.
  IR52b is the strongest router to male-specific neurons of anything measured
  (0.688) while sitting below bitter on pC1; ppk23 is the reverse, positive on
  pC1 and -0.257 on male-specific. Three channels, three behaviours.
- **P4 is the one that matters most.** The baseline's pheromone-food gap on pC1
  is +0.032 while shuffled seeds range up to +0.158. The effect is inside the
  noise band of scrambled wiring, so even the ppk23 and ppk25 positivity is not
  established. A lookup on that null would have said so before the rule was
  written, which is now the standing procedure — see
  [Setting a threshold](#setting-a-threshold).

P2 passes: all three channels beat the 402 unlabelled leg and wing GRNs, so the
anatomical confound is at least partly controlled. It is the only thing this run
establishes.

## Layout

- `data/` — downloaded tables, caches, logs and result JSON; not committed.
  Every script writes its console output next to its results there.
- `scripts/` — run from `data/`, in this order:

| step | scripts |
|---|---|
| network | `fetch.py`, `retention.py`, `retention_or.py`, `compare_steady.py`, `bench_solve.py` |
| ground truth | `shiu_tasks.py`, `lb3_split.py`, `score_shiu.py` |
| model | `fpmodel.py` — the trainable rate model, imported by every step-1 and step-2 script |
| step 1 | `step1_gradcheck.py`, `step1_recovery.py`, `step1_identify.py` |
| step 2 | `step2a_taste.py`, `step2a_mix.py`, `step2b_taste.py` |
| step 4 | `step4_size.py` |
| step 5 | `step5_gapjunction.py` |
| step 6 | `step6_pheromone.py` |
| viewer data | `viz_export.py` |

### Viewer data

`viz_export.py` writes `viz_taste.npz` (~7 MB, 8 s): soma positions and taste
responses for the 139,662 of 166,700 neurons that have a `somaLocation`, plus
optional animation frames of the fixed-point iteration, which starts at h = 0
and so spreads outward from the stimulated neurons.

```sh
uv run --project .. python ../scripts/viz_export.py --frames=16
```

The file carries positions, per-stimulus responses, cell-type and superclass
labels, descending/readout flags, and a `meta` JSON of the model settings, so a
viewer needs no connectome data of its own. Neurons without a position are still
simulated — they carry the dynamics — but are not exported, since a viewer
cannot place them. The script measures and writes; it draws nothing.

## License

Code and documentation in this repository: MIT, see `LICENSE`.

Data are not included; the scripts download them. MaleCNS v1.0 and the FlyWire
annotations are CC-BY 4.0, and the Shiu et al. 2024 supplementary tables are
used under their own terms. Cite those sources when you use the data.
