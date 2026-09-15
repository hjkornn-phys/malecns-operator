# malecns-operator

Network-model work on the MaleCNS v1.0 connectome: fetch, filter, compact,
solve, compare against baseline, and train a steady-state rate model. Every
number below was measured on one 8 GB Apple M1 laptop.

## Setup

```sh
uv sync                 # base: numpy, pandas, pyarrow, scipy (Python 3.13)
uv sync --extra torch   # plus PyTorch with MPS
uv run python <script>
```

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

They read and write the current directory, so run them from `data/`:

```sh
cd data
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
cd data
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
cd data
uv run --project .. python ../scripts/step1_gradcheck.py              # 1a, ~1.5 min
uv run --project .. python ../scripts/step1_recovery.py few sub      # 1b, Adam
uv run --project .. python ../scripts/step1_identify.py few sub      # 1c, spectrum + Levenberg-Marquardt
uv run --project .. python ../scripts/step1_identify.py tens sub 25
uv run --project .. python ../scripts/step1_identify.py few full 15
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

## Step 2a: can taste be read from the untrained network?

Before training anything, check whether the network as wired already carries
taste to its output. Rules were committed (`91bd611`) before the run.

```sh
cd data
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
  **sugar 0.15** (34 of 40 held-out sugar samples called water).
- R4 fails because the task cannot separate pruning rules: OR 1% scored above
  the baseline, but 9 of 10 random seeds came within 0.05, and 160 test samples
  carry about ±0.07 of sampling noise. It is not evidence that compaction hurts.
- Test A is 1.000 for every network, including shuffled ones, so it measures
  nothing here. Shuffled networks drive 98% of descending neurons against 32%
  for the baseline, so part of their failure may be saturation.

## Layout

- `scripts/` — pipeline, task and training scripts; run from `data/`
- `data/`    — downloaded tables, caches, logs and result JSON; not committed

## License

Code and documentation in this repository: MIT, see `LICENSE`.

Data are not included; the scripts download them. MaleCNS v1.0 and the FlyWire
annotations are CC-BY 4.0, and the Shiu et al. 2024 supplementary tables are
used under their own terms. Cite those sources when you use the data.
