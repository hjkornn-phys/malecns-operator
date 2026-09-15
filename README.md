# malecns-operator

Network-model work on the MaleCNS v1.0 connectome: fetch, filter, compact,
solve, compare against baseline.

## Setup

```sh
uv sync                 # base: numpy, pandas, pyarrow, scipy (Python 3.13)
uv sync --extra torch   # plus PyTorch with MPS
uv run python <script>
```

## Compute

Default path: `pyarrow`, `pandas`, `numpy`,
`scipy`, sparse on CPU (~72 s, ~1.1 GB peak for the full comparison).

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

The pipeline scripts in `scripts/` read and write the current
directory, so run them from `data/`:

```sh
cd data
uv run --project .. python ../scripts/fetch.py          # ~1.1 GB, ~2 min
uv run --project .. python ../scripts/retention.py
uv run --project .. python ../scripts/retention_or.py
uv run --project .. python ../scripts/compare_steady.py 0.5 0.9
```

Measured here with the pipeline scripts (readout r at g = 0.9: OR 1%
median 0.968 / worst 0.931; random 0.779 / 0.563), 77 s at 1.41 GB.

## Ground-truth tasks (Shiu et al. 2024)

Downloads go to `data/shiu/` (see each
script's docstring for sources):

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

## Layout

- `data/`   — fetched tables and caches (`nn_edges.npz`); not committed
- `runs/`   — outputs; not committed
