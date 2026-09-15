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

Benchmark CPU vs MPS before adopting torch for sparse neuron-level work.

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

## Layout

- `data/`   — fetched tables and caches (`nn_edges.npz`); not committed
- `runs/`   — outputs; not committed
