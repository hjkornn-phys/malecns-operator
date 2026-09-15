"""Time one steady-state iteration h <- ReLU(g W h + b + U) on CPU (SciPy, torch) and Apple GPU (torch MPS).

Same W as compare_steady.py (baseline k>=3 and OR 1%, g = 0.9), same U shape (28 probes + no-probe).
A fixed iteration count is timed so every backend does identical work; results are checked against SciPy.
Run from data/ after retention.py (needs nn_edges.npz, annotations.parquet, nt.parquet).
"""
import json, time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

G, B, ITERS, REPS, CHUNK = 0.9, 0.1, 40, 3, 2_000_000
rng = np.random.default_rng(0)

ann = pd.read_parquet("annotations.parquet", columns=["bodyId", "superclass"])
ann = ann[ann.superclass.notna()].sort_values("bodyId").reset_index(drop=True)
N = len(ann)
nt = pd.read_parquet("nt.parquet", columns=["body", "consensus_nt"]).set_index("body").consensus_nt
cons = ann.bodyId.map(nt).fillna("missing").astype(str).to_numpy()
SIGN = {"acetylcholine": 1, "dopamine": 1, "serotonin": 1, "octopamine": 1,
        "gaba": -1, "glutamate": -1, "histamine": -1}
sign = np.array([SIGN.get(c, 1) for c in cons], np.float32)

d = np.load("nn_edges.npz"); pre, post, w = d["pre"], d["post"], d["w"].astype(np.float64)
in_tot = np.bincount(post, weights=w, minlength=N)
out_tot = np.bincount(pre, weights=w, minlength=N)
k3 = w >= 3
masks = {"baseline k>=3": k3,
         "OR 1%": k3 & ((w / in_tot[post] >= 0.01) | (w / out_tot[pre] >= 0.01))}

sens_idx = np.flatnonzero(ann.superclass.astype(str).str.contains("sensory").to_numpy())
K = 29
U = np.zeros((N, K), np.float32)
for j in range(K - 1):
    U[rng.choice(sens_idx, len(sens_idx) // 50, replace=False), j] = 1.0


def timed(step, sync=lambda: None):
    step(2); sync()                                   # warm-up
    ts = []
    for _ in range(REPS):
        t = time.perf_counter(); H = step(ITERS); sync(); ts.append(time.perf_counter() - t)
    return H, min(ts)


out = {"N": N, "K": K, "iters": ITERS, "reps": REPS, "torch": torch.__version__,
       "threads": torch.get_num_threads(), "results": {}}
for name, m in masks.items():
    E = int(m.sum())
    val = (G * sign[pre[m]] * w[m] / in_tot[post[m]]).astype(np.float32)
    rows = {}

    W = sp.csr_matrix((val, (post[m], pre[m])), shape=(N, N))
    def scipy_step(n):
        H = np.zeros_like(U)
        for _ in range(n): H = np.maximum(W @ H + B + U, 0.0)
        return H
    H_ref, t = timed(scipy_step); rows["scipy cpu"] = t

    Wt = torch.sparse_csr_tensor(torch.from_numpy(W.indptr.astype(np.int64)),
                                 torch.from_numpy(W.indices.astype(np.int64)),
                                 torch.from_numpy(W.data), size=(N, N))
    Ut = torch.from_numpy(U)
    def torch_cpu_step(n):
        H = torch.zeros_like(Ut)
        for _ in range(n): H = torch.relu(Wt @ H + B + Ut)
        return H
    H, t = timed(torch_cpu_step)
    rows["torch cpu csr"] = t; rows["torch cpu csr maxdiff"] = float(np.abs(H.numpy() - H_ref).max())

    if torch.backends.mps.is_available():
        dev, sync = torch.device("mps"), torch.mps.synchronize
        Um = Ut.to(dev)
        try:                                           # sparse CSR matmul on MPS: may be unsupported
            Wm = Wt.to(dev)
            def mps_csr_step(n):
                H = torch.zeros_like(Um)
                for _ in range(n): H = torch.relu(Wm @ H + B + Um)
                return H
            H, t = timed(mps_csr_step, sync)
            rows["torch mps csr"] = t; rows["torch mps csr maxdiff"] = float(np.abs(H.cpu().numpy() - H_ref).max())
        except Exception as e:
            rows["torch mps csr"] = f"unsupported: {type(e).__name__}: {str(e)[:120]}"
        finally:
            Wm = None

        # index_add_ scatter, chunked so pre-gathered rows stay ~CHUNK*K*4 bytes
        pm = torch.from_numpy(pre[m].astype(np.int64)).to(dev)
        qm = torch.from_numpy(post[m].astype(np.int64)).to(dev)
        vm = torch.from_numpy(val).to(dev)
        def mps_scatter_step(n):
            H = torch.zeros_like(Um)
            for _ in range(n):
                Y = torch.zeros_like(Um)
                for s in range(0, E, CHUNK):
                    Y.index_add_(0, qm[s:s + CHUNK], vm[s:s + CHUNK, None] * H[pm[s:s + CHUNK]])
                H = torch.relu(Y + B + Um)
            return H
        H, t = timed(mps_scatter_step, sync)
        rows["torch mps scatter"] = t; rows["torch mps scatter maxdiff"] = float(np.abs(H.cpu().numpy() - H_ref).max())
        pm = qm = vm = None; torch.mps.empty_cache()

    out["results"][name] = {"edges": E, **rows}
    print(name, E, "edges", flush=True)
    for k, v in rows.items():
        print(f"  {k:26s} {v:.4f}s ({v / ITERS * 1000:.1f} ms/iter)" if isinstance(v, float) and "maxdiff" not in k
              else f"  {k:26s} {v}", flush=True)

json.dump(out, open("bench_solve.json", "w"), indent=1)
