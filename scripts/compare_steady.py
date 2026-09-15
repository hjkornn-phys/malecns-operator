"""Do the compacted networks behave like the k>=3 baseline in a steady-state rate model?

Model (no time axis):  h = ReLU(g * W h + b + u),  W[post, pre] = sign(pre) * w / in_tot[post]
- in_tot is the FULL neuron->neuron input of post (unthresholded), identical for every network,
  so pruning removes drive instead of renormalising it away.
- row abs-sum of W <= 1 and ReLU is 1-Lipschitz, so g < 1 makes the iteration a contraction.
- sign: ACh/DA/5-HT/OA +1, GABA/Glu/His -1, unclear/missing +1 (treated as excitatory).
- b = 0.1 everywhere; responses are h(probe) - h(no probe).
Networks: baseline k>=3; OR 1%; OR 2%; old in>=0.5%; random edge subsets of the baseline matched
in edge count to OR 1% / OR 2% (control: does the criterion matter, or only the size?).
"""
import json, sys, time
import numpy as np
import pandas as pd
import scipy.sparse as sp

G_LIST = [float(x) for x in sys.argv[1:]] or [0.5, 0.9]
B, TOL, MAXIT = 0.1, 1e-6, 2000
rng = np.random.default_rng(0)

ann = pd.read_parquet("annotations.parquet", columns=["bodyId", "superclass", "class"])
ann = ann[ann.superclass.notna()].sort_values("bodyId").reset_index(drop=True)
N = len(ann)
nt = pd.read_parquet("nt.parquet", columns=["body", "consensus_nt"]).set_index("body").consensus_nt
cons = ann.bodyId.map(nt).fillna("missing").astype(str).to_numpy()
SIGN = {"acetylcholine": 1, "dopamine": 1, "serotonin": 1, "octopamine": 1,
        "gaba": -1, "glutamate": -1, "histamine": -1}
sign = np.array([SIGN.get(c, 1) for c in cons], np.float32)
nt_counts = pd.Series(cons).value_counts().to_dict()

d = np.load("nn_edges.npz"); pre, post, w = d["pre"], d["post"], d["w"].astype(np.float64)
in_tot = np.bincount(post, weights=w, minlength=N)
out_tot = np.bincount(pre, weights=w, minlength=N)
fin, fout = w / in_tot[post], w / out_tot[pre]
k3 = w >= 3
masks = {
    "baseline k>=3": k3,
    "OR 1%": k3 & ((fin >= 0.01) | (fout >= 0.01)),
    "OR 2%": k3 & ((fin >= 0.02) | (fout >= 0.02)),
    "old in>=0.5%": k3 & (fin >= 0.005),
}
base_idx = np.flatnonzero(k3)
for name, ref in [("random = OR 1% size", "OR 1%"), ("random = OR 2% size", "OR 2%")]:
    m = np.zeros_like(k3); m[rng.choice(base_idx, int(masks[ref].sum()), replace=False)] = True
    masks[name] = m

sc = ann.superclass.astype(str); cl = ann["class"].astype(str)
readout = (sc.eq("descending_neuron") | sc.str.endswith("_motor") | sc.str.contains("efferent")
           | sc.str.endswith("_endocrine")).to_numpy()
sensory = sc.str.contains("sensory").to_numpy()
probes = {f"sc:{s}": sc.eq(s).to_numpy() for s in
          ["ol_sensory", "cb_sensory", "vnc_sensory", "sensory_ascending"]}
for c in ["olfactory", "gustatory", "mechanosensory", "mechanosensory_tactile",
          "mechanosensory_proprioceptive", "visual", "hygrosensory", "thermosensory"]:
    m = (cl.eq(c) & sensory).to_numpy()
    if m.sum() >= 10: probes[f"class:{c}"] = m
sens_idx = np.flatnonzero(sensory)
for i in range(16):
    m = np.zeros(N, bool); m[rng.choice(sens_idx, max(10, len(sens_idx) // 50), replace=False)] = True
    probes[f"rand{i:02d}"] = m
pnames = list(probes)
U = np.zeros((N, len(pnames) + 1), np.float32)          # last column: no probe
for j, p in enumerate(pnames): U[probes[p], j] = 1.0


def solve(Wg):
    H = np.zeros_like(U)
    for it in range(1, MAXIT + 1):
        Hn = np.maximum(Wg @ H + B + U, 0.0)
        delta = np.abs(Hn - H).max(); H = Hn
        if delta < TOL: return H, it
    return H, -MAXIT


out = {"N": N, "readout_neurons": int(readout.sum()), "sensory_neurons": int(sensory.sum()),
       "nt_counts": nt_counts, "probes": {p: int(probes[p].sum()) for p in pnames},
       "edges": {k: int(v.sum()) for k, v in masks.items()}, "results": {}}
for g in G_LIST:
    R = {}
    for name, m in masks.items():
        t = time.time()
        W = sp.csr_matrix(((g * sign[pre[m]] * w[m] / in_tot[post[m]]).astype(np.float32),
                           (post[m], pre[m])), shape=(N, N))
        H, it = solve(W)
        R[name] = H[:, :-1] - H[:, -1:]
        print(f"g={g} {name}: {int(m.sum())} edges, {it} iters, {time.time()-t:.1f}s", flush=True)
    Rb = R["baseline k>=3"]
    res = {}
    for name in masks:
        if name == "baseline k>=3": continue
        Rc = R[name]; rows = []
        for j, p in enumerate(pnames):
            for scope, sel in [("readout", readout), ("all", slice(None))]:
                b_, c_ = Rb[sel, j].astype(np.float64), Rc[sel, j].astype(np.float64)
                nb = np.linalg.norm(b_)
                if nb < 1e-9: continue
                thr = 1e-3 * np.abs(b_).max()
                resp = np.abs(b_) > thr
                rows.append(dict(probe=p, scope=scope,
                                 r=float(np.corrcoef(b_, c_)[0, 1]) if c_.std() > 0 else 0.0,
                                 rel_err=float(np.linalg.norm(c_ - b_) / nb),
                                 gain=float(np.linalg.norm(c_) / nb),
                                 silenced_pct=float(100 * (resp & (np.abs(c_) <= thr)).sum() / max(resp.sum(), 1))))
        df = pd.DataFrame(rows)
        summ = df.groupby("scope").agg(r_median=("r", "median"), r_min=("r", "min"),
                                       rel_err_median=("rel_err", "median"), rel_err_max=("rel_err", "max"),
                                       gain_median=("gain", "median"),
                                       silenced_pct_median=("silenced_pct", "median"),
                                       probes=("probe", "count"))
        res[name] = {"summary": summ.round(4).to_dict(orient="index"), "per_probe": rows}
    dead = [p for j, p in enumerate(pnames) if np.linalg.norm(Rb[readout, j]) < 1e-9]
    out["results"][str(g)] = {"networks": res, "probes_without_readout_response": dead}
    json.dump(out, open("compare_steady.json", "w"), indent=1)

for g, blk in out["results"].items():
    print(f"\n=== g={g} | probes with no readout response in baseline: {blk['probes_without_readout_response']}")
    for name, v in blk["networks"].items():
        for scope, s in v["summary"].items():
            print(f"{name:22s} {scope:7s} " + " ".join(f"{k}={val}" for k, val in s.items()))
print("\nNT:", nt_counts)
