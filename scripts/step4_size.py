"""Step 4: does the untrained wiring carry a dark patch, by size, to the giant fiber and the jump muscle?

THIS IS NOT A LOOMING TEST. Looming and receding differ only in the ORDER of their frames, and this model
solves every frame independently at steady state, so the two produce the same set of responses by
construction. What is testable without a time axis is angular-size tuning and spatial pooling: does a larger
dark disc drive the escape pathway harder, and does it have to be one contiguous patch?
Nor is it movement. There is no body, no muscle and no physics here: TTMn activity means the command
arrived, not that the fly jumped. The real giant fiber is an all-or-none, single-spike event; this is a rate
model with no threshold, the same gap that Result 1 blamed for the Shiu task failures. Magnitudes, not firing.

Input, an engineered mapping, though a retinotopic one. Photoreceptors (6,098 ol_sensory) carry NO hex
  coordinates in MaleCNS; the coordinates sit on the columnar neurons downstream. So a dark disc drives L2,
  the OFF-pathway lamina output, which has 892 hex-addressed columns per eye (somaSide L/R, both eyes on one
  coordinate system). Hex axial coordinates are placed on a real lattice, x = h1 + h2/2, y = h2 * sqrt(3)/2,
  and the disc is centred on that eye's centroid. A column inside the radius is driven at u = 1.
Radii, in hex units, fixed: 0 (no stimulus), 2, 4, 6, 8, 11, 14, 18.
Model: the repo's, h = ReLU(g W h + b + u), g = 0.9, b = 0.1, float32, tol 1e-6,
  W[post, pre] = sign(pre) * w / in_tot_full[post]. Response = h(stimulus) - h(no stimulus).
Readouts, by max response over the bodies of the group, threshold 0.01 as in score_shiu.py:
  DNp01 (giant fiber) per side, PSI, TTMn (jump muscle), DLMn (wing), and LC4 / LPLC2 as the visual stage.
Networks: baseline k>=3; OR 1%; random baseline edges of OR 1% size, seeds 0..9; shuffled baseline
  (presynaptic partners permuted, degrees and counts kept), seeds 1000..1009.
  To keep the run cheap, only baseline and OR 1% are given the scattered controls and the left-eye stimulus;
  the 20 control networks see the right-eye radius ladder alone, which is all E5 needs.
Scattered control: the same NUMBER of L2 neurons of that eye, drawn at random instead of as a disc, 10 seeds
  (2000..2009), at every radius from 6 up.

Verdicts, fixed before running:
  E1 the command survives   DNp01 ipsilateral response > 0.01 at r = 18
  E2 it is size tuned       Spearman correlation of radius with ipsilateral DNp01 response >= 0.9
  E3 contiguity matters     at every radius >= 6 the disc beats all 10 scattered seeds
  E4 it reaches the muscle  TTMn AND DLMn respond > 0.01 at r = 18
  E5 the wiring did it      baseline's DNp01 at r = 18 above every shuffled seed
  E6 the side is right      right-eye stimulus gives DNp01_R > DNp01_L
  ESCAPE PATHWAY CARRIED = E1 and E4 and E5. E2, E3 and E6 are reported whatever E1 does.
Run from data/ after retention.py:  step4_size.py [--seeds=N]
"""
import json, sys, time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr

SEEDS = int(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--seeds=")), 10))
G, B, TOL, MAXIT, THETA = 0.9, 0.1, 1e-6, 2000, 0.01
RADII = [2, 4, 6, 8, 11, 14, 18]
SCATTER_FROM, N_SCATTER = 6, 10

ann = pd.read_parquet("annotations.parquet",
                      columns=["bodyId", "superclass", "type", "instance", "somaSide",
                               "assignedOlHex1", "assignedOlHex2"])
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
or1 = k3 & ((w / in_tot[post] >= 0.01) | (w / out_tot[pre] >= 0.01))

typ = ann.type.fillna("").astype(str)
inst = ann.instance.fillna("").astype(str)
side = ann.somaSide.fillna("?").astype(str)

# ---- retinotopic input: L2, one hex lattice per eye
l2 = typ.eq("L2") & ann.assignedOlHex1.notna()
EYES = {}
for s in "RL":
    idx = np.flatnonzero((l2 & side.eq(s)).to_numpy())
    h1, h2 = ann.assignedOlHex1.to_numpy()[idx], ann.assignedOlHex2.to_numpy()[idx]
    xy = np.stack([h1 + h2 / 2, h2 * np.sqrt(3) / 2], 1)
    EYES[s] = {"idx": idx, "xy": xy, "centre": xy.mean(0)}
    print(f"eye {s}: {len(idx)} L2 columns, hex centre {EYES[s]['centre'].round(1)}", flush=True)


def disc(eye, r):
    e = EYES[eye]
    return e["idx"][np.linalg.norm(e["xy"] - e["centre"], axis=1) <= r]


def scattered(eye, n, seed):
    e = EYES[eye]
    return np.random.default_rng(seed).choice(e["idx"], min(n, len(e["idx"])), replace=False)


# ---- readouts
def group(name, want_side=None):
    m = typ.eq(name)
    if want_side: m &= inst.str.endswith("_" + want_side)
    return np.flatnonzero(m.to_numpy())


READ = {"DNp01_R": group("DNp01", "R"), "DNp01_L": group("DNp01", "L"),
        "PSI": group("PSI"), "TTMn": group("TTMn"),
        "DLMn": np.flatnonzero(typ.str.startswith("DLMn").to_numpy()),
        "LC4": group("LC4"), "LPLC2": group("LPLC2")}
print("readouts: " + ", ".join(f"{k} {len(v)}" for k, v in READ.items()), flush=True)
assert all(len(v) for v in READ.values()), "a readout group is empty"

# ---- stimulus columns
cols = [("none", None)]
for r in RADII: cols.append((f"R disc r{r}", disc("R", r)))
FULL = len(cols)                                    # control networks stop here
for r in RADII: cols.append((f"L disc r{r}", disc("L", r)))
for r in RADII:
    if r < SCATTER_FROM: continue
    n = len(disc("R", r))
    for s in range(N_SCATTER): cols.append((f"R scatter r{r} s{s}", scattered("R", n, 2000 + s)))
U = np.zeros((N, len(cols)), np.float32)
for j, (_, rows) in enumerate(cols):
    if rows is not None: U[rows, j] = 1.0
print(f"{len(cols)} stimulus columns; disc sizes " +
      ", ".join(f"r{r}={len(disc('R', r))}" for r in RADII), flush=True)


def networks():
    yield "baseline k>=3", pre[k3], post[k3], w[k3], len(cols)
    yield "OR 1%", pre[or1], post[or1], w[or1], len(cols)
    base, n_or = np.flatnonzero(k3), int(or1.sum())
    for s in range(SEEDS):
        i = np.random.default_rng(s).choice(base, n_or, replace=False)
        yield f"random seed {s}", pre[i], post[i], w[i], FULL
    for s in range(SEEDS):
        yield f"shuffled seed {s}", np.random.default_rng(1000 + s).permutation(pre[k3]), post[k3], w[k3], FULL


def solve(W, U):
    H = np.zeros_like(U)
    for it in range(1, MAXIT + 1):
        Hn = np.maximum(W @ H + B + U, 0.0)
        delta = np.abs(Hn - H).max(); H = Hn
        if delta < TOL: return H, it
    raise RuntimeError(f"no convergence in {MAXIT} iterations (last change {delta:.3g})")


out = {"rules": __doc__, "radii": RADII, "theta": THETA,
       "columns": [c[0] for c in cols], "disc_columns": {r: int(len(disc("R", r))) for r in RADII},
       "readout_sizes": {k: int(len(v)) for k, v in READ.items()}, "results": {}}
for name, p_, q_, w_, ncol in networks():
    t0 = time.time()
    W = sp.csr_matrix(((G * sign[p_] * w_ / in_tot[q_]).astype(np.float32), (q_, p_)), shape=(N, N))
    H, it = solve(W, U[:, :ncol]); del W
    resp = {k: (H[v, 1:ncol] - H[v, :1]).max(0).astype(np.float64).tolist() for k, v in READ.items()}
    out["results"][name] = {"resp": resp, "columns": [c[0] for c in cols[1:ncol]], "iters": it,
                            "sec": time.time() - t0}
    dn = dict(zip(out["results"][name]["columns"], resp["DNp01_R"]))
    print(f"{name}: DNp01_R " + " ".join(f"r{r}={dn[f'R disc r{r}']:.3f}" for r in RADII) +
          f" | {it} iters {time.time() - t0:.0f}s", flush=True)
    json.dump(out, open("step4_size.json", "w"), indent=1)

R = out["results"]
at = lambda net, key, col: dict(zip(R[net]["columns"], R[net]["resp"][key]))[col]
base = "baseline k>=3"
rmax = RADII[-1]
ladder = [at(base, "DNp01_R", f"R disc r{r}") for r in RADII]
rho = float(spearmanr(RADII, ladder).statistic)
scat_ok, scat_detail = True, {}
for r in RADII:
    if r < SCATTER_FROM: continue
    s = [at(base, "DNp01_R", f"R scatter r{r} s{i}") for i in range(N_SCATTER)]
    d_ = at(base, "DNp01_R", f"R disc r{r}")
    scat_detail[r] = {"disc": d_, "scatter_max": max(s), "scatter_mean": float(np.mean(s))}
    scat_ok &= d_ > max(s)
shuf = [at(f"shuffled seed {s}", "DNp01_R", f"R disc r{rmax}") for s in range(SEEDS)]
rnd = [at(f"random seed {s}", "DNp01_R", f"R disc r{rmax}") for s in range(SEEDS)]
v = {"E1 command survives": bool(ladder[-1] > THETA),
     "E2 size tuned": bool(rho >= 0.9),
     "E3 contiguity matters": bool(scat_ok),
     "E4 reaches the muscle": bool(at(base, "TTMn", f"R disc r{rmax}") > THETA
                                   and at(base, "DLMn", f"R disc r{rmax}") > THETA),
     "E5 the wiring did it": bool(ladder[-1] > max(shuf)),
     "E6 side is right": bool(ladder[-1] > at(base, "DNp01_L", f"R disc r{rmax}"))}
v["ESCAPE PATHWAY CARRIED"] = bool(v["E1 command survives"] and v["E4 reaches the muscle"]
                                   and v["E5 the wiring did it"])
out["verdicts"] = v
out["stats"] = {"spearman": rho, "ladder": ladder, "scatter": scat_detail,
                "shuffled_rmax": shuf, "random_rmax": rnd,
                "OR1_rmax": at("OR 1%", "DNp01_R", f"R disc r{rmax}")}
json.dump(out, open("step4_size.json", "w"), indent=1)

print(f"\nbaseline DNp01_R by radius: " + " ".join(f"r{r} {x:.4f}" for r, x in zip(RADII, ladder)))
print(f"Spearman(radius, DNp01_R) = {rho:.3f}")
print("stage responses at r%d: " % rmax +
      " ".join(f"{k} {at(base, k, f'R disc r{rmax}'):.4f}" for k in ["LC4", "LPLC2", "DNp01_R", "DNp01_L", "PSI", "TTMn", "DLMn"]))
for r, s in scat_detail.items():
    print(f"  r{r}: disc {s['disc']:.4f} vs scattered max {s['scatter_max']:.4f} mean {s['scatter_mean']:.4f}")
print(f"shuffled r{rmax}: max {max(shuf):.4f} mean {np.mean(shuf):.4f} | random r{rmax}: max {max(rnd):.4f} "
      f"mean {np.mean(rnd):.4f} | OR 1% {out['stats']['OR1_rmax']:.4f}")
for k, val in v.items(): print(f"{k}: {val}")
