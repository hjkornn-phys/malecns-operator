"""Step 5: were the missing gap junctions the bottleneck? A declared modification, run as an ablation.

Step 4 found the escape command reaching DNp01 (the giant fiber) with the right size tuning, sidedness and
spatial pooling, but 1.6x below the response threshold, and nothing surviving past it: DNp01 -> PSI is 4
edges and 16 synapses, DNp01 -> TTMn is 2 edges and 90, against 36,733 synapses arriving at DNp01. In the
fly those two contacts are MIXED: ShakB-mediated rectifying ELECTRICAL synapses alongside the cholinergic
chemical ones (Phelan et al., rectification at GFS electrical synapses; Blagburn et al., eNeuro 2018).
MaleCNS is a chemical-synapse connectome and carries no gap junctions, so only the minor partner is in the
data. This script puts the electrical component back and measures what it is worth.

THIS IS A DECLARED MODIFICATION, NOT A CONNECTOME RESULT. Any positive outcome here is conditional on the
assumption. It may never be reported as "the connectome predicts escape". What it can support is the weaker
and more useful claim: the chemical connectome supplies everything upstream of the giant fiber, and the last
two synapses have to come from physiology.

What changes, and nothing else:
- Only DNp01 -> TTMn and DNp01 -> PSI gain an electrical component.
- No pairing is invented. The chemical edges already sit on exactly those pairs in the k >= 3 baseline, so
  the endpoints (including which side pairs with which) are taken from the data; literature only says that
  the same contact is also electrical. The same pairs are used for every network, including the controls,
  since a declared edge does not depend on which chemical edges a pruning rule kept.
- One direction only, GF -> target, as the rectifying synapse is; sign +1, electrical coupling passing
  depolarisation.
- Strength kappa is a share of the TARGET's input budget: W[target, GF] = g * kappa and the target's other
  inputs are scaled by (1 - kappa). The row sum of |W| therefore stays at most g, the iteration stays a
  contraction, and kappa reads as "the gap junction supplies this fraction of the neuron's drive".
- kappa ladder, fixed: 0, 0.05, 0.1, 0.2, 0.4, 0.8. kappa = 0 must reproduce step 4.

Stimulus, readouts and thresholds are step 4's, unchanged and directly comparable: a dark disc on the right
eye driving L2 over 893 hex columns, radii 2..18, scattered controls of equal neuron count, threshold 0.01.
Networks: baseline k>=3, OR 1%, random seeds 0..9, shuffled seeds 1000..1009 — each with the same declared
edges, so a strong added synapse cannot by itself make a network look right.

Verdicts, fixed before running:
  G0 regression        at kappa = 0, DNp01 and TTMn match step4_size.json within 1e-6
  G1 reaches muscle    some kappa <= 0.4 puts BOTH TTMn and DLMn above 0.01 at r = 18
  G2 computation kept  at that smallest kappa, Spearman(radius, TTMn) >= 0.9
  G3 the wiring did it at that kappa, baseline TTMn above every shuffled seed
  G4 contiguity kept   at that kappa, the disc beats all 10 scattered seeds at every radius >= 6
  G5 side is right     at that kappa, TTMn_R > TTMn_L for a right-eye stimulus
  GAP JUNCTIONS WERE THE BOTTLENECK = G1 and G2 and G3.
G2 and G3 carry the argument. If G1 passes while they fail, the added edge injected a constant rather than
delivering the upstream computation, and filling the gap bought nothing.

Still out of reach, whatever this run says: movement (there is no body, TTMn activity is a command arriving,
not a jump), all-or-none firing (this is a rate model, the gap Result 1 already blamed), and every other gap
junction in the fly, of which only these two are added.
Run from data/ after step4_size.py:  step5_gapjunction.py [--seeds=N]
"""
import json, sys, time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr

SEEDS = int(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--seeds=")), 10))
G, B, TOL, MAXIT, THETA = 0.9, 0.1, 1e-6, 2000, 0.01
RADII = [2, 4, 6, 8, 11, 14, 18]
KAPPAS = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]
KAPPA_MAX_PASS, SCATTER_FROM, N_SCATTER = 0.4, 6, 10

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


def group(name, want_side=None):
    m = typ.eq(name)
    if want_side: m &= inst.str.endswith("_" + want_side)
    return np.flatnonzero(m.to_numpy())


GF = group("DNp01")
GAP_TARGET_TYPES = ["TTMn", "PSI"]
tgt_rows = np.concatenate([group(t) for t in GAP_TARGET_TYPES])
# the declared pairs: exactly where a chemical GF contact already exists in the k>=3 baseline
sel = k3 & np.isin(pre, GF) & np.isin(post, tgt_rows)
GAP_PRE, GAP_POST = pre[sel], post[sel]
GAP_ROWS, inv = np.unique(GAP_POST, return_inverse=True)
GAP_SHARE = 1.0 / np.bincount(inv)[inv]     # kappa is the target's share, split over its declared edges
print(f"declared electrical pairs, taken from existing chemical edges: {len(GAP_PRE)}")
for a, b in zip(GAP_PRE, GAP_POST):
    print(f"  {inst.iloc[a]} -> {inst.iloc[b]}  (chemical synapses {int(w[(pre == a) & (post == b)].sum())})")

# ---- retinotopic input: L2, one hex lattice per eye (identical to step4_size.py)
l2 = typ.eq("L2") & ann.assignedOlHex1.notna()
EYES = {}
for s in "RL":
    idx = np.flatnonzero((l2 & side.eq(s)).to_numpy())
    h1, h2 = ann.assignedOlHex1.to_numpy()[idx], ann.assignedOlHex2.to_numpy()[idx]
    xy = np.stack([h1 + h2 / 2, h2 * np.sqrt(3) / 2], 1)
    EYES[s] = {"idx": idx, "xy": xy, "centre": xy.mean(0)}


def disc(eye, r):
    e = EYES[eye]
    return e["idx"][np.linalg.norm(e["xy"] - e["centre"], axis=1) <= r]


def scattered(eye, n, seed):
    return np.random.default_rng(seed).choice(EYES[eye]["idx"], n, replace=False)


READ = {"DNp01_R": group("DNp01", "R"), "DNp01_L": group("DNp01", "L"),
        "PSI": group("PSI"), "TTMn": group("TTMn"),
        "TTMn_R": group("TTMn", "R"), "TTMn_L": group("TTMn", "L"),
        "DLMn": np.flatnonzero(typ.str.startswith("DLMn").to_numpy()),
        "LC4": group("LC4"), "LPLC2": group("LPLC2")}
assert all(len(v) for v in READ.values()), "a readout group is empty"

cols = [("none", None)]
for r in RADII: cols.append((f"R disc r{r}", disc("R", r)))
FULL = len(cols)
for r in RADII:
    if r < SCATTER_FROM: continue
    n = len(disc("R", r))
    for s in range(N_SCATTER): cols.append((f"R scatter r{r} s{s}", scattered("R", n, 2000 + s)))
U = np.zeros((N, len(cols)), np.float32)
for j, (_, rows) in enumerate(cols):
    if rows is not None: U[rows, j] = 1.0
print(f"{len(cols)} stimulus columns; readouts " + ", ".join(f"{k} {len(v)}" for k, v in READ.items()), flush=True)


def build(p_, q_, w_, kappa):
    """W with the declared electrical component at strength kappa; the target's other inputs give way to it."""
    val = (G * sign[p_] * w_ / in_tot[q_]).astype(np.float32)
    rows, cols_ = q_, p_
    if kappa > 0:
        val = np.where(np.isin(q_, GAP_ROWS), val * (1 - kappa), val).astype(np.float32)
        rows = np.concatenate([rows, GAP_POST])
        cols_ = np.concatenate([cols_, GAP_PRE])
        val = np.concatenate([val, (G * kappa * GAP_SHARE).astype(np.float32)])
    return sp.csr_matrix((val, (rows, cols_)), shape=(N, N))


def networks():
    yield "baseline k>=3", pre[k3], post[k3], w[k3], len(cols)
    yield "OR 1%", pre[or1], post[or1], w[or1], FULL
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


out = {"rules": __doc__, "radii": RADII, "kappas": KAPPAS, "theta": THETA,
       "declared_pairs": [[str(inst.iloc[a]), str(inst.iloc[b])] for a, b in zip(GAP_PRE, GAP_POST)],
       "results": {}}
for kappa in KAPPAS:
    blk = {}
    for name, p_, q_, w_, ncol in networks():
        t0 = time.time()
        W = build(p_, q_, w_, kappa)
        H, it = solve(W, U[:, :ncol]); del W
        blk[name] = {"resp": {k: (H[v, 1:ncol] - H[v, :1]).max(0).astype(np.float64).tolist()
                              for k, v in READ.items()},
                     "columns": [c[0] for c in cols[1:ncol]], "iters": it, "sec": time.time() - t0}
    out["results"][str(kappa)] = blk
    b = dict(zip(blk["baseline k>=3"]["columns"], blk["baseline k>=3"]["resp"]["TTMn"]))
    dn = dict(zip(blk["baseline k>=3"]["columns"], blk["baseline k>=3"]["resp"]["DNp01_R"]))
    print(f"kappa {kappa}: DNp01_R r18 {dn['R disc r18']:.4f} | TTMn " +
          " ".join(f"r{r}={b[f'R disc r{r}']:.4f}" for r in RADII), flush=True)
    json.dump(out, open("step5_gapjunction.json", "w"), indent=1)

at = lambda k, net, key, col: dict(zip(out["results"][str(k)][net]["columns"],
                                       out["results"][str(k)][net]["resp"][key]))[col]
base, rmax = "baseline k>=3", RADII[-1]

# G0: kappa = 0 must reproduce step 4
s4 = json.load(open("step4_size.json"))
s4at = lambda key, col: dict(zip(s4["results"][base]["columns"], s4["results"][base]["resp"][key]))[col]
g0 = max(abs(at(0.0, base, key, f"R disc r{r}") - s4at(key, f"R disc r{r}"))
         for key in ["DNp01_R", "TTMn"] for r in RADII)

passing = [k for k in KAPPAS if k <= KAPPA_MAX_PASS
           and at(k, base, "TTMn", f"R disc r{rmax}") > THETA and at(k, base, "DLMn", f"R disc r{rmax}") > THETA]
kstar = passing[0] if passing else None
v = {"G0 regression": bool(g0 < 1e-6), "G1 reaches muscle": bool(kstar is not None)}
if kstar is not None:
    ladder = [at(kstar, base, "TTMn", f"R disc r{r}") for r in RADII]
    rho = float(spearmanr(RADII, ladder).statistic)
    shuf = [at(kstar, f"shuffled seed {s}", "TTMn", f"R disc r{rmax}") for s in range(SEEDS)]
    scat_ok, scat = True, {}
    for r in RADII:
        if r < SCATTER_FROM: continue
        s_ = [at(kstar, base, "TTMn", f"R scatter r{r} s{i}") for i in range(N_SCATTER)]
        scat[r] = {"disc": at(kstar, base, "TTMn", f"R disc r{r}"), "scatter_max": max(s_)}
        scat_ok &= scat[r]["disc"] > max(s_)
    v.update({"G2 computation kept": bool(rho >= 0.9), "G3 the wiring did it": bool(ladder[-1] > max(shuf)),
              "G4 contiguity kept": bool(scat_ok),
              "G5 side is right": bool(at(kstar, base, "TTMn_R", f"R disc r{rmax}")
                                       > at(kstar, base, "TTMn_L", f"R disc r{rmax}"))})
    out["stats"] = {"kappa_star": kstar, "spearman": rho, "ladder": ladder, "shuffled": shuf, "scatter": scat}
    print(f"\nsmallest kappa reaching the muscle: {kstar}; TTMn by radius " +
          " ".join(f"r{r} {x:.4f}" for r, x in zip(RADII, ladder)))
    print(f"Spearman(radius, TTMn) = {rho:.3f} | shuffled max {max(shuf):.4f} mean {np.mean(shuf):.4f}")
    for r, s_ in scat.items(): print(f"  r{r}: disc {s_['disc']:.4f} vs scattered max {s_['scatter_max']:.4f}")
else:
    print(f"\nno kappa <= {KAPPA_MAX_PASS} put both TTMn and DLMn above {THETA}; G2-G5 not evaluated")
v["GAP JUNCTIONS WERE THE BOTTLENECK"] = bool(v["G1 reaches muscle"] and v.get("G2 computation kept")
                                               and v.get("G3 the wiring did it"))
out["verdicts"] = v
json.dump(out, open("step5_gapjunction.json", "w"), indent=1)

print(f"\nchain at r{rmax}: " + " | ".join(
    "kappa %.2f: " % k + " ".join(f"{n} {at(k, base, n, f'R disc r{rmax}'):.4f}"
                                  for n in ["DNp01_R", "PSI", "TTMn", "DLMn"]) for k in KAPPAS))
print(f"G0 max deviation from step 4: {g0:.2e}")
for k, val in v.items(): print(f"{k}: {val}")
