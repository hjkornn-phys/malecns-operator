"""Step 7b: does the IR52b routing hold on each side's neurons separately?

Step 7 could not answer this. Its side split used `somaSide`, which is null for every one of these gustatory
neurons, so its Q4 columns were empty and Q4 is recorded there as not evaluable rather than failed. The side
information lives in `rootSide`, which splits cleanly: WG1, WG3 and WG4 are 48/48, LgLG2 is 65/65.

WHAT THIS CANNOT DO, established by a feasibility lookup before the rules were written: `rootSide` is null
for all 1,258 male-specific neurons and all 156 pC1 neurons, so there is no way to ask which SIDE of the
circuitry a channel reaches. Laterality is out of reach with these annotations. What is left is a split-half
replication: drive one side's input neurons, read the whole male-specific population, and ask whether each
half on its own reproduces the effect.

The halves are anatomically distinct input sets, not statistically independent samples: they sit in one
connectome whose two sides are interconnected, so activity from the left neurons reaches right-side
circuitry. A replication here means the effect does not depend on one side's neurons, not that it has been
observed twice over.

Stimuli, all driven whole so nothing is subsampled on the primary test:
    wing (primary)   IR52b WG1 L 48 / R 48, ppk23 WG4 L 48 / R 48, ppk25 WG3 L 48 / R 48
    leg (secondary)  IR52b LgLG2 L 65 / R 65, and the ppk23 and ppk25 leg sets matched to 65 per side
                     over 10 draws (seed 5000..5009), since their leg sets are larger and unequal
Model, solve, share measure and networks as in step7_ir52b.py: baseline k>=3, OR 1%, shuffled seeds
1000..1019, random seeds 0..19. Readout: the 1,258 male-specific neurons, with pC1, fru_high and dsx_high
reported alongside.

Thresholds are rank tests against the 20 shuffled seeds, one-sided p ~ 1/21 ~ 0.048, so no magnitude is
chosen by hand — the same reasoning step 7 recorded.

Verdicts, fixed before running:
  R1 wing, left half    IR52b above every one of the 20 shuffled seeds, driving WG1 L alone
  R2 wing, right half   the same, driving WG1 R alone
  R3 dissociation holds within each wing half, IR52b above both ppk23 and ppk25
  R4 the halves agree   the two halves' shares differ by less than the shuffled spread for IR52b
  R5 leg replication    R1 and R2's pattern on LgLG2 L and LgLG2 R
  SPLIT-HALF REPLICATED = R1 and R2 and R3.
R4 is separate on purpose: two halves can both clear the null while still differing in size, and that is
worth seeing rather than folding into a pass.

Limits: `receptorType` is putative; routing is not behaviour; no time axis and no firing threshold; and the
hypothesis under test was generated from step 6's data, so this is replication on a different input set, not
a clean out-of-sample test.
Run from data/ after step7_ir52b.py:  step7b_sides.py [--seeds=N] [--lookup]
--lookup audits every input and output set and exits without scoring: sizes, missing labels, overlaps,
duplicate bodies, and whether each stimulus actually moves the network. It reports NO share on the readout
under test, since that is the comparison the verdicts are about. It shares the set definitions with the run
itself, so it describes what will actually be driven rather than a re-implementation of it.
"""
import json, sys, time
import numpy as np
import pandas as pd
import scipy.sparse as sp

SEEDS = int(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--seeds=")), 20))
LOOKUP = "--lookup" in sys.argv
G, B, TOL, MAXIT = 0.9, 0.1, 1e-6, 2000
LEG_MATCH, LEG_DRAWS, LEG_SEED0 = 65, 10, 5000
WING = {"IR52b": ["WG1"], "ppk23": ["WG4"], "ppk25": ["WG3"]}
LEG = {"IR52b": ["LgLG2"], "ppk23": ["LgLG1a", "LgLG7"], "ppk25": ["LgLG1b", "LgLG8"]}
CHANNELS = ["IR52b", "ppk23", "ppk25"]

ann = pd.read_parquet("annotations.parquet",
                      columns=["bodyId", "superclass", "type", "rootSide", "dimorphism", "fruDsx"])
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
rs = ann.rootSide.fillna("-").astype(str).to_numpy()
dim = ann.dimorphism.fillna("").astype(str); fd = ann.fruDsx.fillna("").astype(str)
READ = {"male-specific": np.flatnonzero(dim.eq("male-specific").to_numpy()),
        "pC1": np.flatnonzero(typ.str.startswith("pC1").to_numpy()),
        "fru_high": np.flatnonzero(fd.eq("fru_high").to_numpy()),
        "dsx_high": np.flatnonzero(fd.eq("dsx_high").to_numpy())}
side_rows = lambda types, s: np.flatnonzero((typ.isin(types).to_numpy()) & (rs == s))

cols = [("none", None)]
for c in CHANNELS:
    for s in "LR":
        wing = side_rows(WING[c], s)
        cols.append((f"wing {c} {s}", wing))
        leg = side_rows(LEG[c], s)
        if len(leg) == LEG_MATCH:
            cols.append((f"leg {c} {s} d0", leg))
        else:
            for j in range(LEG_DRAWS):
                cols.append((f"leg {c} {s} d{j}",
                             np.random.default_rng(LEG_SEED0 + j).choice(leg, LEG_MATCH, replace=False)))
        print(f"{c} {s}: wing {len(wing)}, leg {len(leg)} -> {LEG_MATCH}", flush=True)
assert all(len(side_rows(WING[c], s)) == 48 for c in CHANNELS for s in "LR"), "wing halves are not 48 each"
U = np.zeros((N, len(cols)), np.float32)
for j, (_, r) in enumerate(cols):
    if r is not None: U[r, j] = 1.0
print(f"{len(cols)} stimulus columns; readouts " +
      ", ".join(f"{k} {len(v)}" for k, v in READ.items()), flush=True)


def solve(W):
    H = np.zeros_like(U)
    for it in range(1, MAXIT + 1):
        Hn = np.maximum(W @ H + B + U, 0.0)
        delta = np.abs(Hn - H).max(); H = Hn
        if delta < TOL: return H, it
    raise RuntimeError(f"no convergence in {MAXIT} iterations (last change {delta:.3g})")


if LOOKUP:
    print("\n--- inputs")
    seen = {}
    for name, r in cols[1:]:
        dup = len(r) - len(set(r.tolist()))
        miss = int((rs[r] == "-").sum())
        print(f"  {name:20s} n={len(r):4d} duplicates={dup} rootSide missing={miss} "
              f"types={sorted(set(typ.to_numpy()[r]))}")
        for other, r2 in seen.items():
            ov = len(set(r.tolist()) & set(r2.tolist()))
            if ov: print(f"      overlaps {other} by {ov}")
        seen[name] = r
    print("\n--- outputs")
    allstim = np.unique(np.concatenate([r for _, r in cols[1:]]))
    for k, v in READ.items():
        print(f"  {k:14s} n={len(v):5d} rootSide missing={int((rs[v] == '-').sum()):5d} "
              f"overlap with any stimulus={len(set(v.tolist()) & set(allstim.tolist()))}")
    print("\n--- feasibility on the baseline network (no readout shares: that is the comparison under test)")
    Wl = sp.csr_matrix(((G * sign[pre[k3]] * w[k3] / in_tot[post[k3]]).astype(np.float32),
                        (post[k3], pre[k3])), shape=(N, N))
    H, it = solve(Wl); del Wl
    R = (H[:, 1:] - H[:, :1]).astype(np.float64)
    print(f"  converged in {it} iterations")
    for j, (name, _) in enumerate(cols[1:]):
        col = R[:, j]
        print(f"  {name:20s} network mean {col.mean():+.3e} | max |r| {np.abs(col).max():.4f} | "
              f"neurons moved >1e-4 {int((np.abs(col) > 1e-4).sum()):6d}")
    sys.exit(0)


def networks():
    yield "baseline k>=3", pre[k3], post[k3], w[k3]
    yield "OR 1%", pre[or1], post[or1], w[or1]
    base, n_or = np.flatnonzero(k3), int(or1.sum())
    for s in range(SEEDS):
        i = np.random.default_rng(s).choice(base, n_or, replace=False)
        yield f"random seed {s}", pre[i], post[i], w[i]
    for s in range(SEEDS):
        yield f"shuffled seed {s}", np.random.default_rng(1000 + s).permutation(pre[k3]), post[k3], w[k3]


out = {"rules": __doc__, "channels": CHANNELS, "columns": [c[0] for c in cols[1:]],
       "readouts": {k: int(len(v)) for k, v in READ.items()}, "results": {}}


def share(net, key, colname):
    return float(out["results"][net]["share"][key][out["columns"].index(colname)])


def leg_share(net, key, c, s):
    names = [x for x in out["columns"] if x.startswith(f"leg {c} {s} d")]
    return float(np.median([share(net, key, x) for x in names]))


for name, p_, q_, w_ in networks():
    t0 = time.time()
    W = sp.csr_matrix(((G * sign[p_] * w_ / in_tot[q_]).astype(np.float32), (q_, p_)), shape=(N, N))
    H, it = solve(W); del W
    R = (H[:, 1:] - H[:, :1]).astype(np.float64)
    whole = R.mean(0)
    out["results"][name] = {"share": {k: (R[v].mean(0) / np.where(whole > 0, whole, np.nan)).tolist()
                                      for k, v in READ.items()}, "iters": it, "sec": time.time() - t0}
    print(f"{name}: male-specific wing " + " ".join(
        f"{c} L{share(name, 'male-specific', f'wing {c} L'):+.3f}/R{share(name, 'male-specific', f'wing {c} R'):+.3f}"
        for c in CHANNELS) + f" | {it} it {time.time() - t0:.0f}s", flush=True)
    json.dump(out, open("step7b_sides.json", "w"), indent=1)

base, MS = "baseline k>=3", "male-specific"
shuf = [f"shuffled seed {s}" for s in range(SEEDS)]
wb = {(c, s): share(base, MS, f"wing {c} {s}") for c in CHANNELS for s in "LR"}
ws = {(c, s): [share(n, MS, f"wing {c} {s}") for n in shuf] for c in CHANNELS for s in "LR"}
lb = {(c, s): leg_share(base, MS, c, s) for c in CHANNELS for s in "LR"}
ls_ = {(c, s): [leg_share(n, MS, c, s) for n in shuf] for c in CHANNELS for s in "LR"}

r1 = wb[("IR52b", "L")] > max(ws[("IR52b", "L")])
r2 = wb[("IR52b", "R")] > max(ws[("IR52b", "R")])
r3 = all(wb[("IR52b", s)] > wb[(c, s)] for s in "LR" for c in ["ppk23", "ppk25"])
spread = max(ws[("IR52b", "L")] + ws[("IR52b", "R")]) - min(ws[("IR52b", "L")] + ws[("IR52b", "R")])
r4 = abs(wb[("IR52b", "L")] - wb[("IR52b", "R")]) < spread
r5 = lb[("IR52b", "L")] > max(ls_[("IR52b", "L")]) and lb[("IR52b", "R")] > max(ls_[("IR52b", "R")])
v = {"R1 wing left half": bool(r1), "R2 wing right half": bool(r2), "R3 dissociation within each half": bool(r3),
     "R4 the halves agree": bool(r4), "R5 leg replication": bool(r5)}
v["SPLIT-HALF REPLICATED"] = bool(r1 and r2 and r3)
out["verdicts"] = v
out["stats"] = {"wing_baseline": {f"{c} {s}": wb[(c, s)] for c in CHANNELS for s in "LR"},
                "wing_shuffled_range": {f"{c} {s}": [min(ws[(c, s)]), max(ws[(c, s)])] for c in CHANNELS for s in "LR"},
                "leg_baseline": {f"{c} {s}": lb[(c, s)] for c in CHANNELS for s in "LR"},
                "leg_shuffled_range": {f"{c} {s}": [min(ls_[(c, s)]), max(ls_[(c, s)])] for c in CHANNELS for s in "LR"},
                "IR52b_half_difference": abs(wb[("IR52b", "L")] - wb[("IR52b", "R")]), "shuffled_spread": spread,
                "other_readouts": {k: {f"{c} {s}": share(base, k, f"wing {c} {s}") for c in CHANNELS for s in "LR"}
                                   for k in READ if k != MS}}
json.dump(out, open("step7b_sides.json", "w"), indent=1)

print(f"\nmale-specific share by half, baseline vs shuffled range ({SEEDS} seeds)")
for c in CHANNELS:
    for s in "LR":
        print(f"  {c:6s} {s} wing {wb[(c, s)]:+.3f} [{min(ws[(c, s)]):+.3f},{max(ws[(c, s)]):+.3f}]"
              f"   leg {lb[(c, s)]:+.3f} [{min(ls_[(c, s)]):+.3f},{max(ls_[(c, s)]):+.3f}]")
print(f"IR52b halves differ by {out['stats']['IR52b_half_difference']:.3f}, shuffled spread {spread:.3f}")
for k, val in v.items(): print(f"{k}: {val}")
