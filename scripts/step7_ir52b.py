"""Step 7: do IR52b and ppk23 route oppositely into the male-specific circuitry?

HYPOTHESIS, GENERATED FROM DATA, NOT YET EVIDENCE. Step 6's committed rules all but one failed and stay
failed; its numbers were then read exploratorily and showed that its primary readout was the wrong one. On
pC1 (156 neurons) every channel sat inside the shuffled range, but on the 1,258 male-specific neurons two
channels sat well outside it, in opposite directions: IR52b +0.688 against a shuffled range of +0.179 to
+0.298, and ppk23 -0.257 against +0.290 to +0.553. This run tests that pattern.

Literature, which is the one genuinely independent evidence here: Ir52 receptors (Ir52a/b/d) mediate
detection of courtship-stimulating pheromones (Current Biology 2024, "Function and evolution of Ir52
receptors in mate detection"), and Ir52b marks gustatory populations distinct from ppk23 and ppk25, which is
why dual-colour labelling was needed to compare them. A mate-detection channel routing hard into
male-specific circuitry is what that predicts. ppk23 is the opposite case: ppk23 neurons are REQUIRED for
male courtship (J Neurosci 32:4665), so a model sending them away from male-specific neurons contradicts the
literature. That discordance is recorded, not resolved, and is deliberately kept out of the headline verdict.

Lookup before the thresholds, as the README requires:
- the shuffled null on male-specific neurons spans about 0.3; on pC1 it spans +0.043 to +0.375, wider than
  the effects themselves, so male-specific is the primary readout here and pC1 is reported only alongside;
- the effect is 2.3x the shuffled maximum, large enough that no magnitude threshold needs guessing, so the
  rules below are RANK tests: "above all 20 shuffled seeds" is one-sided p ~ 1/21 ~ 0.048 and contains no
  number chosen by hand.

Stimuli. Each receptor channel splits across sensilla, and the wing sets are all exactly 96 bodies, so they
are driven whole and no subsampling confound arises:
    IR52b   leg LgLG2 130            wing WG1 96
    ppk23   leg LgLG1a 136 + LgLG7 21 wing WG4 96
    ppk25   leg LgLG1b 134 + LgLG8 14 wing WG3 96
Leg channels are matched to the smallest (130) over 10 draws (seed 4000..4009). Wing channels also run split
by somaSide for Q4.
Model, solve and response as everywhere in this repo; share of a channel's drive landing on a readout set as
in step6_pheromone.py. Networks: baseline k>=3, OR 1%, shuffled seeds 1000..1019, random seeds 0..19.

Verdicts, fixed before running:
  Q0 calibration  on LEG neurons, IR52b above every shuffled seed and ppk23 below every shuffled seed.
                  This REPLICATES the observation that generated the hypothesis; it is not evidence for it.
  Q1 the test     on WING neurons, IR52b above every one of the 20 shuffled seeds
  Q2 discordance  on WING neurons, ppk23 below every shuffled seed — recorded either way, see above
  Q3 dissociation on WING neurons, IR52b above both ppk23 and ppk25
  Q4 side         Q1's direction holds on left-side and on right-side wing neurons separately
  Q5 compaction   OR 1% keeps Q1's direction
  HYPOTHESIS SUPPORTED = Q1 and Q3 and Q4.

Limits. This is NOT a clean out-of-sample test: step 6 drew its 17 bodies from each whole channel, legs and
wings mixed, so the wing run is a replication on a different sensillum population rather than held-out data.
`receptorType` is putative, a prediction like the transmitter signs. Routing is not behaviour. And nothing
here has a time axis or a firing threshold.
Run from data/:  step7_ir52b.py [--seeds=N]
"""
import json, sys, time
import numpy as np
import pandas as pd
import scipy.sparse as sp

SEEDS = int(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--seeds=")), 20))
G, B, TOL, MAXIT = 0.9, 0.1, 1e-6, 2000
LEG_MATCH, LEG_DRAWS, LEG_SEED0 = 130, 10, 4000
LEG = {"IR52b": ["LgLG2"], "ppk23": ["LgLG1a", "LgLG7"], "ppk25": ["LgLG1b", "LgLG8"]}
WING = {"IR52b": ["WG1"], "ppk23": ["WG4"], "ppk25": ["WG3"]}
CHANNELS = ["IR52b", "ppk23", "ppk25"]

ann = pd.read_parquet("annotations.parquet",
                      columns=["bodyId", "superclass", "class", "type", "subclass", "receptorType",
                               "dimorphism", "fruDsx", "somaSide"])
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
side = ann.somaSide.fillna("?").astype(str)
dim = ann.dimorphism.fillna("").astype(str)
fd = ann.fruDsx.fillna("").astype(str)
rows_of = lambda types: np.flatnonzero(typ.isin(types).to_numpy())

READ = {"male-specific": np.flatnonzero(dim.eq("male-specific").to_numpy()),
        "pC1": np.flatnonzero(typ.str.startswith("pC1").to_numpy()),
        "fru_high": np.flatnonzero(fd.eq("fru_high").to_numpy()),
        "dsx_high": np.flatnonzero(fd.eq("dsx_high").to_numpy())}
print("readouts: " + ", ".join(f"{k} {len(v)}" for k, v in READ.items()), flush=True)

cols = [("none", None)]
for c in CHANNELS:
    wing = rows_of(WING[c])
    cols.append((f"wing {c}", wing))
    for s_ in "LR":
        cols.append((f"wing {c} {s_}", wing[side.to_numpy()[wing] == s_]))
    leg = rows_of(LEG[c])
    for j in range(LEG_DRAWS):
        cols.append((f"leg {c} d{j}",
                     np.random.default_rng(LEG_SEED0 + j).choice(leg, LEG_MATCH, replace=False)))
for c in CHANNELS:
    print(f"{c}: wing {len(rows_of(WING[c]))} "
          f"(L {int((side.to_numpy()[rows_of(WING[c])] == 'L').sum())}/"
          f"R {int((side.to_numpy()[rows_of(WING[c])] == 'R').sum())}), leg {len(rows_of(LEG[c]))}", flush=True)
assert all(len(rows_of(WING[c])) == 96 for c in CHANNELS), "wing sets are no longer 96 bodies each"
U = np.zeros((N, len(cols)), np.float32)
for j, (_, r) in enumerate(cols):
    if r is not None: U[r, j] = 1.0
print(f"{len(cols)} stimulus columns", flush=True)


def solve(W):
    H = np.zeros_like(U)
    for it in range(1, MAXIT + 1):
        Hn = np.maximum(W @ H + B + U, 0.0)
        delta = np.abs(Hn - H).max(); H = Hn
        if delta < TOL: return H, it
    raise RuntimeError(f"no convergence in {MAXIT} iterations (last change {delta:.3g})")


def networks():
    yield "baseline k>=3", pre[k3], post[k3], w[k3]
    yield "OR 1%", pre[or1], post[or1], w[or1]
    base, n_or = np.flatnonzero(k3), int(or1.sum())
    for s in range(SEEDS):
        i = np.random.default_rng(s).choice(base, n_or, replace=False)
        yield f"random seed {s}", pre[i], post[i], w[i]
    for s in range(SEEDS):
        yield f"shuffled seed {s}", np.random.default_rng(1000 + s).permutation(pre[k3]), post[k3], w[k3]


out = {"rules": __doc__, "channels": CHANNELS, "leg_match": LEG_MATCH,
       "readouts": {k: int(len(v)) for k, v in READ.items()},
       "columns": [c[0] for c in cols[1:]], "results": {}}


def share(net, key, colname):
    return float(out["results"][net]["share"][key][out["columns"].index(colname)])


def leg_share(net, key, c):
    return float(np.median([share(net, key, f"leg {c} d{j}") for j in range(LEG_DRAWS)]))


for name, p_, q_, w_ in networks():
    t0 = time.time()
    W = sp.csr_matrix(((G * sign[p_] * w_ / in_tot[q_]).astype(np.float32), (q_, p_)), shape=(N, N))
    H, it = solve(W); del W
    R = (H[:, 1:] - H[:, :1]).astype(np.float64)
    whole = R.mean(0)
    out["results"][name] = {"share": {k: (R[v].mean(0) / np.where(whole > 0, whole, np.nan)).tolist()
                                      for k, v in READ.items()}, "iters": it, "sec": time.time() - t0}
    sh = lambda c: share(name, "male-specific", f"wing {c}")
    print(f"{name}: male-specific wing " + " ".join(f"{c} {sh(c):+.3f}" for c in CHANNELS) +
          f" | {it} it {time.time() - t0:.0f}s", flush=True)
    json.dump(out, open("step7_ir52b.json", "w"), indent=1)


base, MS = "baseline k>=3", "male-specific"
shuf = [f"shuffled seed {s}" for s in range(SEEDS)]
w_base = {c: share(base, MS, f"wing {c}") for c in CHANNELS}
w_shuf = {c: [share(n, MS, f"wing {c}") for n in shuf] for c in CHANNELS}
l_base = {c: leg_share(base, MS, c) for c in CHANNELS}
l_shuf = {c: [leg_share(n, MS, c) for n in shuf] for c in CHANNELS}

q0 = l_base["IR52b"] > max(l_shuf["IR52b"]) and l_base["ppk23"] < min(l_shuf["ppk23"])
q1 = w_base["IR52b"] > max(w_shuf["IR52b"])
q2 = w_base["ppk23"] < min(w_shuf["ppk23"])
q3 = w_base["IR52b"] > w_base["ppk23"] and w_base["IR52b"] > w_base["ppk25"]
q4 = all(share(base, MS, f"wing IR52b {s_}") > max(share(n, MS, f"wing IR52b {s_}") for n in shuf)
         for s_ in "LR")
q5 = share("OR 1%", MS, "wing IR52b") > max(share(f"random seed {s}", MS, "wing IR52b") for s in range(SEEDS))
v = {"Q0 calibration (replication, not evidence)": bool(q0), "Q1 the test": bool(q1),
     "Q2 discordance with the literature": bool(q2), "Q3 dissociation": bool(q3),
     "Q4 side": bool(q4), "Q5 compaction": bool(q5)}
v["HYPOTHESIS SUPPORTED"] = bool(q1 and q3 and q4)
out["verdicts"] = v
out["stats"] = {"wing_baseline": w_base, "wing_shuffled_range": {c: [min(x), max(x)] for c, x in w_shuf.items()},
                "leg_baseline": l_base, "leg_shuffled_range": {c: [min(x), max(x)] for c, x in l_shuf.items()},
                "wing_by_side": {c: {s_: share(base, MS, f"wing {c} {s_}") for s_ in "LR"} for c in CHANNELS},
                "other_readouts": {k: {c: share(base, k, f"wing {c}") for c in CHANNELS} for k in READ}}
json.dump(out, open("step7_ir52b.json", "w"), indent=1)

print(f"\nmale-specific share, baseline vs shuffled range ({SEEDS} seeds)")
for c in CHANNELS:
    print(f"  {c:6s} wing {w_base[c]:+.3f} [{min(w_shuf[c]):+.3f},{max(w_shuf[c]):+.3f}]"
          f"   leg {l_base[c]:+.3f} [{min(l_shuf[c]):+.3f},{max(l_shuf[c]):+.3f}]")
print("  IR52b by side: " + " ".join(f"{s_} {share(base, MS, f'wing IR52b {s_}'):+.3f}" for s_ in "LR"))
print("  other readouts (wing): " + " | ".join(
    f"{k} " + " ".join(f"{c} {share(base, k, f'wing {c}'):+.3f}" for c in CHANNELS) for k in READ if k != MS))
for k, val in v.items(): print(f"{k}: {val}")
