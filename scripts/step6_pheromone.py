"""Step 6: does the male wiring route pheromone channels somewhere different from food channels?

Instinct, in the sense this repo can test it: no training, no time axis, just the question of where the
wiring sends each kind of input. Step 2a showed identity survives to the descending neurons and step 4
showed spatial structure does, so routing is inside what this model class can answer. Whether the fly
actually courts is not.

MaleCNS is the male CNS, so the male-specific courtship circuitry exists here and cannot be asked of FlyWire,
which is female. The inputs are unusually well labelled for once: `receptorType` marks 752 gustatory neurons,
751 of them leg or wing bristle GRNs, as putative ppk23 (269), ppk25 (257) or IR52b (226). The food channels
(sugar 34, water 17, bitter 38) come from shiu_tasks.json and share no bodies with them.

Channels, each driven at u = 1:
  pheromone  ppk23, ppk25, IR52b            by receptorType
  food       sugar, water, bitter           as in score_shiu.py
  anatomy    leg/wing GRNs with NO receptor label (402)
Matched size, fixed: every channel is subsampled to 17 bodies (the size of the smallest, water), 10 seeds
  (3000..3009), and the median over seeds is used. Comparing 269 driven bodies against 17 would let channel
  size decide the answer. Whole-channel responses are reported alongside, descriptively.
Readouts: pC1 (156, courtship command), male-specific neurons (1,258), fru_high (2,611), dsx_high (138),
  and MN9 (2, proboscis motor) as a reverse control.
Measure, share of a channel's drive that lands on a readout set S:
  A(c, S) = mean response over S / mean response over all neurons
  Scale-free by construction, so a channel's overall strength cancels to first order.
Model, solve and networks as elsewhere: h = ReLU(0.9 W h + 0.1 + u), float32, tol 1e-6, response =
  h(channel) - h(none); baseline k>=3, OR 1%, random seeds 0..9, shuffled seeds 1000..1009.

Verdicts, fixed before running:
  P1 pheromone to courtship   on pC1, all three pheromone channels score above all three food channels
  P2 not just anatomy         on pC1, all three pheromone channels score above the unlabelled leg/wing control
  P3 male-specific routing    P1's ordering also holds on the 1,258 male-specific neurons
  P4 the wiring did it        baseline's (pheromone - food) gap on pC1 above every shuffled seed
  P5 reverse control          on MN9 all three FOOD channels score above all three pheromone channels
  P6 compaction keeps it      OR 1% reproduces P1's ordering
  ROUTING REVERSE-ENGINEERED = P1 and P2 and P4 and P5.
P5 validates the method on known biology: food should reach the proboscis motor neuron. If P5 fails, P1 is
not to be believed whatever it says, because the measure itself did not work. P2 is the other load-bearing
one: the pheromone channels sit on legs and wings while the food channels sit on the labellum, so without it
"pheromone" and "leg" cannot be told apart.

Limits, to be read with any result: `receptorType` is putative, a prediction like the transmitter signs;
the sugar/water/bitter assignment rests on morphology (see the Step 2a notes); male-specific and fru/dsx are
dataset annotations, not measurements here; P2 is a partial control, since the unlabelled 402 also sit on
legs and wings; and routing is not behaviour.
Run from data/ after retention.py and shiu_tasks.py:  step6_pheromone.py [--seeds=N]
"""
import json, sys, time
import numpy as np
import pandas as pd
import scipy.sparse as sp

SEEDS = int(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--seeds=")), 10))
G, B, TOL, MAXIT = 0.9, 0.1, 1e-6, 2000
MATCH_N, N_DRAW, DRAW_SEED0 = 17, 10, 3000
PHERO = ["ppk23", "ppk25", "IR52b"]
FOOD = ["sugar", "water", "bitter"]
CONTROL = "leg/wing unlabelled"

T = json.load(open("shiu_tasks.json"))
ann = pd.read_parquet("annotations.parquet",
                      columns=["bodyId", "superclass", "class", "type", "subclass", "receptorType",
                               "dimorphism", "fruDsx"])
ann = ann[ann.superclass.notna()].sort_values("bodyId").reset_index(drop=True)
N = len(ann)
assert N == T["N"]
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

sc = ann.superclass.astype(str); cl = ann["class"].astype(str)
typ = ann.type.fillna("").astype(str); sub = ann.subclass.fillna("").astype(str)
rec = ann.receptorType.fillna("").astype(str)
gust = cl.eq("gustatory") & sc.str.contains("sensory")
legwing = gust & sub.str.contains("leg bristle|wing bristle")

CH = {p: np.flatnonzero(rec.eq("putative_" + p).to_numpy()) for p in PHERO}
for f in FOOD: CH[f] = np.asarray(T["stimuli"][f]["rows"], np.int64)
CH[CONTROL] = np.flatnonzero((legwing & rec.eq("")).to_numpy())
assert not set(np.concatenate([CH[p] for p in PHERO])) & set(np.concatenate([CH[f] for f in FOOD]))
print("channels: " + ", ".join(f"{k} {len(v)}" for k, v in CH.items()), flush=True)

dim = ann.dimorphism.fillna("").astype(str); fd = ann.fruDsx.fillna("").astype(str)
READ = {"pC1": np.flatnonzero(typ.str.startswith("pC1").to_numpy()),
        "male-specific": np.flatnonzero(dim.eq("male-specific").to_numpy()),
        "fru_high": np.flatnonzero(fd.eq("fru_high").to_numpy()),
        "dsx_high": np.flatnonzero(fd.eq("dsx_high").to_numpy()),
        "MN9": np.asarray(T["readouts"]["MN9"]["rows"], np.int64)}
print("readouts: " + ", ".join(f"{k} {len(v)}" for k, v in READ.items()), flush=True)
assert all(len(v) for v in READ.values())

cols = [("none", None)]
for c, rows in CH.items():
    cols.append((f"full {c}", rows))
    for s in range(N_DRAW):
        rng = np.random.default_rng(DRAW_SEED0 + s)
        cols.append((f"match {c} s{s}", rng.choice(rows, MATCH_N, replace=False)))
U = np.zeros((N, len(cols)), np.float32)
for j, (_, rows) in enumerate(cols):
    if rows is not None: U[rows, j] = 1.0
print(f"{len(cols)} stimulus columns (matched size {MATCH_N}, {N_DRAW} draws per channel)", flush=True)


def solve(W, U):
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


out = {"rules": __doc__, "channels": {k: int(len(v)) for k, v in CH.items()},
       "readouts": {k: int(len(v)) for k, v in READ.items()}, "matched_n": MATCH_N, "draws": N_DRAW,
       "results": {}}
for name, p_, q_, w_ in networks():
    t0 = time.time()
    W = sp.csr_matrix(((G * sign[p_] * w_ / in_tot[q_]).astype(np.float32), (q_, p_)), shape=(N, N))
    H, it = solve(W, U); del W
    R = (H[:, 1:] - H[:, :1]).astype(np.float64)
    whole = R.mean(0)
    share = {k: (R[v].mean(0) / np.where(whole > 0, whole, np.nan)).tolist() for k, v in READ.items()}
    out["results"][name] = {"share": share, "columns": [c[0] for c in cols[1:]],
                            "whole_mean": whole.tolist(), "iters": it, "sec": time.time() - t0}
    col = {c: j for j, c in enumerate(out["results"][name]["columns"])}
    med = lambda k, c: float(np.median([share[k][col[f"match {c} s{s}"]] for s in range(N_DRAW)]))
    print(f"{name}: pC1 share " + " ".join(f"{c} {med('pC1', c):.3f}" for c in PHERO + FOOD) +
          f" ctrl {med('pC1', CONTROL):.3f} | MN9 " +
          " ".join(f"{c} {med('MN9', c):.3f}" for c in PHERO + FOOD) + f" | {it} it {time.time() - t0:.0f}s",
          flush=True)
    json.dump(out, open("step6_pheromone.json", "w"), indent=1)


def med(net, key, c):
    r = out["results"][net]
    col = {x: j for j, x in enumerate(r["columns"])}
    return float(np.median([r["share"][key][col[f"match {c} s{s}"]] for s in range(N_DRAW)]))


base = "baseline k>=3"
gap = lambda net, key: (np.mean([med(net, key, c) for c in PHERO])
                        - np.mean([med(net, key, c) for c in FOOD]))
p1 = min(med(base, "pC1", c) for c in PHERO) > max(med(base, "pC1", c) for c in FOOD)
p2 = min(med(base, "pC1", c) for c in PHERO) > med(base, "pC1", CONTROL)
p3 = (min(med(base, "male-specific", c) for c in PHERO)
      > max(med(base, "male-specific", c) for c in FOOD))
shuf = [gap(f"shuffled seed {s}", "pC1") for s in range(SEEDS)]
p4 = gap(base, "pC1") > max(shuf)
p5 = min(med(base, "MN9", c) for c in FOOD) > max(med(base, "MN9", c) for c in PHERO)
p6 = min(med("OR 1%", "pC1", c) for c in PHERO) > max(med("OR 1%", "pC1", c) for c in FOOD)
v = {"P1 pheromone to courtship": bool(p1), "P2 not just anatomy": bool(p2),
     "P3 male-specific routing": bool(p3), "P4 the wiring did it": bool(p4),
     "P5 reverse control": bool(p5), "P6 compaction keeps it": bool(p6)}
v["ROUTING REVERSE-ENGINEERED"] = bool(p1 and p2 and p4 and p5)
out["verdicts"] = v
out["stats"] = {"gap_pC1": gap(base, "pC1"), "shuffled_gap_pC1": shuf,
                "median_share": {k: {c: med(base, k, c) for c in PHERO + FOOD + [CONTROL]} for k in READ}}
json.dump(out, open("step6_pheromone.json", "w"), indent=1)

print(f"\nbaseline median share (matched {MATCH_N} bodies, {N_DRAW} draws)")
print(f"{'readout':14s} " + " ".join(f"{c:>9s}" for c in PHERO + FOOD) + f" {CONTROL:>20s}")
for k in READ:
    print(f"{k:14s} " + " ".join(f"{med(base, k, c):9.3f}" for c in PHERO + FOOD) +
          f" {med(base, k, CONTROL):20.3f}")
print(f"\npC1 pheromone-food gap: baseline {gap(base, 'pC1'):+.3f} | shuffled max {max(shuf):+.3f} "
      f"mean {np.mean(shuf):+.3f} | OR 1% {gap('OR 1%', 'pC1'):+.3f}")
for k, val in v.items(): print(f"{k}: {val}")
