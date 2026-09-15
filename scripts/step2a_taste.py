"""Step 2a: can taste quality be read from descending neurons of the UNTRAINED network?

Task (fixed before any result was seen):
- Classes: the four labellar GRN groups of shiu_tasks.json: sugar (LB3b+c, 34), water (LB3a, 17),
  bitter (LB1a-d, 38), ir94e (LB1e, LB2a-c, 32).
- Neuron split, seed 0: per class 30% of bodies (at least 3) are HELD OUT, never stimulated in training.
- One sample: a random 50% (at least 3) of the class's pool, amplitude U(0.5, 1); plus background noise,
  each of the other gustatory sensory neurons on with probability 0.05 at amplitude U(0, 0.5).
- Sets: train 60 per class (train pools, seed 1); test A 40 per class (train pools, new draws, seed 2);
  test B 40 per class (held-out pools only, seed 3). Saved to step2a_taste_task.npz.
Model: compare_steady.py's, h = ReLU(g W h + b + u), g = 0.9, b = 0.1, float32, tol 1e-6.
  Response = h(sample) - h(no stimulus).
Features: responses of the 1,314 descending neurons (primary); all 2,333 readout neurons (secondary).
  A feature is dropped when its largest |response| over the train set is below 1e-4 (solver noise floor)
  or its train variance is 0.
Classifier: multinomial logistic regression on train-standardised features, L2 1e-2, L-BFGS.
Metric: balanced accuracy; chance 0.25.
Networks: baseline k>=3; OR 1%; random baseline edges of OR 1% size (seeds 0..N-1, as score_shiu.py);
  shuffled baseline, presynaptic partners permuted across edges (seeds 1000..), which keeps every neuron's
  in- and out-degree and the synapse counts but not who connects to whom.
Input control: the same classifier on the stimulus vector itself (gustatory neurons as features).

Verdicts (primary features, fixed before running):
  R1 the readout carries taste:        baseline test A >= 0.5
  R2 it generalises to unseen neurons: baseline test B >= 0.5 and above the input control's test B
  R3 the wiring matters:               baseline test B above every shuffled seed
  R4 compaction keeps it:              OR 1% test B >= baseline - 0.05 and above every random seed;
                                       if random seeds also come within 0.05, the task cannot judge compaction.
Run from data/ after retention.py and shiu_tasks.py:  step2a_taste.py [--seeds=N] [--smoke]
--smoke: baseline only, 5 samples per class, prints timing and shapes but no scores.
"""
import json, os, resource, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as tnf
import fpmodel as fm

SEEDS = int(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--seeds=")), 10))
SMOKE = "--smoke" in sys.argv
G, B, TOL, MAXIT, CHUNK = 0.9, 0.1, 1e-6, 2000, 160
CLASSES = ["sugar", "water", "bitter", "ir94e"]
N_TRAIN, N_TEST = (5, 5) if SMOKE else (60, 40)
HOLDOUT, ACTIVE, BG_P, BG_AMP = 0.3, 0.5, 0.05, 0.5
L2, MIN_RESP = 1e-2, 1e-4

T = json.load(open("shiu_tasks.json"))
ann = pd.read_parquet("annotations.parquet", columns=["bodyId", "superclass", "class"])
ann = ann[ann.superclass.notna()].sort_values("bodyId").reset_index(drop=True)
N = len(ann)
assert N == T["N"]
nt = pd.read_parquet("nt.parquet", columns=["body", "consensus_nt"]).set_index("body").consensus_nt
cons = ann.bodyId.map(nt).fillna("missing").astype(str).to_numpy()
sign = np.array([fm.SIGN.get(c, 1) for c in cons], np.float64)

d = np.load("nn_edges.npz"); pre, post, w = d["pre"], d["post"], d["w"].astype(np.float64)
in_tot = np.bincount(post, weights=w, minlength=N)
out_tot = np.bincount(pre, weights=w, minlength=N)
k3 = w >= 3
or1 = k3 & ((w / in_tot[post] >= 0.01) | (w / out_tot[pre] >= 0.01))

sc = ann.superclass.astype(str)
dn = np.flatnonzero(sc.eq("descending_neuron").to_numpy())
readout = np.flatnonzero((sc.eq("descending_neuron") | sc.str.endswith("_motor") | sc.str.contains("efferent")
                          | sc.str.endswith("_endocrine")).to_numpy())
dn_in_readout = np.searchsorted(readout, dn)
gust = np.flatnonzero((ann["class"].astype(str).eq("gustatory") & sc.str.contains("sensory")).to_numpy())


# ---- task
rng = np.random.default_rng(0)
pools = {}
for c in CLASSES:
    r = rng.permutation(np.asarray(T["stimuli"][c]["rows"], np.int64))
    k = max(3, int(round(HOLDOUT * len(r))))
    pools[c] = {"heldout": np.sort(r[:k]), "train": np.sort(r[k:])}
taste = np.concatenate([np.asarray(T["stimuli"][c]["rows"], np.int64) for c in CLASSES])
bg = np.setdiff1d(gust, taste)


def draw(pool, n, seed):
    rng = np.random.default_rng(seed)
    cols, labels = [], []
    for ci, c in enumerate(CLASSES):
        p = pools[c][pool]
        for _ in range(n):
            on = rng.choice(p, max(3, int(round(ACTIVE * len(p)))), replace=False)
            b_on = bg[rng.random(len(bg)) < BG_P]
            rows = np.concatenate([on, b_on])
            vals = np.concatenate([rng.uniform(0.5, 1.0, len(on)), rng.uniform(0.0, BG_AMP, len(b_on))])
            cols.append((rows, vals)); labels.append(ci)
    return cols, np.array(labels)


SETS = {"train": draw("train", N_TRAIN, 1), "test A": draw("train", N_TEST, 2), "test B": draw("heldout", N_TEST, 3)}
np.savez("step2a_taste_task.npz", classes=np.array(CLASSES), bg=bg,
         **{f"pool_{c}_{k}": v for c in CLASSES for k, v in pools[c].items()},
         **{f"{s.replace(' ', '')}_{f}": a for s, (cols, y) in SETS.items() for f, a in [
             ("col", np.concatenate([np.full(len(r), j) for j, (r, _) in enumerate(cols)])),
             ("row", np.concatenate([r for r, _ in cols])), ("val", np.concatenate([v for _, v in cols])), ("y", y)]})


def dense_u(cols):
    U = torch.zeros(N, len(cols), dtype=torch.float32)
    for j, (r, v) in enumerate(cols): U[torch.from_numpy(r), j] = torch.from_numpy(v.astype(np.float32))
    return U


# ---- networks, generated one at a time (each edge mask is 25 MB)
def networks():
    yield "baseline k>=3", pre[k3], post[k3], w[k3]
    if SMOKE: return
    yield "OR 1%", pre[or1], post[or1], w[or1]
    base, n_or = np.flatnonzero(k3), int(or1.sum())
    for seed in range(SEEDS):
        idx = np.random.default_rng(seed).choice(base, n_or, replace=False)
        yield f"random seed {seed}", pre[idx], post[idx], w[idx]
    for seed in range(SEEDS):
        yield f"shuffled seed {seed}", np.random.default_rng(1000 + seed).permutation(pre[k3]), post[k3], w[k3]


def solve(W, U):
    H = torch.zeros_like(U)
    for it in range(1, MAXIT + 1):
        Hn = torch.relu(W @ H + B + U)
        delta = (Hn - H).abs().max().item(); H = Hn
        if delta < TOL: return H, it
    raise RuntimeError(f"no convergence in {MAXIT} iterations (last change {delta:.3g})")


def responses(W):
    h0, it0 = solve(W, torch.zeros(N, 1))
    out, iters = {}, [it0]
    for s, (cols, _) in SETS.items():
        parts = []
        for j in range(0, len(cols), CHUNK):
            H, it = solve(W, dense_u(cols[j:j + CHUNK])); iters.append(it)
            parts.append((H[readout] - h0[readout]).double().numpy().T)
        out[s] = np.concatenate(parts)
    return out, max(iters)


# ---- classifier
def classify(X):
    Xtr, ytr = X["train"], SETS["train"][1]
    sd = Xtr.std(0)
    keep = (np.abs(Xtr).max(0) >= MIN_RESP) & (sd > 0)
    res = {"features_used": int(keep.sum()), "features": int(Xtr.shape[1])}
    if not keep.any():
        return {**res, **{s: {"balanced": 0.25, "confusion": None} for s in ["test A", "test B"]}}
    mu, sd = Xtr[:, keep].mean(0), sd[keep]
    z = lambda A: torch.from_numpy((A[:, keep] - mu) / sd)
    Wc = torch.zeros(int(keep.sum()), len(CLASSES), dtype=torch.float64, requires_grad=True)
    bc = torch.zeros(len(CLASSES), dtype=torch.float64, requires_grad=True)
    Ztr, Ytr = z(Xtr), torch.from_numpy(ytr)
    opt = torch.optim.LBFGS([Wc, bc], max_iter=1000, tolerance_grad=1e-9, tolerance_change=1e-12,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        l = tnf.cross_entropy(Ztr @ Wc + bc, Ytr) + 0.5 * L2 * (Wc ** 2).sum()
        l.backward(); return l
    opt.step(closure)
    for s in ["test A", "test B"]:
        y = SETS[s][1]
        with torch.no_grad(): p = (z(X[s]) @ Wc + bc).argmax(1).numpy()
        conf = np.zeros((len(CLASSES), len(CLASSES)), int)
        np.add.at(conf, (y, p), 1)
        res[s] = {"balanced": float(np.mean(np.diag(conf) / conf.sum(1))), "confusion": conf.tolist()}
    return res


def peak_gb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9   # bytes on macOS


out = {"rules": __doc__, "classes": CLASSES,
       "pools": {c: {k: len(v) for k, v in pools[c].items()} for c in CLASSES}, "background_neurons": int(len(bg)),
       "samples": {s: len(v[0]) for s, v in SETS.items()}, "dn": int(len(dn)), "readout": int(len(readout)),
       "results": {}}
print(f"pools {out['pools']}; background {len(bg)}; samples {out['samples']}; DN {len(dn)}, readout {len(readout)}",
      flush=True)

gidx = {int(r): i for i, r in enumerate(gust)}
Xin = {}
for s, (cols, _) in SETS.items():
    A = np.zeros((len(cols), len(gust)))
    for j, (r, v) in enumerate(cols): A[j, [gidx[int(i)] for i in r]] = v
    Xin[s] = A
out["results"]["input control"] = {"input": classify(Xin)}

for name, p_, q_, w_ in networks():
    t0 = time.time()
    A = sp.csr_matrix(((G * sign[p_] * w_ / in_tot[q_]).astype(np.float32), (q_, p_)), shape=(N, N))
    W = fm._torch_csr(A, torch.float32); del A
    R, it = responses(W); del W
    Xdn = {s: v[:, dn_in_readout] for s, v in R.items()}
    live = float((np.abs(Xdn["train"]).max(0) >= MIN_RESP).mean())
    out["results"][name] = {"DN": classify(Xdn), "readout": classify(R), "max_iters": it,
                            "dn_responsive_share": live, "sec": time.time() - t0}
    r = out["results"][name]
    if SMOKE:
        print(f"{name}: {it} iters, {r['sec']:.0f}s, peak {peak_gb():.2f} GB; features {Xdn['train'].shape} "
              f"DN responsive {live:.3f}; classifier ran (scores withheld in smoke mode)", flush=True)
        sys.exit(0)
    print(f"{name}: DN test A {r['DN']['test A']['balanced']:.3f} test B {r['DN']['test B']['balanced']:.3f} | "
          f"readout A {r['readout']['test A']['balanced']:.3f} B {r['readout']['test B']['balanced']:.3f} | "
          f"DN responsive {live:.3f} | {it} iters {r['sec']:.0f}s peak {peak_gb():.2f} GB", flush=True)
    json.dump(out, open("step2a_taste.json", "w"), indent=1)

R = out["results"]
acc = lambda n, s: R[n]["DN"][s]["balanced"]
rnd = [n for n in R if n.startswith("random seed")]
shf = [n for n in R if n.startswith("shuffled seed")]
bA, bB, oB = acc("baseline k>=3", "test A"), acc("baseline k>=3", "test B"), acc("OR 1%", "test B")
inB = R["input control"]["input"]["test B"]["balanced"]
near = sum(acc(n, "test B") >= bB - 0.05 for n in rnd)
out["verdicts"] = {
    "R1 readout carries taste": bool(bA >= 0.5),
    "R2 generalises to held-out neurons": bool(bB >= 0.5 and bB > inB),
    "R3 wiring matters": bool(bB > max(acc(n, "test B") for n in shf)),
    "R4 compaction keeps it": bool(oB >= bB - 0.05 and oB > max(acc(n, "test B") for n in rnd)),
    "R4 random seeds within 0.05 of baseline": f"{near}/{len(rnd)}",
}
spread = lambda ns, s: {"mean": float(np.mean([acc(n, s) for n in ns])), "min": float(min(acc(n, s) for n in ns)),
                        "max": float(max(acc(n, s) for n in ns))}
out["aggregate"] = {k: {s: spread(ns, s) for s in ["test A", "test B"]} for k, ns in [("random", rnd), ("shuffled", shf)]}
json.dump(out, open("step2a_taste.json", "w"), indent=1)
print(f"\ninput control: test A {R['input control']['input']['test A']['balanced']:.3f} test B {inB:.3f}")
for k, v in out["aggregate"].items():
    print(f"{k} ({len(rnd)} seeds): " + " | ".join(f"{s} mean {x['mean']:.3f} [{x['min']:.3f}, {x['max']:.3f}]" for s, x in v.items()))
for k, v in out["verdicts"].items(): print(f"{k}: {v}")
