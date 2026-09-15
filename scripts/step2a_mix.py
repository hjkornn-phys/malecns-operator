"""Step 2a-mix: a taste task that can judge compaction (step2a_taste.py could not: R4 failed with 9/10 random
seeds within 0.05 of the baseline on 160 test samples, and test A was 1.000 for every network).

Task (fixed before any result was seen):
- Classes, neuron split (seed 0, 30% held out) and background noise exactly as step2a_taste.py.
- One sample is a MIXTURE: a dominant class (the label) and one other class drawn uniformly from the remaining
  three. Dominant: random 50% (at least 3) of its pool at amplitude U(0.5, 1). Other: random 50% (at least 3)
  of its pool at amplitude U(0.5, 1) * r, r ~ U(0.3, 0.8).
- Sets: train 150 per class (train pools, seed 11); test A 100 per class (train pools, seed 12);
  test B 250 per class (held-out pools only, seed 13). Saved to step2a_mix_task.npz.
Model, features (1,314 descending neurons), noise floor 1e-4, classifier (logistic, L2 1e-2, L-BFGS) and
balanced accuracy as step2a_taste.py.
Networks: baseline k>=3; OR 1%; random baseline edges of OR 1% size, seeds 0..9 (as score_shiu.py);
  shuffled baseline, seeds 1000..1002 (sanity only).

Statistics, all on test B, paired over the same samples:
  d(net)        = balanced accuracy(net) - balanced accuracy(baseline), 95% bootstrap CI (2000 resamples, seed 7)
  disagree(net) = share of samples whose predicted label differs from the baseline's prediction, 95% CI
Verdicts (fixed before running):
  M0 the task is informative:     baseline test B in [0.35, 0.95]
  S  wiring still matters:        baseline test B above every shuffled seed
  C1 OR 1% loses no accuracy:     lower CI bound of d(OR 1%) >= -0.05
  C2 OR 1% beats random pruning:  disagree(OR 1%) below every random seed's disagree
  COMPACTION KEPT = M0 and C1 and C2. If M0 fails, C1/C2 are printed but the task decides nothing.
Run from data/ after retention.py and shiu_tasks.py:  step2a_mix.py [--seeds=N] [--smoke]
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
SHUFFLED = 3
SMOKE = "--smoke" in sys.argv
G, B, TOL, MAXIT, CHUNK = 0.9, 0.1, 1e-6, 2000, 160
CLASSES = ["sugar", "water", "bitter", "ir94e"]
N_TRAIN, N_A, N_B = (5, 5, 5) if SMOKE else (150, 100, 250)
HOLDOUT, ACTIVE, BG_P, BG_AMP, R_LO, R_HI = 0.3, 0.5, 0.05, 0.5, 0.3, 0.8
L2, MIN_RESP, BOOT = 1e-2, 1e-4, 2000

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
gust = np.flatnonzero((ann["class"].astype(str).eq("gustatory") & sc.str.contains("sensory")).to_numpy())


# ---- task (pools identical to step2a_taste.py: same seed, same draw order)
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
    cols, labels, others, ratios = [], [], [], []
    for ci, c in enumerate(CLASSES):
        for _ in range(n):
            oi = int(rng.choice([j for j in range(len(CLASSES)) if j != ci]))
            ratio = rng.uniform(R_LO, R_HI)
            pa, pb = pools[c][pool], pools[CLASSES[oi]][pool]
            on_a = rng.choice(pa, max(3, int(round(ACTIVE * len(pa)))), replace=False)
            on_b = rng.choice(pb, max(3, int(round(ACTIVE * len(pb)))), replace=False)
            b_on = bg[rng.random(len(bg)) < BG_P]
            rows = np.concatenate([on_a, on_b, b_on])
            vals = np.concatenate([rng.uniform(0.5, 1.0, len(on_a)), ratio * rng.uniform(0.5, 1.0, len(on_b)),
                                   rng.uniform(0.0, BG_AMP, len(b_on))])
            cols.append((rows, vals)); labels.append(ci); others.append(oi); ratios.append(ratio)
    return cols, np.array(labels), np.array(others), np.array(ratios)


SETS = {"train": draw("train", N_TRAIN, 11), "test A": draw("train", N_A, 12), "test B": draw("heldout", N_B, 13)}
np.savez("step2a_mix_task.npz", classes=np.array(CLASSES), bg=bg,
         **{f"pool_{c}_{k}": v for c in CLASSES for k, v in pools[c].items()},
         **{f"{s.replace(' ', '')}_{f}": a for s, (cols, y, o, rt) in SETS.items() for f, a in [
             ("col", np.concatenate([np.full(len(r), j) for j, (r, _) in enumerate(cols)])),
             ("row", np.concatenate([r for r, _ in cols])), ("val", np.concatenate([v for _, v in cols])),
             ("y", y), ("other", o), ("ratio", rt)]})


def dense_u(cols):
    U = torch.zeros(N, len(cols), dtype=torch.float32)
    for j, (r, v) in enumerate(cols): U[torch.from_numpy(r), j] = torch.from_numpy(v.astype(np.float32))
    return U


def networks():
    yield "baseline k>=3", pre[k3], post[k3], w[k3]
    if SMOKE: return
    yield "OR 1%", pre[or1], post[or1], w[or1]
    base, n_or = np.flatnonzero(k3), int(or1.sum())
    for seed in range(SEEDS):
        idx = np.random.default_rng(seed).choice(base, n_or, replace=False)
        yield f"random seed {seed}", pre[idx], post[idx], w[idx]
    for seed in range(SHUFFLED):
        yield f"shuffled seed {seed}", np.random.default_rng(1000 + seed).permutation(pre[k3]), post[k3], w[k3]


def solve(W, U):
    H = torch.zeros_like(U)
    for it in range(1, MAXIT + 1):
        Hn = torch.relu(W @ H + B + U)
        delta = (Hn - H).abs().max().item(); H = Hn
        if delta < TOL: return H, it
    raise RuntimeError(f"no convergence in {MAXIT} iterations (last change {delta:.3g})")


def dn_responses(W):
    h0, it0 = solve(W, torch.zeros(N, 1))
    out, iters = {}, [it0]
    for s, (cols, *_) in SETS.items():
        parts = []
        for j in range(0, len(cols), CHUNK):
            H, it = solve(W, dense_u(cols[j:j + CHUNK])); iters.append(it)
            parts.append((H[dn] - h0[dn]).double().numpy().T)
        out[s] = np.concatenate(parts)
    return out, max(iters)


def balanced(y, p):
    return float(np.mean([(p[y == c] == c).mean() for c in range(len(CLASSES))]))


def classify(X):
    Xtr, ytr = X["train"], SETS["train"][1]
    sd = Xtr.std(0)
    keep = (np.abs(Xtr).max(0) >= MIN_RESP) & (sd > 0)
    res = {"features_used": int(keep.sum())}
    if not keep.any():
        return {**res, **{s: {"balanced": 0.25, "pred": [0] * len(SETS[s][1])} for s in ["test A", "test B"]}}
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
        with torch.no_grad(): p = (z(X[s]) @ Wc + bc).argmax(1).numpy()
        res[s] = {"balanced": balanced(SETS[s][1], p), "pred": p.tolist()}
    return res


def peak_gb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9   # bytes on macOS


out = {"rules": __doc__, "classes": CLASSES,
       "pools": {c: {k: len(v) for k, v in pools[c].items()} for c in CLASSES}, "background_neurons": int(len(bg)),
       "samples": {s: len(v[0]) for s, v in SETS.items()}, "dn": int(len(dn)), "results": {}}
print(f"pools {out['pools']}; background {len(bg)}; samples {out['samples']}; DN {len(dn)}", flush=True)

for name, p_, q_, w_ in networks():
    t0 = time.time()
    A = sp.csr_matrix(((G * sign[p_] * w_ / in_tot[q_]).astype(np.float32), (q_, p_)), shape=(N, N))
    W = fm._torch_csr(A, torch.float32); del A
    X, it = dn_responses(W); del W
    live = float((np.abs(X["train"]).max(0) >= MIN_RESP).mean())
    out["results"][name] = r = {**classify(X), "max_iters": it, "dn_responsive_share": live, "sec": time.time() - t0}
    if SMOKE:
        print(f"{name}: {it} iters, {r['sec']:.0f}s, peak {peak_gb():.2f} GB; features {X['train'].shape} "
              f"DN responsive {live:.3f}; classifier ran (scores withheld in smoke mode)", flush=True)
        sys.exit(0)
    print(f"{name}: test A {r['test A']['balanced']:.3f} test B {r['test B']['balanced']:.3f} | "
          f"DN responsive {live:.3f} | {it} iters {r['sec']:.0f}s peak {peak_gb():.2f} GB", flush=True)
    json.dump(out, open("step2a_mix.json", "w"), indent=1)

# ---- paired statistics on test B
R = out["results"]
yB, ratioB = SETS["test B"][1], SETS["test B"][3]
pb = np.array(R["baseline k>=3"]["test B"]["pred"])
boot = np.random.default_rng(7).integers(0, len(yB), (BOOT, len(yB)))
ci = lambda v: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]
for name, r in R.items():
    p = np.array(r["test B"]["pred"])
    d_b = [balanced(yB[i], p[i]) - balanced(yB[i], pb[i]) for i in boot]
    dis = p != pb
    r["d"] = {"value": r["test B"]["balanced"] - R["baseline k>=3"]["test B"]["balanced"], "ci": ci(d_b)}
    r["disagree"] = {"value": float(dis.mean()), "ci": ci(dis[boot].mean(1))}
    r["test B by ratio"] = {f"r<0.55": balanced(yB[ratioB < 0.55], p[ratioB < 0.55]),
                            f"r>=0.55": balanced(yB[ratioB >= 0.55], p[ratioB >= 0.55])}
rnd = [n for n in R if n.startswith("random seed")]
shf = [n for n in R if n.startswith("shuffled seed")]
bB, o = R["baseline k>=3"]["test B"]["balanced"], R["OR 1%"]
v = {"M0 task is informative": bool(0.35 <= bB <= 0.95),
     "S wiring still matters": bool(bB > max(R[n]["test B"]["balanced"] for n in shf)),
     "C1 OR 1% loses no accuracy": bool(o["d"]["ci"][0] >= -0.05),
     "C2 OR 1% beats random pruning": bool(o["disagree"]["value"] < min(R[n]["disagree"]["value"] for n in rnd))}
v["COMPACTION KEPT"] = bool(v["M0 task is informative"] and v["C1 OR 1% loses no accuracy"]
                            and v["C2 OR 1% beats random pruning"])
out["verdicts"] = v
json.dump(out, open("step2a_mix.json", "w"), indent=1)

print(f"\n{'network':18s} {'test A':>7s} {'test B':>7s} {'d vs baseline [95% CI]':>28s} {'disagree [95% CI]':>26s}")
for name, r in R.items():
    print(f"{name:18s} {r['test A']['balanced']:7.3f} {r['test B']['balanced']:7.3f} "
          f"{r['d']['value']:+8.3f} [{r['d']['ci'][0]:+.3f}, {r['d']['ci'][1]:+.3f}] "
          f"{r['disagree']['value']:9.3f} [{r['disagree']['ci'][0]:.3f}, {r['disagree']['ci'][1]:.3f}]")
dis_r = [R[n]["disagree"]["value"] for n in rnd]
print(f"random ({len(rnd)} seeds) disagree mean {np.mean(dis_r):.3f} [{min(dis_r):.3f}, {max(dis_r):.3f}]; "
      f"test B mean {np.mean([R[n]['test B']['balanced'] for n in rnd]):.3f}")
for k, val in v.items(): print(f"{k}: {val}")
