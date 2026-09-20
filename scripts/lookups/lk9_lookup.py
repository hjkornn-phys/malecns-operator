"""Recorded as LK-9 in PREDICTIONS.md; written to data/lk9_lookup.json.
LK-9 lookup for step 9 (delayed match). Reads no delay-task accuracy and trains no arm.
Run from ~/repos/malecns-operator/data:  uv run --project .. --extra torch python <this>
"""
import json, math, os, sys, time, resource, itertools
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import scipy.sparse as sp
import torch
import fpmodel as fm

torch.manual_seed(0)
OUT = {}
z = np.load("step2a_taste_task.npz")
CLASSES = [str(c) for c in z["classes"]]
gb = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9

# ---------------- (c) resolution and pool diversity: arithmetic and draws only, no model
pool = {c: {k: len(z[f"pool_{c}_{k}"]) for k in ("train", "heldout")} for c in CLASSES}
act = lambda n: max(3, int(round(0.5 * n)))
div = {c: {k: math.comb(n, act(n)) for k, n in pool[c].items()} for c in CLASSES}
rng = np.random.default_rng(0)
overlap = {}
for c in CLASSES:
    for k in ("train", "heldout"):
        n = pool[c][k]; a = act(n); js, ident = [], 0
        for _ in range(4000):
            x = set(rng.choice(n, a, replace=False)); y = set(rng.choice(n, a, replace=False))
            js.append(len(x & y) / len(x | y)); ident += x == y
        overlap[f"{c}/{k}"] = {"pool": n, "active": a, "distinct_subsets": div[c][k],
                               "mean_jaccard_same_pair": round(float(np.mean(js)), 3),
                               "identical_subset_share": round(ident / 4000, 3)}
res = {}
for p in (0.6, 0.65, 0.7, 0.8):          # smallest n whose 95% lower bound clears 0.5 at accuracy p
    res[f"n for lower bound > 0.5 at acc {p}"] = next(n for n in range(10, 10000) if p - 1.96 * math.sqrt(p * (1 - p) / n) > 0.5)
for q in (0.1, 0.2, 0.3):                # paired gap 0.05 with discordant share q: sign test, 80% power
    # discordant pairs d = n q; right share among them r = (q + 0.05) / (2 q); need z_{.975}+z_{.8}
    r = (q + 0.05) / (2 * q)
    d = ((1.96 * 0.5 + 0.8416 * math.sqrt(r * (1 - r))) / (r - 0.5)) ** 2
    res[f"n for paired gap 0.05, discordant {q}, 80% power"] = int(math.ceil(d / q))
res["binomial SE at 0.5: n=160 / 400 / 800 / 1600"] = [round(0.5 / math.sqrt(n), 4) for n in (160, 400, 800, 1600)]
OUT["c_resolution"] = res
OUT["c_pool_diversity"] = overlap
print(json.dumps(OUT, indent=1), flush=True)

# ---------------- network and samples
net = fm.load_network("baseline", torch.float32)
N = net.N
sc = net.ann.superclass.astype(str)
dn = np.flatnonzero(sc.eq("descending_neuron").to_numpy())
cls = net.ann["class"].astype(str)
gust = np.flatnonzero((cls.eq("gustatory") & sc.str.contains("sensory")).to_numpy())
taste = np.concatenate([z[f"pool_{c}_{k}"] for c in CLASSES for k in ("train", "heldout")])


def U_of(set_, idx):
    U = torch.zeros(N, len(idx))
    col, row, val = z[f"{set_}_col"], z[f"{set_}_row"], z[f"{set_}_val"]
    for j, s in enumerate(idx):
        m = col == s
        U[torch.from_numpy(row[m].astype(np.int64)), j] = torch.from_numpy(val[m].astype(np.float32))
    return U


def W_of(p_, q_, v_, g):
    A = sp.csr_matrix(((g * v_).astype(np.float32), (q_, p_)), shape=(N, N))
    return fm._torch_csr(A, torch.float32)


def iterate(W, U, H=None, tol=1e-6, maxit=3000):
    H = torch.zeros_like(U) if H is None else H.clone()
    for it in range(1, maxit + 1):
        Hn = torch.relu(W @ H + 0.1 + U); d = (Hn - H).abs().max().item(); H = Hn
        if d < tol: return H, it
    raise RuntimeError(f"no convergence, last {d:.3g}")


# ---------------- (a) timing, baseline, batch 160
P = fm.Params(len(net.sc_names), dtype=torch.float32)
with torch.no_grad():
    P.a.fill_(math.log(10)); P.c.fill_(10.0); P.s_raw.fill_(math.log(math.e - 1)); P.b.fill_(0.1)
U = U_of("train", range(160))
t = time.time(); H = fm.solve(U, P, net); fwd = time.time() - t; fit = fm.STATS["fwd_iters"]
Wb = W_of(net.pre, net.post, net.val, 0.9)
h0, _ = iterate(Wb, torch.zeros(N, 1))
H2, _ = iterate(Wb, U)
v0_diff = float((H - H2).abs().max())
Hq = H.detach().requires_grad_(False)
Uq = U.clone(); Hw = fm.solve(Uq, P, net)
v = torch.zeros_like(Hw); v[dn] = torch.randn(len(dn), 160)
t = time.time(); Hg = fm.solve(U, P, net); (Hg * v).sum().backward(); fb = time.time() - t; bit = fm.STATS["bwd_iters"]
# warm start: next frame = blank plus a small memory-like injection on the taste neurons
inj = torch.zeros(N, 160); inj[torch.from_numpy(taste.astype(np.int64))] = 0.1 * torch.rand(len(taste), 160)
t = time.time(); _, it_cold = iterate(Wb, inj); tc = time.time() - t
t = time.time(); _, it_warm = iterate(Wb, inj, H=H2); tw = time.time() - t
OUT["a_timing_batch160"] = {"forward_s": round(fwd, 1), "forward_iters": fit, "forward+backward_s": round(fb, 1),
                            "backward_iters": bit, "blank+injection cold s/iters": [round(tc, 1), it_cold],
                            "warm from previous frame s/iters": [round(tw, 1), it_warm],
                            "fpmodel vs plain iteration max abs diff (V0 feasibility)": v0_diff, "peak_GB": round(gb(), 2)}
print(json.dumps(OUT["a_timing_batch160"], indent=1), flush=True)
del Hg, Hw, v

# ---------------- (d) injection sets: size, and whether a unit injection moves DNs
Wraw = sp.csr_matrix((np.ones(len(net.pre)), (net.post, net.pre)), shape=(N, N))   # post x pre, edge presence
cnt = sp.csr_matrix((np.abs(net.val), (net.post, net.pre)), shape=(N, N))
d = np.load("nn_edges.npz"); syn = sp.csr_matrix((d["w"].astype(np.float64), (d["post"], d["pre"])), shape=(N, N)).tocsc()
from_taste = np.asarray(syn[:, taste].sum(1)).ravel()
issens = sc.str.contains("sensory").to_numpy()
cand = {"taste GRNs (step 2a pools)": taste.astype(np.int64),
        "all gustatory sensory": gust.astype(np.int64)}
for thr in (10, 50):
    cand[f"second-order: >= {thr} synapses from taste GRNs, non-sensory"] = np.flatnonzero((from_taste >= thr) & ~issens)
inj_res = {}
g = torch.Generator().manual_seed(1)
for name, rows in cand.items():
    r = {"size": int(len(rows))}
    for amp in (0.1, 1.0):
        X = torch.zeros(N, 16); X[torch.from_numpy(rows)] = amp * torch.rand(len(rows), 16, generator=g)
        Hx, it = iterate(Wb, X)
        R = (Hx[dn] - h0[dn]).abs()
        r[f"amp {amp}: DNs > 1e-4 (median over 16)"] = int((R > 1e-4).sum(0).median())
        r[f"amp {amp}: max DN response"] = round(float(R.max()), 4)
        r[f"amp {amp}: iters"] = it
    inj_res[name] = r
    print(name, r, flush=True)
OUT["d_injection"] = inj_res

# ---------------- (b) the null: shuffled seed 1000 vs gain, DN responsive share on 160 train samples (no labels)
base = (H2[dn] - h0[dn]).abs()
OUT["b_null"] = {"baseline g0.9 DN responsive share": round(float((base.max(1).values > 1e-4).float().mean()), 3),
                 "baseline median DN max response": round(float(base.max(1).values.median()), 5)}
s_neuron = np.ones(N); s_neuron[net.pre] = np.sign(net.val)
perm = np.random.default_rng(1000).permutation(net.pre)
sv = s_neuron[perm] * np.abs(net.val)
del Wb, H, H2
for gg in (0.9, 0.7, 0.5, 0.3, 0.2):
    Ws = W_of(perm, net.post, sv, gg)
    hs0, _ = iterate(Ws, torch.zeros(N, 1)); Hs, it = iterate(Ws, U)
    R = (Hs[dn] - hs0[dn]).abs()
    OUT["b_null"][f"shuffled 1000 g{gg}"] = {"DN responsive share": round(float((R.max(1).values > 1e-4).float().mean()), 3),
                                              "median DN max response": round(float(R.max(1).values.median()), 5),
                                              "iters": it,
                                              "baseline h0 mean / shuffled h0 mean": [round(float(h0.mean()), 4), round(float(hs0.mean()), 4)]}
    print(gg, OUT["b_null"][f"shuffled 1000 g{gg}"], flush=True)
    del Ws, Hs
OUT["peak_GB"] = round(gb(), 2)
json.dump(OUT, open("lk9_lookup.json", "w"), indent=1)
print(json.dumps(OUT, indent=1))
