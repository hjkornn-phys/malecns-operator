"""Recorded as LK-11 in PREDICTIONS.md (first pass, stopped by its own SCALE rule). Needs lk10_lookup.json.
LK-11 lookup for step 10 (delayed match on odors). No arm trained, no delay-task accuracy, no odor-identity
readout of any kind. Run from data/ with --extra torch.
"""
import csv, json, math, os, sys, time, resource
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import scipy.sparse as sp
import torch
import fpmodel as fm

HERE = "."; D = "door"
OUT = {}
gb = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
dump = lambda: json.dump(OUT, open(f"{HERE}/lk11_lookup.json", "w"), indent=1)

# ---------------- Hallem 2006 raw panel
units = json.load(open(f"{HERE}/lk10_lookup.json"))["1_rectangle_units"]["23x112"]
mp = {r["door_unit"]: r["malecns_type"] for r in csv.DictReader(open(f"{D}/door_to_malecns_DRAFT.csv"))}
resp, cls, sfr = {}, {}, {}
for u in units:
    r = list(csv.reader(open(f"{D}/{u}.csv"), delimiter=";")); h = r[0]; i = h.index("Hallem.2006.EN") + 1
    for x in r[1:]:
        if x[i] in ("NA", ""): continue
        if x[3] == "SFR": sfr[u] = float(x[i]); continue
        resp.setdefault(x[3], {})[u] = float(x[i]); cls[x[3]] = x[1]
odors = sorted(o for o in resp if len(resp[o]) == len(units))
R = np.array([[resp[o][u] for u in units] for o in odors])            # odors x receptors, spikes/s
assert R.shape == (110, 23), R.shape

# ---------------- splits (seeds fixed in the draft)
rng = np.random.default_rng(101)
held_odor = []
by_cls = {}
for j, o in enumerate(odors): by_cls.setdefault(cls[o], []).append(j)
for c, js in sorted(by_cls.items()):
    js = rng.permutation(js); held_odor += list(js[: int(round(0.3 * len(js)))])
held_odor = np.sort(held_odor); train_odor = np.setdiff1d(np.arange(len(odors)), held_odor)

net = fm.load_network("baseline", torch.float32)
N = net.N
ann = net.ann
typ, sc, klass = ann.type.astype(str), ann.superclass.astype(str), ann["class"].astype(str)
side = ann.get("rootSide")
import pandas as pd
full = pd.read_parquet("annotations.parquet", columns=["bodyId", "rootSide"]).set_index("bodyId").rootSide
sides = ann.bodyId.map(full).fillna("unknown").astype(str).to_numpy()
r102, r103 = np.random.default_rng(102), np.random.default_rng(103)
pools = {}
for u in units:
    rows = np.flatnonzero(typ.eq(mp[u]).to_numpy())
    p = r102.permutation(rows); k = int(round(0.3 * len(p)))
    held, rest = np.sort(p[:k]), r103.permutation(p[k:])
    pools[u] = {"held": held, "h1": np.sort(rest[: len(rest) // 2]), "h2": np.sort(rest[len(rest) // 2:])}
olf = np.flatnonzero(klass.eq("olfactory").to_numpy())
panel = np.concatenate([np.concatenate(list(v.values())) for v in pools.values()])
bg = np.setdiff1d(olf, panel)
OUT["a_mapping"] = {
    "odors": len(odors), "held_out_odors": int(len(held_odor)),
    "held_out_odors_by_class": {c: int(sum(j in set(held_odor) for j in js)) for c, js in sorted(by_cls.items())},
    "classes": {c: len(js) for c, js in sorted(by_cls.items())},
    "panel_orns": int(len(panel)), "background_orns": int(len(bg)),
    "per_type": {mp[u]: {"orns": int(sum(len(v) for v in pools[u].values())),
                         "held/h1/h2": [len(pools[u]["held"]), len(pools[u]["h1"]), len(pools[u]["h2"])],
                         "sides": {s: int((sides[np.concatenate(list(pools[u].values()))] == s).sum()) for s in ("L", "R", "unknown")},
                         "max_response": float(R[:, units.index(u)].max()), "sfr": sfr.get(u)} for u in units},
}
mins = min(min(v["held/h1/h2"]) for v in OUT["a_mapping"]["per_type"].values())
OUT["a_mapping"]["smallest pool after split"] = mins
print(json.dumps({k: v for k, v in OUT["a_mapping"].items() if k != "per_type"}, indent=1), flush=True)
dump()

# ---------------- sample builders
def odor_sample(j, half, scale, bgamp, rng):
    rows, vals = [], []
    for ui, u in enumerate(units):
        p = pools[u][half]; on = p[rng.random(len(p)) < 0.5]
        rows.append(on); vals.append(np.full(len(on), R[j, ui] / scale) * rng.uniform(0.75, 1.25, len(on)))
    b = bg[rng.random(len(bg)) < 0.05]
    rows.append(b); vals.append(rng.uniform(0, bgamp, len(b)))
    return np.concatenate(rows), np.concatenate(vals)

def dense(cols):
    U = torch.zeros(N, len(cols))
    for k, (r, v) in enumerate(cols): U[torch.from_numpy(r.astype(np.int64)), k] = torch.from_numpy(v.astype(np.float32))
    return U

def W_of(pre):
    s_neuron = np.ones(N); s_neuron[net.pre] = np.sign(net.val)
    val = s_neuron[pre] * np.abs(net.val)
    return fm._torch_csr(sp.csr_matrix(((0.9 * val).astype(np.float32), (net.post, pre)), shape=(N, N)), torch.float32)

def solve(W, U, tol=1e-6):
    H = torch.zeros_like(U)
    for it in range(1, 3001):
        Hn = torch.relu(W @ H + 0.1 + U); d = (Hn - H).abs().max().item(); H = Hn
        if d < tol: return H, it
    raise RuntimeError("no convergence")

SETS = {"DN": np.flatnonzero(sc.eq("descending_neuron").to_numpy()),
        "MBON": np.flatnonzero(klass.eq("MBON").to_numpy()),
        "LH output (LHAV/LHAD/LHPV/LHPD types)": np.flatnonzero(typ.str.match(r"^LH(AV|AD|PV|PD)").to_numpy())}
OUT["c_set_sizes"] = {k: int(len(v)) for k, v in SETS.items()}
print(OUT["c_set_sizes"], flush=True)

W = W_of(net.pre)
t = time.time(); h0, _ = solve(W, torch.zeros(N, 1)); t_blank = time.time() - t

# taste reference: step 2a's first 40 train samples
z = np.load("step2a_taste_task.npz")
tcols = [(z["train_row"][z["train_col"] == j], z["train_val"][z["train_col"] == j]) for j in range(0, 240, 6)]
Ht, it_t = solve(W, dense(tcols))
per_sample = lambda H, rows: ((H[rows] - h0[rows]).abs() > 1e-4).sum(0).numpy()
taste_dn = per_sample(Ht, SETS["DN"])
taste_norm = [float(np.linalg.norm(v)) for _, v in tcols]
OUT["b_taste_reference"] = {"median DNs > 1e-4 per sample": float(np.median(taste_dn)),
                            "median input L2 norm": float(np.median(taste_norm)), "iters": it_t}
print(OUT["b_taste_reference"], flush=True)
del Ht

# scale sweep: 40 train-odor samples on h1, bg amplitude 0.02 while sweeping scale
rng = np.random.default_rng(120)
pick = rng.choice(train_odor, 40, replace=False)
scales = {}
for S in (100, 300, 1000):
    r_ = np.random.default_rng(121)
    cols = [odor_sample(j, "h1", S, 0.02, r_) for j in pick]
    t = time.time(); H, it = solve(W, dense(cols)); sec = time.time() - t
    odor_only = [float(np.linalg.norm(v[np.isin(r, panel)])) for r, v in cols]
    scales[S] = {"median DNs > 1e-4 per sample": float(np.median(per_sample(H, SETS["DN"]))),
                 "median panel input L2 norm": float(np.median(odor_only)),
                 "max |u|": float(max(np.abs(v).max() for _, v in cols)), "iters": it, "sec_40": round(sec, 1),
                 **{f"{k}: responsive share over batch": float(((H[v] - h0[v]).abs().max(1).values > 1e-4).float().mean())
                    for k, v in SETS.items()}}
    print(S, scales[S], flush=True)
    del H
OUT["b_scale_sweep"] = scales
ok = [S for S in scales if scales[S]["median DNs > 1e-4 per sample"] >= OUT["b_taste_reference"]["median DNs > 1e-4 per sample"]]
SCALE = max(ok) if ok else None
OUT["b_SCALE_by_rule"] = SCALE
dump()
if SCALE is None:
    print("no candidate scale reaches the taste reference; rule cannot be applied as written"); sys.exit(0)

# background amplitude: largest candidate whose median background norm stays below the median odor's panel norm
med_odor = scales[SCALE]["median panel input L2 norm"]
bgres = {}
for B in (0.5, 0.1, 0.02):
    r_ = np.random.default_rng(122)
    norms = [float(np.linalg.norm(r_.uniform(0, B, int((r_.random(len(bg)) < 0.05).sum())))) for _ in range(200)]
    bgres[B] = float(np.median(norms))
OUT["b_background"] = {"median background L2 norm": bgres, "median odor panel norm at SCALE": med_odor,
                       "BG_by_rule": max([B for B, n in bgres.items() if n < med_odor], default=None)}
BG = OUT["b_background"]["BG_by_rule"]
print(OUT["b_background"], flush=True)
dump()

# V0 batch at SCALE, BG: 160 train-odor samples on h1, seed 130; feature sets
r_ = np.random.default_rng(130)
cols = [odor_sample(j, "h1", SCALE, BG if BG is not None else 0.0, r_) for j in r_.choice(train_odor, 160)]
t = time.time(); H, it = solve(W, dense(cols)); t160 = time.time() - t
feat = {}
for k, v in SETS.items():
    X = (H[v] - h0[v]).T
    keep = (X.abs().max(0).values >= 1e-4) & (X.var(0) > 0)
    feat[k] = {"responsive share (V0 value)": float((X.abs().max(0).values >= 1e-4).float().mean()),
               "features kept": int(keep.sum())}
OUT["c_features_at_SCALE"] = feat
OUT["c_FEATURE_SET_by_rule"] = "DN" if feat["DN"]["features kept"] >= 50 else "decide between MBON and LH output"
OUT["e_cost"] = {"real: 160 samples s": round(t160, 1), "iters": it, "blank s": round(t_blank, 1)}
print(feat, OUT["c_FEATURE_SET_by_rule"], OUT["e_cost"], flush=True)
dump()
del H, W

# shuffled null on the same batch
shuf = {}
for seed in (1000, 1001):
    Ws = W_of(np.random.default_rng(seed).permutation(net.pre))
    hs0, _ = solve(Ws, torch.zeros(N, 1))
    t = time.time(); Hs, it = solve(Ws, dense(cols)); sec = time.time() - t
    shuf[seed] = {"sec_160": round(sec, 1), "iters": it,
                  **{f"{k} responsive share": float(((Hs[v] - hs0[v]).abs().max(1).values >= 1e-4).float().mean()) for k, v in SETS.items()}}
    print(seed, shuf[seed], flush=True)
    del Ws, Hs
OUT["e_shuffled"] = shuf
samples = 800 * 2 + 400 * 2 + 1000 * 2 + 1000 * 2
OUT["e_projection"] = {"encoder samples per network": samples,
                       "real network min": round(samples / 160 * t160 / 60, 1),
                       "one shuffled network min": round(samples / 160 * np.mean([s["sec_160"] for s in shuf.values()]) / 60, 1)}
OUT["peak_GB"] = round(gb(), 2)
dump()
print(json.dumps(OUT["e_projection"]), "peak GB", OUT["peak_GB"])
