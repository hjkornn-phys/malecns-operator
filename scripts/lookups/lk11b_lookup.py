"""Recorded as LK-11 in PREDICTIONS.md (revised pass: SCALE by input norm, LH output readout). Needs lk11_lookup.json.
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
dump = lambda: json.dump(OUT, open(f"{HERE}/lk11b_lookup.json", "w"), indent=1)

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

# ================= LK-11b: revised after LK-11 found no SCALE reaching the DN taste reference
prev = json.load(open(f"{HERE}/lk11_lookup.json"))
OUT["from_LK-11"] = {k: prev[k] for k in ("b_scale_sweep", "b_SCALE_by_rule")}
TASTE_NORM = OUT["b_taste_reference"]["median input L2 norm"]

# SCALE by input norm: the panel input norm is exactly proportional to 1/SCALE for fixed draws
rng =np.random.default_rng(120); pick = rng.choice(train_odor, 40, replace=False)
r_ = np.random.default_rng(121)
norm100 = float(np.median([np.linalg.norm(v[np.isin(r, panel)]) for r, v in (odor_sample(j, "h1", 100, 0.0, r_) for j in pick)]))
SCALE = int(round(100 * norm100 / TASTE_NORM))
OUT["b2_SCALE_by_input_norm"] = {"taste median norm": TASTE_NORM, "odor median panel norm at 100": norm100, "SCALE": SCALE}
print(OUT["b2_SCALE_by_input_norm"], flush=True)

bgres = {}
for B in (0.5, 0.1, 0.02):
    rb = np.random.default_rng(122)
    bgres[B] = float(np.median([np.linalg.norm(rb.uniform(0, B, int((rb.random(len(bg)) < 0.05).sum()))) for _ in range(200)]))
BG = max([B for B, n in bgres.items() if n < TASTE_NORM], default=0.0)
OUT["b2_background"] = {"median background norm": bgres, "BG_by_rule": BG}
print(OUT["b2_background"], flush=True)
dump()

# V0 batch and feature sets, real wiring
r_ = np.random.default_rng(130)
cols = [odor_sample(j, "h1", SCALE, BG, r_) for j in r_.choice(train_odor, 160)]
t = time.time(); H, it = solve(W, dense(cols)); t160 = time.time() - t

def describe(H, h0):
    res = {}
    for k, v in SETS.items():
        X = (H[v] - h0[v]).T
        mx = X.abs().max(0).values
        res[k] = {"size": int(len(v)), "share >= 1e-4 (V0 value)": float((mx >= 1e-4).float().mean()),
                  "n >= 1e-3": int((mx >= 1e-3).sum()), "n >= 1e-2": int((mx >= 1e-2).sum()),
                  "median max |x|": float(mx.median()),
                  "features kept (>=1e-4, var>0)": int(((mx >= 1e-4) & (X.var(0) > 0)).sum())}
    return res

OUT["c2_real"] = {"iters": it, "sec_160": round(t160, 1), **describe(H, h0)}
print("real", json.dumps(OUT["c2_real"]), flush=True)
dump()
del H, W

OUT["c2_shuffled"] = {}
for seed in (1000, 1001):
    Ws = W_of(np.random.default_rng(seed).permutation(net.pre))
    hs0, _ = solve(Ws, torch.zeros(N, 1))
    t = time.time(); Hs, it = solve(Ws, dense(cols)); sec = time.time() - t
    OUT["c2_shuffled"][seed] = {"iters": it, "sec_160": round(sec, 1), **describe(Hs, hs0)}
    print(seed, json.dumps(OUT["c2_shuffled"][seed]), flush=True)
    del Ws, Hs
    dump()

lh = "LH output (LHAV/LHAD/LHPV/LHPD types)"
OUT["c2_FEATURE_SET_by_rule"] = lh if OUT["c2_real"][lh]["n >= 1e-3"] >= 50 else "LH output fails the 50-neuron floor; decide"
samples = 800 * 2 + 400 * 2 + 1000 * 2 + 1000 * 2
sh = np.mean([s["sec_160"] for s in OUT["c2_shuffled"].values()])
OUT["e2_projection"] = {"encoder samples per network": samples, "real min": round(samples / 160 * t160 / 60, 1),
                        "one shuffled min": round(samples / 160 * sh / 60, 1),
                        "real + 5 shuffled min": round(samples / 160 * (t160 + 5 * sh) / 60, 1)}
OUT["peak_GB"] = round(gb(), 2)
dump()
print(OUT["c2_FEATURE_SET_by_rule"], json.dumps(OUT["e2_projection"]), "peak", OUT["peak_GB"])
