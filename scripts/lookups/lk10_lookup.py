"""Recorded as LK-10 in PREDICTIONS.md. Run lk10_mapping.py first; table arithmetic only, no torch needed.
LK-10: DoOR 2.0 missing entries, for the olfactory step. Table arithmetic only: no network, no task.
(1) largest complete odorant x unit rectangles over the draft mapping's units
(2) masked validation of three imputations, paired, with intervals over masks
(3) is missingness selective? within a systematic panel, are odorants that other studies also chose
    stronger responders in the panel's own measurement than odorants nobody else chose?
"""
import csv, json, os, subprocess, sys
import numpy as np

D = "door"
RAW = "https://raw.githubusercontent.com/ropensci/DoOR.data/v2.0.1/data/"
rng = np.random.default_rng(0)
OUT = {}

m = list(csv.DictReader(open(f"{D}/door_to_malecns_DRAFT.csv")))
units = sorted({r["door_unit"] for r in m if r["door_unit"] and int(r["odorants"]) > 0})
types_of = {u: [r["malecns_type"] for r in m if r["door_unit"] == u] for u in units}
rows = list(csv.reader(open(f"{D}/door_response_matrix.csv"), delimiter=";"))
hdr = rows[0]
odors = [x[0] for x in rows[1:] if x[0] != "SFR"]
M = np.array([[float(x[hdr.index(u) + 1]) if x[hdr.index(u) + 1] not in ("NA", "") else np.nan for u in units]
              for x in rows[1:] if x[0] != "SFR"])
obs = ~np.isnan(M)
OUT["universe"] = {"units": len(units), "malecns_types": sum(len(v) for v in types_of.values()),
                   "odorants_with_any": int(obs.any(1).sum()), "observed_share": round(float(obs.mean()), 3)}
print(OUT["universe"], flush=True)

# ---------------- (1) complete rectangles: greedy peeling from many starts, keep the Pareto frontier
def peel(order_noise):
    R, C = np.ones(len(odors), bool), np.ones(len(units), bool)
    R &= obs.any(1)
    front = []
    while R.any() and C.any():
        sub = obs[np.ix_(R, C)]
        if sub.all():
            front.append((int(C.sum()), int(R.sum()), [units[i] for i in np.flatnonzero(C)]))
            # continue by dropping the unit with fewest odorants to trace the rest of the frontier
            cc = np.flatnonzero(C); C[cc[np.argmin(sub.sum(0) + order_noise[cc] * 1e-3)]] = False
            R = obs[:, C].all(1) if C.any() else R
            continue
        miss_r = (~sub).mean(1); miss_c = (~sub).mean(0)
        rr, cc = np.flatnonzero(R), np.flatnonzero(C)
        # drop whichever line is most incomplete, rows and columns compared on missing share
        if miss_c.max() >= miss_r.max():
            C[cc[np.argmax(miss_c + order_noise[cc] * 1e-3)]] = False
        else:
            R[rr[np.argmax(miss_r)]] = False
    return front

best = {}
for s in range(200):
    for k, n, us in peel(rng.random(len(units))):
        if n > best.get(k, (0,))[0]: best[k] = (n, us)
frontier, top = [], 0
for k in sorted(best, reverse=True):
    if best[k][0] > top:
        top = best[k][0]
        frontier.append({"units": k, "odorants": best[k][0],
                         "malecns_types": sum(len(types_of[u]) for u in best[k][1]),
                         "orns": sum(int(r["orns"]) for r in m if r["door_unit"] in best[k][1]),
                         "unit_list": best[k][1]})
OUT["1_complete_rectangles"] = [{k: v for k, v in f.items() if k != "unit_list"} for f in frontier]
OUT["1_rectangle_units"] = {f"{f['units']}x{f['odorants']}": f["unit_list"] for f in frontier}
for f in OUT["1_complete_rectangles"]: print("rect", f, flush=True)

# ---------------- (2) masked validation on odorants measured on >= 10 units
keep = obs.sum(1) >= 10
X, O = M[keep], obs[keep]

def impute(A, W, how, rank=None):
    if how == "zero": return np.where(W, A, 0.0)
    colmean = np.array([A[W[:, j], j].mean() if W[:, j].any() else 0 for j in range(A.shape[1])])
    Z = np.where(W, A, colmean)
    if how == "unit mean": return Z
    for _ in range(300):                                  # hard-impute: rank-r SVD, observed entries held
        U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        L = (U[:, :rank] * S[:rank]) @ Vt[:rank]
        Zn = np.where(W, A, L)
        if np.abs(Zn - Z).max() < 1e-6: break
        Z = Zn
    return Z

METHODS = [("zero", None), ("unit mean", None), ("low rank r2", 2), ("low rank r4", 4), ("low rank r8", 8)]
err = {n: [] for n, _ in METHODS}
idx = np.argwhere(O)
for rep in range(30):
    pick = idx[rng.random(len(idx)) < 0.1]
    W = O.copy(); W[pick[:, 0], pick[:, 1]] = False
    truth = X[pick[:, 0], pick[:, 1]]
    for n, r in METHODS:
        Z = impute(X, W, n.split(" r")[0] if r else n, r)
        err[n].append(float(np.sqrt(np.mean((Z[pick[:, 0], pick[:, 1]] - truth) ** 2))))
res = {}
for n, _ in METHODS:
    e = np.array(err[n]); res[n] = {"rmse_mean": round(e.mean(), 4), "rmse_2.5_97.5": [round(float(x), 4) for x in np.percentile(e, [2.5, 97.5])]}
base = np.array(err["zero"])
for n, _ in METHODS[1:]:
    d = np.array(err[n]) - base
    res[n]["minus zero, paired over masks: mean, share of masks better"] = [round(d.mean(), 4), round(float((d < 0).mean()), 3)]
res["observed value spread"] = {"sd": round(float(np.nanstd(X)), 4), "share exactly 0": round(float(np.mean(X[O] == 0)), 3)}
OUT["2_masked_validation"] = {"odorants": int(keep.sum()), "units": len(units), "masks": 30, "masked_share": 0.1, **res}
print(json.dumps(OUT["2_masked_validation"], indent=1), flush=True)

# ---------------- (3) selective missingness
for u in units:
    f = f"{D}/{u}.csv"
    if not os.path.exists(f):
        subprocess.run(["curl", "-sfL", "-o", f, RAW + u + ".csv"], check=False)
per = {}
for u in units:
    f = f"{D}/{u}.csv"
    if not os.path.exists(f): continue
    r = list(csv.reader(open(f), delimiter=";"))
    h = r[0]; studies = h[5:]
    for x in r[1:]:
        if x[3] == "SFR": continue
        for s, v in zip(studies, x[6:]):
            if v not in ("NA", ""):
                per.setdefault(s, {}).setdefault(u, {})[x[3]] = float(v)
panels = {s: d for s, d in per.items() if len(d) >= 10}        # a study that measured >= 10 of our units
OUT["3_panels"] = {s: {"units": len(d), "odorants_per_unit_median": int(np.median([len(v) for v in d.values()]))}
                   for s, d in panels.items()}
print(OUT["3_panels"], flush=True)
stats = {}
for s, d in panels.items():
    zs, labels = [], []
    for u, vals in d.items():
        chosen = {o for s2, d2 in per.items() if s2 != s for o in d2.get(u, {})}
        v = np.array(list(vals.values())); k = list(vals.keys())
        if v.std() == 0: continue
        z = (np.abs(v - np.median(v))) / (v.std())                  # response strength within this unit, this study
        zs += list(z); labels += [o in chosen for o in k]
    zs, labels = np.array(zs), np.array(labels)
    if labels.sum() < 10 or (~labels).sum() < 10: continue
    obsd = zs[labels].mean() - zs[~labels].mean()
    null = np.array([zs[p].mean() - zs[~p].mean() for p in (rng.permutation(labels) for _ in range(5000))])
    stats[s] = {"pairs chosen elsewhere / not": [int(labels.sum()), int((~labels).sum())],
                "strength chosen - not (z units)": round(float(obsd), 3),
                "one-sided permutation p": round(float((1 + (null >= obsd).sum()) / 5001), 4)}
OUT["3_selection_bias"] = stats
print(json.dumps(stats, indent=1), flush=True)
json.dump(OUT, open("lk10_lookup.json", "w"), indent=1)
