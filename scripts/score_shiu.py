"""Score the baseline and compacted networks against Shiu et al. 2024 experimental outcomes (shiu_tasks.json).

Rules, fixed before any result was seen:
- Model as compare_steady.py: h = ReLU(g W h + b + u), b = 0.1, W[post, pre] = sign * w / in_tot_full[post].
  g = 0.9 primary, 0.5 secondary.
- Stimulus: u = 1 on every body of the group. Response = h(stimulus) - h(no stimulus), both under the same
  silencing.
- "responds": max response over the neuron's bodies > THETA (primary 0.01; 1e-4..1e-1 swept).
- "required": clamping the neuron's bodies to h = 0 lowers the readout response by more than 20% (Shiu's rule).
- "sufficient": activating the type gives MN9 (both bodies summed) a response > THETA; truth = PE rate > 0.
- ipsilateral: one-side LB3b+c stimulus; same-side MN9 responds less than opposite-side MN9.
- bitter / Ir94e: adding the group to sugar lowers MN9's response by more than 20% (bitter yes, Ir94e no).
- aBN2 has no MaleCNS match; its silencing task is skipped.
Networks: baseline k>=3; OR 1% (compacted); random baseline edges of OR 1% size (control, seed 0).
Run from data/ after retention.py and shiu_tasks.py.
"""
import json, sys, time
import numpy as np
import pandas as pd
import scipy.sparse as sp

GAINS = [float(x) for x in sys.argv[1:]] or [0.9, 0.5]
B, TOL, MAXIT, REQ = 0.1, 1e-6, 2000, 0.2
THETA, THETAS = 1e-2, [1e-4, 1e-3, 1e-2, 1e-1]
rng = np.random.default_rng(0)

T = json.load(open("shiu_tasks.json"))
ann = pd.read_parquet("annotations.parquet")
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
masks = {"baseline k>=3": k3, "OR 1%": k3 & ((w / in_tot[post] >= 0.01) | (w / out_tot[pre] >= 0.01))}
m = np.zeros_like(k3); m[rng.choice(np.flatnonzero(k3), int(masks["OR 1%"].sum()), replace=False)] = True
masks["random = OR 1% size"] = m

rows = lambda a: np.asarray(a, np.int64)
S = {k: rows(v["rows"]) for k, v in T["stimuli"].items()}
S["JON_all"] = np.unique(np.concatenate([S["JON_CE"], S["JON_F"], S["JON_D_m"]]))
side = ann.rootSide.fillna("?").astype(str).to_numpy()
for s in "LR": S[f"sugar_{s}"] = S["sugar"][side[S["sugar"]] == s]
S["sugar+bitter"] = np.concatenate([S["sugar"], S["bitter"]])
S["sugar+ir94e"] = np.concatenate([S["sugar"], S["ir94e"]])
NAMED = {n: rows(e["rows"]) for n, e in T["named_neurons"].items() if e["rows"]}
SEZ = {t["name"]: rows(t["rows"]) for t in T["sez_activation"] if t["rows"]}
for n, r in SEZ.items(): S[f"sez:{n}"] = r
RD = {k: rows(v["rows"]) for k, v in T["readouts"].items()}
MN9 = RD["MN9"]
inst = ann.instance.fillna("").astype(str).to_numpy()
MN9_SIDE = {inst[i][-1]: i for i in MN9}
SIL = {**NAMED, "aBN1": RD["aBN1"]}

# columns: (stimulus, silenced) pairs
cols = []
def col(stim, sil=None):
    if (stim, sil) not in cols: cols.append((stim, sil))
    return cols.index((stim, sil))

col("none")
for k in ["sugar", "water", "sugar_L", "sugar_R", "sugar+bitter", "sugar+ir94e", "JON_all", "JON_CE", "JON_F"]: col(k)
for n in SEZ: col(f"sez:{n}")
req_sugar = [n for n, e in T["named_neurons"].items() if e.get("required_for_sugar_pe") is not None and n != "MN9" and n in NAMED]
req_water = [n for n, e in T["named_neurons"].items() if e.get("required_for_water_pe") is not None and n != "MN9" and n in NAMED]
for n in req_sugar: col("sugar", n); col("none", n)
for n in req_water: col("water", n); col("none", n)
col("JON_all", "aBN1"); col("none", "aBN1")
K = len(cols)
U = np.zeros((N, K), np.float32)
for j, (stim, _) in enumerate(cols):
    if stim != "none": U[S[stim], j] = 1.0
sil_cols = [(j, SIL[s]) for j, (_, s) in enumerate(cols) if s is not None]
print(f"{K} columns, {len(sil_cols)} with silencing", flush=True)


def solve(W):
    H = np.zeros_like(U)
    for it in range(1, MAXIT + 1):
        Hn = np.maximum(W @ H + B + U, 0.0)
        for j, r in sil_cols: Hn[r, j] = 0.0
        delta = np.abs(Hn - H).max(); H = Hn
        if delta < TOL: return H, it
    return H, -MAXIT


def predictions(H):
    resp = lambda stim, r, sil=None: H[r, col(stim, sil)].astype(np.float64) - H[r, col("none", sil)]
    P = []
    def add(task, item, truth, value, kind):          # kind "theta": pred = value > THETA; else a bool
        P.append(dict(task=task, item=item, truth=bool(truth), value=float(value), kind=kind))
    for s, o in [("L", "R"), ("R", "L")]:
        ipsi, contra = resp(f"sugar_{s}", [MN9_SIDE[s]])[0], resp(f"sugar_{s}", [MN9_SIDE[o]])[0]
        add("1 MN9 ipsi < contra", f"sugar {s}", True, float(ipsi < contra), "bool")
    for n, e in T["named_neurons"].items():
        if n not in NAMED: continue
        for taste in ["sugar", "water"]:
            if e.get(f"responds_to_{taste}") is not None:
                add(f"{2 if taste == 'sugar' else 6} responds to {taste}", n, e[f"responds_to_{taste}"],
                    resp(taste, NAMED[n]).max(), "theta")
    mn9 = lambda stim, sil=None: resp(stim, MN9, sil).sum()
    for taste, names, tid in [("sugar", req_sugar, 3), ("water", req_water, 7)]:
        base = mn9(taste)
        for n in names:
            add(f"{tid} required for {taste} PE", n, T["named_neurons"][n][f"required_for_{taste}_pe"],
                float(mn9(taste, n) < (1 - REQ) * base), "bool")
    for t in T["sez_activation"]:
        if t["name"] in SEZ:
            add("4 sufficient for PE (106 screen)", t["name"], t["opto_pe_rate"] > 0, mn9(f"sez:{t['name']}"), "theta")
    base = mn9("sugar")
    add("5 aversive inhibits sugar MN9", "bitter", True, float(mn9("sugar+bitter") < (1 - REQ) * base), "bool")
    add("5 aversive inhibits sugar MN9", "ir94e", False, float(mn9("sugar+ir94e") < (1 - REQ) * base), "bool")
    for k in ["aBN1", "aDN1", "aDN2"]:
        add("8 responds to JON", k, True, resp("JON_all", RD[k]).max(), "theta")
    a = resp("JON_all", RD["aDN1"]).sum(); a_sil = resp("JON_all", RD["aDN1"], "aBN1").sum()
    add("9 aBN1 required for JON -> aDN1", "aBN1", True, float(a_sil < (1 - REQ) * a), "bool")
    for g in ["JON_CE", "JON_F"]:
        add("10 aBN1 responds to JO group", g, True, resp(g, RD["aBN1"]).max(), "theta")
    return P


def pred(p, theta): return p["value"] > theta if p["kind"] == "theta" else bool(p["value"])


out = {"rules": __doc__, "theta": THETA, "columns": [list(map(str, c)) for c in cols],
       "stimulus_sizes": {k: int(len(v)) for k, v in S.items() if not k.startswith("sez:")}, "results": {}}
for g in GAINS:
    blk = {}
    for name, mk in masks.items():
        t0 = time.time()
        W = sp.csr_matrix(((g * sign[pre[mk]] * w[mk] / in_tot[post[mk]]).astype(np.float32),
                           (post[mk], pre[mk])), shape=(N, N))
        H, it = solve(W)
        blk[name] = predictions(H)
        print(f"g={g} {name}: {it} iters, {time.time() - t0:.0f}s", flush=True)
    out["results"][str(g)] = blk
    json.dump(out, open("score_shiu.json", "w"), indent=1)

    print(f"\n=== g={g}, THETA={THETA}")
    names = list(masks)
    rows_ = []
    for i, p0 in enumerate(blk[names[0]]):
        r = {"task": p0["task"], "item": p0["item"], "truth": p0["truth"]}
        for nm in names: r[nm] = pred(blk[nm][i], THETA)
        rows_.append(r)
    df = pd.DataFrame(rows_)
    summ = df.groupby("task").apply(lambda x: pd.Series({"n": len(x), **{nm: f"{(x[nm] == x.truth).sum()}/{len(x)}" for nm in names}}),
                                    include_groups=False)
    print(summ.to_string())
    for nm in names:
        c = (df[nm] == df.truth)
        sez = df[df.task.str.startswith("4")]
        tpr = ((sez[nm]) & sez.truth).sum() / max(sez.truth.sum(), 1)
        tnr = ((~sez[nm]) & ~sez.truth).sum() / max((~sez.truth).sum(), 1)
        agree = (df[nm] == df[names[0]]).mean()
        sweep = {th: round(float(np.mean([pred(p, th) == p["truth"] for p in blk[nm]])), 3) for th in THETAS}
        print(f"{nm:22s} overall {c.sum()}/{len(c)} = {c.mean():.3f} | screen TPR {tpr:.2f} TNR {tnr:.2f} "
              f"| agrees with baseline {agree:.3f} | accuracy by THETA {sweep}")
    for nm in names[1:]:
        flips = df[df[nm] != df[names[0]]]
        print(f"\n{nm} vs baseline, {len(flips)} changed predictions:")
        if len(flips): print(flips[["task", "item", "truth", names[0], nm]].to_string(index=False))
