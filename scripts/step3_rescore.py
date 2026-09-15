"""Step 3: does the trained model score better against the real fly than the untrained one?

This closes the loop the project opened. Result 1 scored the UNTRAINED model on 149 experimental outcomes
from Shiu et al. 2024 and could not establish similarity to the fly. Step 2b then trained the gains on a
taste classification task. The 149 outcomes were never part of that training, so they are genuinely held
out, and this run scores the trained parameters on them under the SAME rules.

Model, fpmodel.py's form written out in scipy so a silencing clamp can be applied as score_shiu.py does:
    h = ReLU( gamma[superclass(post)] * (W0 @ (alpha[transmitter(pre)] * h)) + s * u + b )
    W0[post, pre] = sign(pre) * weight / in_tot_full[post]
Parameter sets compared, both on the same networks and tasks:
  untrained  alpha = 0.9, gamma = sigmoid(10), s = 1, b = 0.1 — step 2b's starting point, which is also
             score_shiu.py's g = 0.9, b = 0.1 model, so S0 can check the two agree
  trained    step2b_taste_baseline.json's final_raw, read through the same Params layout (alpha x5,
             gamma per superclass, s, b)
  shuffled-trained  step2b_taste_shuffled.json's final_raw, scored on the REAL network, so that a gain
             learned on scrambled wiring is held against the gain learned on real wiring
Scoring rules are score_shiu.py's, unchanged and not re-tuned: responds = max response > 0.01, required =
silencing lowers the readout by more than 20%, sufficient = MN9 response > 0.01 against an optogenetic PE
rate above 0; aBN2 is skipped for want of a MaleCNS match. The task construction here is a transcription of
score_shiu.py, and S0 exists to catch any divergence between the two.
Networks: baseline k>=3, OR 1%, random baseline edges of OR 1% size (seeds 0..9), as in score_shiu.py.

Verdicts, fixed before running:
  S0 transcription is faithful  untrained parameters reproduce score_shiu.json's g = 0.9 baseline:
                                at least 145 of 149 predictions identical
  S1 training moved the biology trained balanced accuracy differs from untrained by more than 0.01 on the
                                baseline network (either direction; a drop is a result too)
  S2 it moved it the right way  trained balanced accuracy > untrained balanced accuracy
  S3 the wiring earned it       the trained gain beats the shuffled-trained gain on balanced accuracy
  S4 compaction still holds up  with trained parameters, OR 1% changes fewer answers from its own baseline
                                than every one of the 10 random seeds does
  TRAINING IMPROVED BIOLOGICAL SIMILARITY = S2 and S3.
S1 is deliberately two-sided: step 2b's loss moved very little, so the honest prior is that almost nothing
changes here, and "nothing changed" is the result to report if that is what happens, not a failure to fix.

Run from data/ after score_shiu.py and both step2b_taste.py runs:  step3_rescore.py [--seeds=N]
"""
import json, os, sys, time
import numpy as np
import pandas as pd
import scipy.sparse as sp

SEEDS = int(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--seeds=")), 10))
B_DEFAULT, TOL, MAXIT, REQ, THETA = 0.1, 1e-6, 2000, 0.2, 1e-2
NT_CLASSES = ["acetylcholine", "gaba", "glutamate", "histamine", "other"]
SIGN = {"acetylcholine": 1, "dopamine": 1, "serotonin": 1, "octopamine": 1,
        "gaba": -1, "glutamate": -1, "histamine": -1}

T = json.load(open("shiu_tasks.json"))
ann = pd.read_parquet("annotations.parquet")
ann = ann[ann.superclass.notna()].sort_values("bodyId").reset_index(drop=True)
N = len(ann)
assert N == T["N"]
nt = pd.read_parquet("nt.parquet", columns=["body", "consensus_nt"]).set_index("body").consensus_nt
cons = ann.bodyId.map(nt).fillna("missing").astype(str).to_numpy()
sign = np.array([SIGN.get(c, 1) for c in cons], np.float32)
ntc = np.array([NT_CLASSES.index(c) if c in NT_CLASSES[:4] else 4 for c in cons])
sc_cat = pd.Categorical(ann.superclass.astype(str))          # same ordering fpmodel.Net builds
sc_names, sc_code = list(sc_cat.categories), sc_cat.codes.astype(np.int64)

d = np.load("nn_edges.npz"); pre, post, w = d["pre"], d["post"], d["w"].astype(np.float64)
in_tot = np.bincount(post, weights=w, minlength=N)
out_tot = np.bincount(pre, weights=w, minlength=N)
k3 = w >= 3
masks = {"baseline k>=3": k3, "OR 1%": k3 & ((w / in_tot[post] >= 0.01) | (w / out_tot[pre] >= 0.01))}
for s in range(SEEDS):
    m = np.zeros_like(k3)
    m[np.random.default_rng(s).choice(np.flatnonzero(k3), int(masks["OR 1%"].sum()), replace=False)] = True
    masks[f"random seed {s}"] = m

# ---- parameter sets
sigmoid = lambda x: 1 / (1 + np.exp(-x))
softplus = lambda x: np.log1p(np.exp(x))


def unpack(raw):
    raw = np.asarray(raw, np.float64)
    a, c, s_raw, b = raw[:5], raw[5:5 + len(sc_names)], raw[-2], raw[-1]
    assert len(c) == len(sc_names), f"{len(c)} gammas for {len(sc_names)} superclasses"
    return 0.99 * sigmoid(a), sigmoid(c), softplus(s_raw), b


PARAMS = {"untrained": unpack([np.log(10.0)] * 5 + [10.0] * len(sc_names) + [np.log(np.e - 1), B_DEFAULT])}
for tag, f in [("trained", "step2b_taste_baseline.json"), ("shuffled-trained", "step2b_taste_shuffled.json")]:
    if os.path.exists(f): PARAMS[tag] = unpack(json.load(open(f))["final_raw"])
    else: print(f"note: {f} missing, {tag} not scored", flush=True)
for k, (a, g, s, b) in PARAMS.items():
    print(f"{k:16s} alpha {np.round(a, 4)} s {s:.4f} b {b:.4f} gamma[min,med,max] "
          f"{g.min():.4f} {np.median(g):.4f} {g.max():.4f}", flush=True)

# ---- tasks, transcribed from score_shiu.py (S0 checks the transcription)
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

cols = []
def col(stim, sil=None):
    if (stim, sil) not in cols: cols.append((stim, sil))
    return cols.index((stim, sil))


col("none")
for k in ["sugar", "water", "sugar_L", "sugar_R", "sugar+bitter", "sugar+ir94e", "JON_all", "JON_CE", "JON_F"]: col(k)
for n in SEZ: col(f"sez:{n}")
req_sugar = [n for n, e in T["named_neurons"].items()
             if e.get("required_for_sugar_pe") is not None and n != "MN9" and n in NAMED]
req_water = [n for n, e in T["named_neurons"].items()
             if e.get("required_for_water_pe") is not None and n != "MN9" and n in NAMED]
for n in req_sugar: col("sugar", n); col("none", n)
for n in req_water: col("water", n); col("none", n)
col("JON_all", "aBN1"); col("none", "aBN1")
K = len(cols)
U = np.zeros((N, K), np.float32)
for j, (stim, _) in enumerate(cols):
    if stim != "none": U[S[stim], j] = 1.0
sil_cols = [(j, SIL[s]) for j, (_, s) in enumerate(cols) if s is not None]
print(f"{K} columns, {len(sil_cols)} with silencing", flush=True)


def solve(W, s, b):
    H = np.zeros_like(U)
    for it in range(1, MAXIT + 1):
        Hn = np.maximum(W @ H + s * U + b, 0.0)
        for j, r in sil_cols: Hn[r, j] = 0.0
        delta = np.abs(Hn - H).max(); H = Hn
        if delta < TOL: return H, it
    raise RuntimeError(f"no convergence in {MAXIT} iterations (last change {delta:.3g})")


def predictions(H):
    resp = lambda stim, r, sil=None: H[r, col(stim, sil)].astype(np.float64) - H[r, col("none", sil)]
    P = []
    add = lambda task, item, truth, value, kind: P.append(
        dict(task=task, item=item, truth=bool(truth), value=float(value), kind=kind))
    for s_, o in [("L", "R"), ("R", "L")]:
        ipsi, contra = resp(f"sugar_{s_}", [MN9_SIDE[s_]])[0], resp(f"sugar_{s_}", [MN9_SIDE[o]])[0]
        add("1 MN9 ipsi < contra", f"sugar {s_}", True, float(ipsi < contra), "bool")
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
    a_ = resp("JON_all", RD["aDN1"]).sum(); a_sil = resp("JON_all", RD["aDN1"], "aBN1").sum()
    add("9 aBN1 required for JON -> aDN1", "aBN1", True, float(a_sil < (1 - REQ) * a_), "bool")
    for g_ in ["JON_CE", "JON_F"]:
        add("10 aBN1 responds to JO group", g_, True, resp(g_, RD["aBN1"]).max(), "theta")
    return P


pred = lambda p: p["value"] > THETA if p["kind"] == "theta" else bool(p["value"])


def score(P):
    t = np.array([p["truth"] for p in P]); q = np.array([pred(p) for p in P])
    tpr = (q & t).sum() / max(t.sum(), 1); tnr = (~q & ~t).sum() / max((~t).sum(), 1)
    return {"accuracy": float((q == t).mean()), "balanced": float((tpr + tnr) / 2), "answers": q.tolist()}


out = {"rules": __doc__, "theta": THETA, "n_tasks": None, "results": {}}
# untrained: baseline (S0) and OR 1%; trained: every network (S4 needs the random seeds);
# shuffled-trained: baseline only, which is all S3 compares.
WANTED = {"untrained": lambda n: not n.startswith("random"),
          "trained": lambda n: True,
          "shuffled-trained": lambda n: n == "baseline k>=3"}
for pname, (alpha, gamma, s, b) in PARAMS.items():
    for nname, mk in masks.items():
        if not WANTED[pname](nname): continue
        t0 = time.time()
        val = (sign[pre[mk]] * w[mk] / in_tot[post[mk]]).astype(np.float64)
        val = val * alpha[ntc[pre[mk]]] * gamma[sc_code[post[mk]]]
        W = sp.csr_matrix((val.astype(np.float32), (post[mk], pre[mk])), shape=(N, N))
        H, it = solve(W, s, b); del W
        P = predictions(H)
        out["n_tasks"] = len(P)
        out["results"].setdefault(pname, {})[nname] = {**score(P), "iters": it, "sec": time.time() - t0,
                                                       "tasks": [{k: p[k] for k in ("task", "item", "truth")} for p in P]
                                                       if (pname, nname) == ("untrained", "baseline k>=3") else None}
        r = out["results"][pname][nname]
        print(f"{pname:16s} {nname:16s} acc {r['accuracy']:.3f} balanced {r['balanced']:.3f} "
              f"| {it} it {time.time() - t0:.0f}s", flush=True)
        json.dump(out, open("step3_rescore.json", "w"), indent=1)

R = out["results"]
base = "baseline k>=3"
s4 = json.load(open("score_shiu.json"))
ref = [q["value"] > THETA if q["kind"] == "theta" else bool(q["value"])
       for q in s4["results"]["0.9"]["networks"]["baseline k>=3"]]
mine = R["untrained"][base]["answers"]
same = int(sum(a == b_ for a, b_ in zip(ref, mine)))
v = {"S0 transcription is faithful": bool(len(ref) == len(mine) and same >= 145)}
un, tr = R["untrained"][base]["balanced"], R.get("trained", {}).get(base, {}).get("balanced")
sh = R.get("shuffled-trained", {}).get(base, {}).get("balanced")
if tr is not None:
    v["S1 training moved the biology"] = bool(abs(tr - un) > 0.01)
    v["S2 it moved it the right way"] = bool(tr > un)
    changed = lambda net: int(sum(a != b_ for a, b_ in zip(R["trained"][base]["answers"], R["trained"][net]["answers"])))
    rnd = [changed(f"random seed {s}") for s in range(SEEDS)]
    v["S4 compaction still holds up"] = bool(changed("OR 1%") < min(rnd))
    out["stats"] = {"changed_or1": changed("OR 1%"), "changed_random": rnd}
if sh is not None: v["S3 the wiring earned it"] = bool(tr > sh)
v["TRAINING IMPROVED BIOLOGICAL SIMILARITY"] = bool(v.get("S2 it moved it the right way")
                                                     and v.get("S3 the wiring earned it"))
out["verdicts"] = v
out["agreement_with_score_shiu"] = f"{same}/{len(ref)}"
json.dump(out, open("step3_rescore.json", "w"), indent=1)

print(f"\ntranscription agreement with score_shiu.json (g=0.9 baseline): {same}/{len(ref)}")
print(f"balanced accuracy on {out['n_tasks']} outcomes: untrained {un:.3f}" +
      (f" | trained {tr:.3f} ({tr - un:+.3f})" if tr is not None else "") +
      (f" | shuffled-trained {sh:.3f}" if sh is not None else ""))
if "stats" in out:
    print(f"trained model, answers changed from its baseline: OR 1% {out['stats']['changed_or1']} | "
          f"random {min(out['stats']['changed_random'])}-{max(out['stats']['changed_random'])}")
for k, val in v.items(): print(f"{k}: {val}")
