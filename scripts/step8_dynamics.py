"""Step 8: add a time axis to the escape pathway and ask what the wiring alone does with it.

Nothing is trained. The network is run forward only, so there is no stored activation and no BPTT.

Model, Euler on a leaky rate unit, one lambda = dt / tau per cell TYPE:
    h <- h + lam * (-h + ReLU(g W h + b + u(t))),    W[post, pre] = sign(pre) * w / in_tot[post]
At lam = 1 this is exactly step 4's fixed-point iteration, which is what D0 checks.
tau heterogeneity is a spread sigma over types: m_t = exp(sigma * z_t), z_t ~ N(0,1) at a fixed seed,
centred so mean log m = 0, and lam_i = LAM0 / m_type(i) with LAM0 chosen so max lam <= 1 (stability).
Which type is fast is never chosen by hand: the assignment is random at a seed, and a result that needs
a particular hand-made assignment is not a wiring result.

Stimulus, the same retinotopic mapping step 4 used. A dark disc drives L2 at u = 1 over the 892-column
hex lattice of one eye, right eye, centred on that eye's centroid. Radii 2, 4, 6, 8, 11, 14, 18 hex units.
  static r        the disc held at radius r for all T frames         -- the D0 regression and the size arm
  loom s          radius walks r2 -> r18 over s frames, then holds   -- the velocity arm
  recede s        the same schedule reversed, r18 -> r2, then holds  -- the arithmetic control
Loom speeds s = 20, 40, 60, 90, 130 frames. Every column runs the same T = 200 frames.
Two responses are recorded per column and readout group, both against the blank column at the matching
frame and both taken as the max over the group's bodies: PEAK, the largest over frames, which is what a
transient event is read by, and FINAL, the value at the last frame, which is the steady state step 4
measured. D0 compares FINAL, because a transient may legitimately overshoot the fixed point and D0 is a
check that the two models are the same model, not that the transient is flat. Everything else reads PEAK.

Readouts: DNp01_R, DNp01_L, LC4, LPLC2, TTMn, PSI. Networks: baseline k>=3 and shuffled seeds 1000..1004
(presynaptic partners permuted, degrees and counts kept), which is the null every rank test below uses.

WHY THERE ARE NO ABSOLUTE THRESHOLDS HERE. The lookup recorded as LK-8 in PREDICTIONS.md
measured step 4's own readouts against its shuffled seeds and found: LPLC2's response level never clears
its own shuffled null at any radius above r2 (shuffled max 0.0395 against baseline 0.0198 at r18); LC4
saturates by r6 (0.0461, 0.0470, 0.0471, 0.0471, 0.0477); and DNp01_R's entire range is 0.0002 to 0.0061
with DNp01_L, PSI and TTMn at or below 0.0002. A rule asking any of these to exceed a fixed number would
be unreachable before it was run, which is step 4's E1 mistake. So every verdict below is a rank test
against the shuffled seeds or an ordering, per METHOD.md.

Verdicts, fixed before running:
  D0 same model           at lam = 1, static columns reproduce step4_size.json's baseline DNp01_R, LC4 and
                          LPLC2 ladders to max abs difference < 1e-4. If this fails nothing below is
                          interpretable and the run stops.
  D1 asymmetry is wiring  at uniform tau, |loom - recede| at DNp01_R, at the slowest speed, is above every
                          shuffled seed. Expected to FAIL: a causal leak is not time-reversal symmetric and
                          path lengths already differ, so some asymmetry is arithmetic. D1 is what separates
                          the arithmetic from the wiring, which is why it is a rank test and not "is there
                          any asymmetry".
  D2 velocity tuned       at uniform tau, Spearman of loom speed (as 1/s, faster = larger) with DNp01_R
                          response, over the five speeds at the same final radius r18, is <= -0.9 or >= 0.9.
  D3 velocity is wiring   at uniform tau, |Spearman| of D2 for baseline is above every shuffled seed.
  D4 size arm survives    at uniform tau, Spearman of radius with DNp01_R over the seven static columns
                          is >= 0.9, as step 4's E2 found 1.000 without a time axis.
  D5 Gaussian size        LPLC2's static response falls from its maximum by r18, i.e. the peak is interior.
                          Expected to FAIL: step 4 found a saturating rise and a leak adds no falling flank.
  D6 how much tau         the smallest sigma in 0.25, 0.5, 1.0, 1.5, 2.0 at which D2 and D3 both hold for
                          baseline, over tau seeds 0 and 1. Reported as that sigma, or as NONE IN RANGE,
                          which is a result and is reported as one.
  WIRING CARRIES VELOCITY = D2 and D3 at uniform tau.

Run from data/ after retention.py:   step8_dynamics.py [--quick]
"""
import json, sys, time, resource
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr

QUICK = "--quick" in sys.argv[1:]
G, B = 0.9, 0.1
RADII = [2, 4, 6, 8, 11, 14, 18]
SPEEDS = [20, 40, 60, 90, 130]
T = 200 if not QUICK else 60
SHUF = [1000, 1001, 1002, 1003, 1004] if not QUICK else [1000]
SIGMAS = [0.25, 0.5, 1.0, 1.5, 2.0] if not QUICK else [1.0]
TAU_SEEDS = [0, 1] if not QUICK else [0]
D0_TOL = 1e-4

ann = pd.read_parquet("annotations.parquet",
                      columns=["bodyId", "superclass", "type", "instance", "somaSide",
                               "assignedOlHex1", "assignedOlHex2"])
ann = ann[ann.superclass.notna()].sort_values("bodyId").reset_index(drop=True)
N = len(ann)
nt = pd.read_parquet("nt.parquet", columns=["body", "consensus_nt"]).set_index("body").consensus_nt
cons = ann.bodyId.map(nt).fillna("missing").astype(str).to_numpy()
SIGN = {"acetylcholine": 1, "dopamine": 1, "serotonin": 1, "octopamine": 1,
        "gaba": -1, "glutamate": -1, "histamine": -1}
sign = np.array([SIGN.get(c, 1) for c in cons], np.float32)

d = np.load("nn_edges.npz"); pre, post, w = d["pre"], d["post"], d["w"].astype(np.float64)
in_tot = np.bincount(post, weights=w, minlength=N)
k3 = w >= 3

typ = ann.type.fillna("").astype(str)
inst = ann.instance.fillna("").astype(str)
side = ann.somaSide.fillna("?").astype(str)

# ---- retinotopic input, step 4's mapping, right eye
l2 = typ.eq("L2") & ann.assignedOlHex1.notna()
idx = np.flatnonzero((l2 & side.eq("R")).to_numpy())
h1, h2 = ann.assignedOlHex1.to_numpy()[idx], ann.assignedOlHex2.to_numpy()[idx]
xy = np.stack([h1 + h2 / 2, h2 * np.sqrt(3) / 2], 1)
centre = xy.mean(0)
dist = np.linalg.norm(xy - centre, axis=1)
DISC = {r: idx[dist <= r] for r in RADII}
print(f"right eye: {len(idx)} L2 columns, disc sizes " +
      ", ".join(f"r{r}={len(DISC[r])}" for r in RADII), flush=True)

def group(name, want_side=None):
    m = typ.eq(name)
    if want_side: m &= inst.str.endswith("_" + want_side)
    return np.flatnonzero(m.to_numpy())

READ = {"DNp01_R": group("DNp01", "R"), "DNp01_L": group("DNp01", "L"),
        "LC4": group("LC4"), "LPLC2": group("LPLC2"),
        "TTMn": group("TTMn"), "PSI": group("PSI")}
assert all(len(v) for v in READ.values()), "a readout group is empty"
print("readouts: " + ", ".join(f"{k} {len(v)}" for k, v in READ.items()), flush=True)

# ---- stimulus schedules: for each column, the radius shown at each frame (None = blank)
def sched_static(r):  return [r] * T
def sched_loom(s, rev=False):
    lad = RADII[::-1] if rev else RADII
    out = []
    for f in range(T):
        k = min(int(f * len(lad) / s), len(lad) - 1) if s > 0 else len(lad) - 1
        out.append(lad[k])
    return out

COLS = [("blank", [None] * T)]
COLS += [(f"static r{r}", sched_static(r)) for r in RADII]
COLS += [(f"loom s{s}", sched_loom(s)) for s in SPEEDS]
COLS += [(f"recede s{s}", sched_loom(s, rev=True)) for s in SPEEDS]
NC = len(COLS)
print(f"{NC} columns x {T} frames", flush=True)

# precompute the input frame stack lazily: rows per (column, frame)
FRAME_ROWS = [[(None if r is None else DISC[r]) for r in sch] for _, sch in COLS]

# ---- tau per cell type
types = pd.Categorical(typ)
tcode = types.codes.astype(np.int64)
NT_TYPES = len(types.categories)

def lam_vector(sigma, seed):
    if sigma == 0.0:
        return np.ones(N, np.float32), 1.0
    z = np.random.default_rng(seed).standard_normal(NT_TYPES)
    z -= z.mean()
    m = np.exp(sigma * z)
    lam0 = 1.0 / m.max()          # stability: every lam <= 1
    return (lam0 / m[tcode]).astype(np.float32), float(lam0)

# ---- forward run
def run(W, lam):
    H = np.zeros((N, NC), np.float32)
    peak = np.zeros((N, NC), np.float32)
    lam_c = lam[:, None]
    for f in range(T):
        U = np.zeros((N, NC), np.float32)
        for j in range(NC):
            rows = FRAME_ROWS[j][f]
            if rows is not None: U[rows, j] = 1.0
        H = H + lam_c * (np.maximum(W @ H + B + U, 0.0) - H)
        resp = H - H[:, :1]                        # against the blank column, same frame
        np.maximum(peak, resp, out=peak)
    return peak, resp

def resp_table(peak, final):
    return ({k: peak[v, :].max(0).astype(np.float64).tolist() for k, v in READ.items()},
            {k: final[v, :].max(0).astype(np.float64).tolist() for k, v in READ.items()})

def build(p_, q_, w_):
    return sp.csr_matrix(((G * sign[p_] * w_ / in_tot[q_]).astype(np.float32), (q_, p_)), shape=(N, N))

NETS = {"baseline k>=3": (pre[k3], post[k3], w[k3])}
for s in SHUF:
    NETS[f"shuffled seed {s}"] = (np.random.default_rng(s).permutation(pre[k3]), post[k3], w[k3])

out = {"rules": __doc__, "radii": RADII, "speeds": SPEEDS, "T": T,
       "columns": [c[0] for c in COLS], "sigmas": SIGMAS, "tau_seeds": TAU_SEEDS,
       "readout_sizes": {k: int(len(v)) for k, v in READ.items()},
       "disc_columns": {r: int(len(DISC[r])) for r in RADII}, "results": {}}

def record(tag, name, peak, final, t0):
    rp, rf = resp_table(peak, final)
    out["results"].setdefault(tag, {})[name] = {"resp": rp, "final": rf, "sec": time.time() - t0}
    json.dump(out, open("step8_dynamics.json", "w"), indent=1)

def at(tag, net, key, col, which="resp"):
    return dict(zip(out["columns"], out["results"][tag][net][which][key]))[col]

# ---- uniform tau, every network
lam1, _ = lam_vector(0.0, 0)
for name, (p_, q_, w_) in NETS.items():
    t0 = time.time(); W = build(p_, q_, w_)
    peak, final = run(W, lam1); del W
    record("uniform", name, peak, final, t0)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    print(f"uniform | {name}: DNp01_R r18 {at('uniform', name, 'DNp01_R', 'static r18'):.4f} "
          f"loom s20 {at('uniform', name, 'DNp01_R', 'loom s20'):.4f} "
          f"recede s20 {at('uniform', name, 'DNp01_R', 'recede s20'):.4f} "
          f"| {time.time() - t0:.0f}s peak {rss:.2f} GB", flush=True)

# ---- D0 regression against step 4
try:
    s4 = json.load(open("step4_size.json"))
    b4 = s4["results"]["baseline k>=3"]
    c4 = dict((c, i) for i, c in enumerate(b4["columns"]))
    diffs = {}
    for key in ["DNp01_R", "LC4", "LPLC2"]:
        a_ = [b4["resp"][key][c4[f"R disc r{r}"]] for r in RADII]
        b_ = [at("uniform", "baseline k>=3", key, f"static r{r}", "final") for r in RADII]
        diffs[key] = {"step4": a_, "step8": b_, "max_abs": float(np.abs(np.array(a_) - np.array(b_)).max())}
    d0 = all(v["max_abs"] < D0_TOL for v in diffs.values())
except FileNotFoundError:
    diffs, d0 = {"error": "step4_size.json missing"}, False
out["D0_detail"] = diffs
print(f"\nD0 same model: {d0}  " +
      " ".join(f"{k} max|d|={v['max_abs']:.2e}" for k, v in diffs.items() if isinstance(v, dict)), flush=True)
json.dump(out, open("step8_dynamics.json", "w"), indent=1)
if not d0:
    out["verdicts"] = {"D0 same model": False, "STOPPED": "D0 failed; nothing below is interpretable"}
    json.dump(out, open("step8_dynamics.json", "w"), indent=1)
    print("D0 failed. Stopping, as the rules say.", flush=True)
    sys.exit(0)

# ---- D1..D5 at uniform tau
def asym(tag, net):
    return abs(at(tag, net, "DNp01_R", f"loom s{SPEEDS[-1]}") - at(tag, net, "DNp01_R", f"recede s{SPEEDS[-1]}"))

def vel_rho(tag, net):
    x = [1.0 / s for s in SPEEDS]
    y = [at(tag, net, "DNp01_R", f"loom s{s}") for s in SPEEDS]
    r = spearmanr(x, y).statistic
    return 0.0 if np.isnan(r) else float(r)

shuf_names = [f"shuffled seed {s}" for s in SHUF]
a_base, a_shuf = asym("uniform", "baseline k>=3"), [asym("uniform", n) for n in shuf_names]
v_base, v_shuf = vel_rho("uniform", "baseline k>=3"), [abs(vel_rho("uniform", n)) for n in shuf_names]
ladder = [at("uniform", "baseline k>=3", "DNp01_R", f"static r{r}") for r in RADII]
rho_size = float(spearmanr(RADII, ladder).statistic)
lp = [at("uniform", "baseline k>=3", "LPLC2", f"static r{r}") for r in RADII]

V = {"D0 same model": bool(d0),
     "D1 asymmetry is wiring": bool(a_base > max(a_shuf)),
     "D2 velocity tuned": bool(abs(v_base) >= 0.9),
     "D3 velocity is wiring": bool(abs(v_base) > max(v_shuf)),
     "D4 size arm survives": bool(rho_size >= 0.9),
     "D5 Gaussian size": bool(max(lp) > lp[-1])}
V["WIRING CARRIES VELOCITY"] = bool(V["D2 velocity tuned"] and V["D3 velocity is wiring"])
out["stats"] = {"asym_baseline": a_base, "asym_shuffled": a_shuf,
                "vel_rho_baseline": v_base, "vel_rho_shuffled_abs": v_shuf,
                "size_ladder": ladder, "size_spearman": rho_size, "lplc2_ladder": lp}
out["verdicts"] = V
json.dump(out, open("step8_dynamics.json", "w"), indent=1)
for k, val in V.items(): print(f"{k}: {val}", flush=True)

# ---- D6 the tau sweep
sweep = {}
D6 = "NONE IN RANGE"
for sigma in SIGMAS:
    for seed in TAU_SEEDS:
        tag = f"sigma{sigma}_seed{seed}"
        lam, lam0 = lam_vector(sigma, seed)
        for name, (p_, q_, w_) in NETS.items():
            t0 = time.time(); W = build(p_, q_, w_)
            peak, final = run(W, lam); del W
            record(tag, name, peak, final, t0)
        vb = vel_rho(tag, "baseline k>=3")
        vs = [abs(vel_rho(tag, n)) for n in shuf_names]
        ok = bool(abs(vb) >= 0.9 and abs(vb) > max(vs))
        sweep[tag] = {"lam0": lam0, "vel_rho_baseline": vb, "vel_rho_shuffled_abs": vs,
                      "D2": bool(abs(vb) >= 0.9), "D3": bool(abs(vb) > max(vs)), "both": ok}
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
        print(f"{tag}: lam0 {lam0:.3f} vel_rho {vb:+.3f} vs shuffled max {max(vs):.3f} -> {ok} "
              f"| peak {rss:.2f} GB", flush=True)
        out["sweep"] = sweep; json.dump(out, open("step8_dynamics.json", "w"), indent=1)
    if any(sweep[f"sigma{sigma}_seed{s}"]["both"] for s in TAU_SEEDS) and D6 == "NONE IN RANGE":
        D6 = sigma

V["D6 how much tau"] = D6
out["verdicts"] = V
json.dump(out, open("step8_dynamics.json", "w"), indent=1)
print(f"\nD6 how much tau: {D6}", flush=True)
for k, val in V.items(): print(f"{k}: {val}")
