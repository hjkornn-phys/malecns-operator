"""Step 11: which antenna smelled it? One fixed point, no training, no memory.

Every earlier step that asked the wiring to do a task put a trained box outside it, and twice the answer was
VOID because a null arm could not learn (step 9) or could not learn even in-distribution (step 10). This step
removes the box. Nothing is trained. The only fitted quantity in the whole run is ONE SCALAR per arm and
stimulus set -- the centering offset m -- estimated on calibration trials that share no odor with any test set.

Question. Presented once with an odor that reaches the two antennae at different concentrations, does the real
wiring's lateral horn output say which side was stimulated more, on odors and on receptor neurons never used to
estimate the offset, better than shuffled wiring and better than a random relabelling of the readout's sides?
This is NOT navigation and nothing here says the fly steers this way; it is one stimulus, one fixed point, one bit.

WHY A NULL ARM AT CHANCE IS NOT VOID HERE (the difference from steps 9 and 10). There, an arm at chance might
merely have failed to train, so it could not be compared. Here no arm trains, so an arm at chance has nothing
left to fail at: chance IS its answer. The gate that replaces G is H below, on a positive control -- if the
ORACLE cannot do the task, the harness is wrong and nothing else is read.

Odor input. Hallem & Carlson 2006 via DoOR v2.0.1, exactly the panel, SCALE and BG that step 10 fixed
(LK-10, LK-11): 23 receptors x 110 odorants, SCALE = 163, BG = 0.5, Or33b excluded. Each ORN of a panel type is
on with probability 0.5 at c * (r / SCALE) * U(0.75, 1.25) where c is that side's concentration; each background
olfactory ORN is on with probability 0.05 at U(0, BG).

Sides, and what had to be cut for them (LK-12). `somaSide` is null for every ORN and `rootSide` is null for
every LH output neuron, so the two ends use different fields: ORN side from `rootSide`, readout side from
`somaSide`. The panel is laterally unbalanced (410 L / 611 R / 261 neither), which would tilt a
left-minus-right readout before any odor arrived, so each receptor type is cut to min(nL, nR) per side:
804 ORNs, 402 per side, down from step 10's 1,282. Readout is step 10's LH output set, ^LH(AV|AD|PV|PD),
1,929 bodies, 967 L + 962 R by `somaSide`.

Readout and statistic. Encoder h = ReLU(0.9 W h + 0.1 + u), float32, tol 1e-6, x = h(u) - h(0); baseline k>=3,
NOT the compacted network (the skill's known weak spot is olfaction, and step 10 used baseline for the same
reason). d = (sum_L x - sum_R x) / (sum_L x + sum_R x). Answer "left was stimulated more" iff d > m.
WHY CENTERED AND NOT RAW (LK-12): d carries a standing offset of -3.9e-2 to -4.4e-2 on the real wiring at every
contrast measured, including near-symmetric ones, while its own spread is 1.8e-2 to 5.9e-2. The network leans
one way before the odor arrives, and a raw sign test reads that lean as evidence. m is estimated per arm, per
contrast and per ORN set on calibration trials; it moved by less than 6% across contrasts in the lookup.
WHY NOT PAIRED. A paired statistic (same odor, gradient both ways) scored 1.000 for REAL at every contrast in
the lookup, because resetting the noise draws leaves the swap as the only difference between the two frames.
That measures sensitivity, not a task, and no animal gets the same stimulus twice. It is not used.

Splits. Odors: the step 10 split, 30% held out stratified by DoOR chemical class, seed 101 -> 77 train, 33 held
out. ORNs: per receptor type per side, 30% held out at seed 202, drawn so the held-out set is balanced L/R
within each type; the rest is the train set, also balanced.
  calib C  train odors,    train ORNs    400 trials  seed 211   estimates m for test A and test C
  calib B  train odors,    held-out ORNs 400 trials  seed 212   estimates m for test B
  test A   train odors,    train ORNs    400 trials  seed 213   in-distribution
  test C   held-out odors, train ORNs   1000 trials  seed 214   new odors, seen neurons
  test B   held-out odors, held-out ORNs 1000 trials seed 215   new odors, new neurons (the verdicts)
Every set is half left-stronger and half right-stronger, alternating, odor drawn uniformly from its pool.
1000 trials: a gap of 0.05 against chance needs far fewer, and LK-9's 290-934 covers the paired case.

Contrasts, primary declared before the run. PRIMARY = 1.0/0.5. Also scored: 1.0/0.0, 1.0/0.25, 1.0/0.8.
WHY 1.0/0.5 AND NOT 1.0/0.0: at 1.0/0.0 one antenna is silent, which is a unilateral stimulus rather than a
gradient and makes the question easier than the one asked. 1.0/0.8 is the shallowest measured and sat at 0.632
for REAL in the lookup, above chance but near the floor; it is scored and reported, not the primary.

Arms.
  REAL       baseline k>=3
  SHUF       presynaptic partners permuted, seeds 1000..1004
  PERM-READ  REAL's responses, but the 1,929 LH neurons relabelled into two groups of 967/962 at random,
             seeds 2000..2004. Asks whether the ANATOMICAL side labels carry the answer or any split would.
  ORACLE     the stimulus itself, summed over left panel ORNs minus right, same centering. A positive control
             and the harness gate; it sees the input directly and must pass.

Verdicts, fixed before running. Lower bound = 2.5% percentile of 2000 bootstrap resamples of trials, seed 7;
paired differences resample the same trials for both arms. All at PRIMARY contrast unless stated.
  H   harness works      ORACLE test B lower bound > 0.90. Else STOP and read nothing else.
  V1  works in-dist      REAL test A lower bound > 0.5.
  V2  generalises        REAL test B lower bound > 0.5.
  V3  wiring earns it    REAL test B above every SHUF seed (one-sided p ~ 1/6).
  V4  the sides earn it  REAL test B above every PERM-READ seed (one-sided p ~ 1/6).
  V5  new odors alone    REAL test C lower bound > 0.5.
  V6  contrast orders    REAL test B accuracy is non-increasing over contrasts 1.0/0.0, 1.0/0.25, 1.0/0.5,
                         1.0/0.8, allowing each step a bootstrap-lower-bound overlap. A readout that tracks
                         laterality must fade as the gradient shallows; an artefact need not.
  THE WIRING LATERALISES = V2 and V3 and V4. V1, V5, V6 are read alongside and do not gate it.
Readings, not verdicts: ORACLE against REAL as a ceiling; the offset m per arm and contrast; per-odor accuracy
against the odor's Hallem response norm.

Run from data/ after retention.py (and step10_odor_delay.py for door/):
  step11_side.py [--smoke] [--lookup] [--seeds=N]
--seeds=N: N shuffled AND N readout-permutation seeds instead of 5. The verdicts are unchanged; only the
  strength of the rank test moves, to one-sided p ~ 1/(N+1). Results go to step11_side_<N>seeds.json so the
  five-seed run is not overwritten.
--lookup: side-field coverage, pool balance, offset stability and timing. No verdicts.
--smoke: 24 trials per set, one SHUF and one PERM seed.
"""
import csv, json, os, resource, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import fpmodel as fm

SMOKE, LOOKUP = "--smoke" in sys.argv, "--lookup" in sys.argv
N_CAL, N_A, N_C, N_B = (24, 24, 24, 24) if SMOKE else (400, 400, 1000, 1000)
NSEED = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--seeds=")), 5))
SHUF_SEEDS = [1000] if SMOKE else list(range(1000, 1000 + NSEED))
PERM_SEEDS = [2000] if SMOKE else list(range(2000, 2000 + NSEED))
OUT = "step11_side.json" if NSEED == 5 else f"step11_side_{NSEED}seeds.json"
SCALE, BG, BOOT, CHUNK = 163, 0.5, 2000, 100
CONTRASTS = [(1.0, 0.0), (1.0, 0.25), (1.0, 0.5), (1.0, 0.8)]
PRIMARY = (1.0, 0.5)
UNITS = ["Or10a", "Or19a", "Or22a", "Or23a", "Or2a", "Or35a", "Or43a", "Or43b", "Or47a", "Or47b", "Or49b",
         "Or59b", "Or65a", "Or67a", "Or67c", "Or7a", "Or82a", "Or85a", "Or85b", "Or85f", "Or88a", "Or98a", "Or9a"]
RECEPTOR_TYPE = {"Or10a": "DL1", "Or19a": "DC1", "Or22a": "DM2", "Or23a": "DA3", "Or2a": "DA4m", "Or35a": "VC3",
                 "Or43a": "DA4l", "Or43b": "VM2", "Or47a": "DM3", "Or47b": "VA1v", "Or49b": "VA5", "Or59b": "DM4",
                 "Or65a": "DL3", "Or67a": "DM6", "Or67c": "VC4", "Or7a": "DL5", "Or82a": "VA6", "Or85a": "DM5",
                 "Or85b": "VM5d", "Or85f": "DL4", "Or88a": "VA1d", "Or98a": "VM5v", "Or9a": "VM3"}
torch.set_num_threads(max(1, os.cpu_count() - 1))
gb = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
t_start = time.time()

# ---- Hallem panel (step 10's loader, unchanged)
os.makedirs("door", exist_ok=True)
resp, cls = {}, {}
for u in UNITS:
    f = f"door/{u}.csv"
    if not os.path.exists(f):
        subprocess.run(["curl", "-sfL", "-o", f,
                        f"https://raw.githubusercontent.com/ropensci/DoOR.data/v2.0.1/data/{u}.csv"], check=True)
    r = list(csv.reader(open(f), delimiter=";")); i = r[0].index("Hallem.2006.EN") + 1
    for x in r[1:]:
        if x[i] in ("NA", "") or x[3] == "SFR": continue
        resp.setdefault(x[3], {})[u] = float(x[i]); cls[x[3]] = x[1]
ODORS = sorted(o for o in resp if len(resp[o]) == len(UNITS))
R = np.array([[resp[o][u] for u in UNITS] for o in ODORS])
assert R.shape == (110, 23), R.shape

rng101 = np.random.default_rng(101)
by_cls = {}
for j, o in enumerate(ODORS): by_cls.setdefault(cls[o], []).append(j)
held = []
for c, js in sorted(by_cls.items()):
    js = rng101.permutation(js); held += list(js[: int(round(0.3 * len(js)))])
HELD_ODOR = np.sort(held); TRAIN_ODOR = np.setdiff1d(np.arange(len(ODORS)), HELD_ODOR)
assert (len(TRAIN_ODOR), len(HELD_ODOR)) == (77, 33)

# ---- network, sides, pools
net = fm.load_network("baseline", torch.float32); N = net.N
side_cols = pd.read_parquet("annotations.parquet", columns=["bodyId", "superclass", "rootSide", "somaSide"])
side_cols = side_cols[side_cols.superclass.notna()].sort_values("bodyId").reset_index(drop=True)
assert len(side_cols) == N and (side_cols.bodyId.to_numpy() == net.ann.bodyId.to_numpy()).all(), "row order"
ROOT = side_cols.rootSide.fillna("-").astype(str).to_numpy()
SOMA = side_cols.somaSide.fillna("-").astype(str).to_numpy()
typ, klass = net.ann.type.astype(str), net.ann["class"].astype(str)

r202 = np.random.default_rng(202)
POOLS, cut = {}, {}
for u in UNITS:
    idx = np.flatnonzero(typ.eq("ORN_" + RECEPTOR_TYPE[u]).to_numpy())
    L, Rr = idx[ROOT[idx] == "L"], idx[ROOT[idx] == "R"]
    k = min(len(L), len(Rr))                              # balance the two sides within the type
    cut[u] = (len(idx), len(L), len(Rr), k)
    L, Rr = r202.permutation(L)[:k], r202.permutation(Rr)[:k]
    nh = int(round(0.3 * k))                              # same count held out on both sides
    POOLS[u] = {"held": {"L": np.sort(L[:nh]), "R": np.sort(Rr[:nh])},
                "train": {"L": np.sort(L[nh:]), "R": np.sort(Rr[nh:])}}
PANEL = np.concatenate([np.concatenate([p[s] for p in v.values() for s in "LR"]) for v in POOLS.values()])
OLF = np.flatnonzero(klass.eq("olfactory").to_numpy())
BGN = np.setdiff1d(OLF, PANEL)
LH = np.flatnonzero(typ.str.match(r"^LH(AV|AD|PV|PD)").to_numpy())
LH_L, LH_R = LH[SOMA[LH] == "L"], LH[SOMA[LH] == "R"]
assert len(LH_L) + len(LH_R) == len(LH), "an LH neuron has no somaSide"
for u in UNITS:
    for w in ("held", "train"):
        assert len(POOLS[u][w]["L"]) == len(POOLS[u][w]["R"]), f"{u} {w} unbalanced"
n_tr = sum(len(POOLS[u]["train"]["L"]) for u in UNITS)
n_hd = sum(len(POOLS[u]["held"]["L"]) for u in UNITS)
print(f"panel {len(PANEL)} = 2 x ({n_tr} train + {n_hd} held) per side; bg {len(BGN)}; "
      f"LH {len(LH)} = {len(LH_L)}L + {len(LH_R)}R; odors {len(TRAIN_ODOR)}/{len(HELD_ODOR)}", flush=True)


def sample(j, which, cL, cR, rng):
    """One presentation of odor j on ORN set `which`, left antenna at cL and right at cR."""
    rows, vals = [], []
    for ui, u in enumerate(UNITS):
        for s, c in (("L", cL), ("R", cR)):
            p = POOLS[u][which][s]; on = p[rng.random(len(p)) < 0.5]
            rows.append(on)
            vals.append(np.full(len(on), c * R[j, ui] / SCALE) * rng.uniform(0.75, 1.25, len(on)))
    b = BGN[rng.random(len(BGN)) < 0.05]
    rows.append(b); vals.append(rng.uniform(0, BG, len(b)))
    return np.concatenate(rows), np.concatenate(vals)


def trials(n, odors, which, cL, cR, seed):
    rng = np.random.default_rng(seed)
    cols, y, od = [], [], []
    for t in range(n):
        left = t % 2 == 0                                  # half the trials each way
        j = int(rng.choice(odors))
        cols.append(sample(j, which, cL if left else cR, cR if left else cL, rng))
        y.append(1.0 if left else 0.0); od.append(j)
    return cols, np.array(y), np.array(od)


def dense(cols):
    U = torch.zeros(N, len(cols))
    for k, (r, v) in enumerate(cols):
        U[torch.from_numpy(r.astype(np.int64)), k] = torch.from_numpy(v.astype(np.float32))
    return U


def solve(W, U, tol=1e-6, maxit=3000):
    H = torch.zeros_like(U)
    for it in range(1, maxit + 1):
        Hn = torch.relu(W @ H + 0.1 + U); d = (Hn - H).abs().max().item(); H = Hn
        if d < tol: return H, it
    raise RuntimeError(f"no convergence in {maxit} iterations (last change {d:.3g})")


def encoder(seed):
    s_neuron = np.ones(N); s_neuron[net.pre] = np.sign(net.val)
    pre = net.pre if seed is None else np.random.default_rng(seed).permutation(net.pre)
    A = sp.csr_matrix(((0.9 * s_neuron[pre] * np.abs(net.val)).astype(np.float32), (net.post, pre)), shape=(N, N))
    return fm._torch_csr(A, torch.float32)


def responses(W, h0, cols):
    """LH rows only: the full N x n_trials array would be 0.7 GB at 1000 trials."""
    out = []
    for i in range(0, len(cols), CHUNK):
        out.append((solve(W, dense(cols[i:i + CHUNK]))[0] - h0).numpy()[LH])
    return np.concatenate(out, 1)


def stat(X, gl, gr):
    a, b = X[gl].sum(0), X[gr].sum(0)
    return (a - b) / (a + b + 1e-12)


def boot_lb(correct, seed=7):
    r = np.random.default_rng(seed); n = len(correct)
    return float(np.percentile([correct[r.integers(0, n, n)].mean() for _ in range(BOOT)], 2.5))


# ---- stimulus sets: built once, identical for every arm
SETSPEC = {"calib C": (N_CAL, TRAIN_ODOR, "train", 211), "calib B": (N_CAL, TRAIN_ODOR, "held", 212),
           "test A": (N_A, TRAIN_ODOR, "train", 213), "test C": (N_C, HELD_ODOR, "train", 214),
           "test B": (N_B, HELD_ODOR, "held", 215)}
CAL_FOR = {"test A": "calib C", "test C": "calib C", "test B": "calib B"}
TESTS = ["test A", "test C", "test B"]
SETS = {c: {k: trials(n, od, wh, c[0], c[1], sd) for k, (n, od, wh, sd) in SETSPEC.items()} for c in CONTRASTS}
LH_POS = {int(b): i for i, b in enumerate(LH)}                       # LH body index -> row of the X arrays
ROW_L = np.array([LH_POS[int(b)] for b in LH_L]); ROW_R = np.array([LH_POS[int(b)] for b in LH_R])
PANEL_L = np.concatenate([POOLS[u][w]["L"] for u in UNITS for w in ("train", "held")])
PANEL_R = np.concatenate([POOLS[u][w]["R"] for u in UNITS for w in ("train", "held")])

if LOOKUP:
    print("\n-- LK-12 side-field coverage --")
    print(f"{'recep':7s} {'type':6s} {'all':>4s} {'rootL':>6s} {'rootR':>6s} {'kept/side':>9s}")
    for u in UNITS: print(f"{u:7s} {RECEPTOR_TYPE[u]:6s} {cut[u][0]:4d} {cut[u][1]:6d} {cut[u][2]:6d} {cut[u][3]:9d}")
    print(f"panel dropped by balancing: {sum(c[0] for c in cut.values())} -> {2 * sum(c[3] for c in cut.values())}")
    print(f"LH somaSide L/R = {len(LH_L)}/{len(LH_R)}, rootSide missing for {int((ROOT[LH] == '-').sum())}/{len(LH)}")
    print(f"ORN somaSide missing for {int((SOMA[PANEL] == '-').sum())}/{len(PANEL)}")
    W = encoder(None); h0, it0 = solve(W, torch.zeros(N, 1))
    print(f"\n-- offset stability, REAL, {N_CAL} calib trials per contrast (h0 in {it0} iters) --")
    for c in CONTRASTS:
        for k in ("calib C", "calib B"):
            d = stat(responses(W, h0, SETS[c][k][0]), ROW_L, ROW_R)
            print(f"  {c[0]}/{c[1]} {k}: m {d.mean():+.4e}  sd {d.std():.4e}")
    print(f"({time.time() - t_start:.0f}s, {gb():.2f} GB)")
    sys.exit(0)

# ---- run
res = {}
W_real = encoder(None); h0_real, _ = solve(W_real, torch.zeros(N, 1))
X_real = {}
for c in CONTRASTS:
    X_real[c] = {k: responses(W_real, h0_real, SETS[c][k][0]) for k in SETSPEC}
    print(f"REAL {c[0]}/{c[1]} done ({time.time() - t_start:.0f}s, {gb():.2f} GB)", flush=True)


def score(arm, dfun):
    """dfun(contrast, setname) -> the statistic per trial. Returns accuracy and lower bound per set."""
    out = {}
    for c in CONTRASTS:
        for k in TESTS:
            m = dfun(c, CAL_FOR[k]).mean()
            d, y = dfun(c, k), SETS[c][k][1]
            ok = ((d > m).astype(float) == y).astype(float)
            out[f"{c[0]}/{c[1]}|{k}"] = {"acc": float(ok.mean()), "lb": boot_lb(ok), "m": float(m), "n": len(ok)}
    res[arm] = out
    p = PRIMARY
    print(f"  {arm:14s} " + "  ".join(f"{k} {out[f'{p[0]}/{p[1]}|{k}']['acc']:.3f}"
                                      f"({out[f'{p[0]}/{p[1]}|{k}']['lb']:.3f})" for k in TESTS), flush=True)


print(f"\n-- accuracy at primary contrast {PRIMARY[0]}/{PRIMARY[1]}, acc(lower bound) --")
score("REAL", lambda c, k: stat(X_real[c][k], ROW_L, ROW_R))

for sd in PERM_SEEDS:                                              # REAL's responses, sides relabelled
    pr = np.random.default_rng(sd).permutation(len(LH))
    gl, gr = pr[: len(LH_L)], pr[len(LH_L):]
    score(f"PERM-READ-{sd}", lambda c, k, gl=gl, gr=gr: stat(X_real[c][k], gl, gr))


def oracle_stat(c, k):                                             # the stimulus itself, by antenna
    d = []
    for rows, vals in SETS[c][k][0]:
        a = vals[np.isin(rows, PANEL_L)].sum(); b = vals[np.isin(rows, PANEL_R)].sum()
        d.append((a - b) / (a + b + 1e-12))
    return np.array(d)


score("ORACLE", oracle_stat)

for sd in SHUF_SEEDS:
    Ws = encoder(sd); h0s, _ = solve(Ws, torch.zeros(N, 1))
    Xs = {c: {k: responses(Ws, h0s, SETS[c][k][0]) for k in SETSPEC} for c in CONTRASTS}
    score(f"SHUF-{sd}", lambda c, k, Xs=Xs: stat(Xs[c][k], ROW_L, ROW_R))
    del Xs
    print(f"    ({time.time() - t_start:.0f}s, {gb():.2f} GB)", flush=True)

# ---- verdicts
P = f"{PRIMARY[0]}/{PRIMARY[1]}"
g = lambda arm, k: res[arm][f"{P}|{k}"]
V = {}
V["H  harness works"] = g("ORACLE", "test B")["lb"] > 0.90
if not V["H  harness works"]:
    print(f"\nH FAILED: ORACLE test B lower bound {g('ORACLE', 'test B')['lb']:.3f} <= 0.90. Reading nothing else.")
    json.dump({"verdicts": {k: bool(v) for k, v in V.items()}, "arms": res}, open(OUT, "w"), indent=1)
    sys.exit(1)
V["V1 works in-dist"] = g("REAL", "test A")["lb"] > 0.5
V["V2 generalises"] = g("REAL", "test B")["lb"] > 0.5
V["V3 wiring earns it"] = all(g("REAL", "test B")["acc"] > g(f"SHUF-{s}", "test B")["acc"] for s in SHUF_SEEDS)
V["V4 the sides earn it"] = all(g("REAL", "test B")["acc"] > g(f"PERM-READ-{s}", "test B")["acc"] for s in PERM_SEEDS)
V["V5 new odors alone"] = g("REAL", "test C")["lb"] > 0.5
seq = [res["REAL"][f"{c[0]}/{c[1]}|test B"] for c in CONTRASTS]
V["V6 contrast orders"] = all(seq[i]["acc"] >= seq[i + 1]["lb"] for i in range(len(seq) - 1))
V["THE WIRING LATERALISES"] = V["V2 generalises"] and V["V3 wiring earns it"] and V["V4 the sides earn it"]

print("\n-- verdicts --")
for k, v in V.items(): print(f"  {k:24s} {'PASS' if v else 'FAIL'}")
print("\n-- REAL test B over contrasts --")
for c, s in zip(CONTRASTS, seq): print(f"  {c[0]}/{c[1]}: {s['acc']:.3f} ({s['lb']:.3f})  m {s['m']:+.3e}")
json.dump({"primary": P, "verdicts": {k: bool(v) for k, v in V.items()}, "arms": res,
           "panel": {"per_side": int(sum(c[3] for c in cut.values())), "lh": [len(LH_L), len(LH_R)]}},
          open(OUT, "w"), indent=1)
print(f"\nwrote {OUT} ({time.time() - t_start:.0f}s, {gb():.2f} GB)")
