"""Step 10: delayed match on real odors, with a memory box outside the connectome.

The confirmatory version of step 9, whose taste pilot passed every verdict but passed V3 and V4 on broken arms:
its shuffled and memory-only arms scored at chance on test A too, having memorised the training trials. This
step keeps step 9's question and fixes that, and moves to olfaction, where the input pools are large.

Question. With an outside memory carried between frames, does the real wiring let that memory decide whether two
odor presentations are the same odor when both the ODOR and the RECEPTOR NEURONS carrying it were never seen in
training, better than the memory alone and better than shuffled wiring? The memory is not a fly structure and
nothing here says the fly does this; the connectome is a fixed per-frame encoder (open loop).

Odor input:
  Hallem & Carlson 2006 as stored in DoOR.data v2.0.1 (per-receptor files, column Hallem.2006.EN): 23 odorant
  receptors x 110 odorants, one concentration (10^-2), spikes/s change from spontaneous, no missing entry.
  WHY THIS PANEL ALONE (LK-10): it is the largest complete rectangle of DoOR over MaleCNS-mapped units, and
  DoOR's missing entries are not missing at random (odorants other studies chose respond more strongly, +0.26 z,
  p = 0.0002), so imputing them would bias the input. Or33b is excluded: DoOR maps it to DM5+DM3.
  Caveat, stated and not corrected: empty-neuron (ab3A) recordings; Ir25a is co-expressed in most native ORN
  classes (Task et al. 2022) and may shift native responses. The claim is about this input code.
  Receptor -> MaleCNS type: RECEPTOR_TYPE below, 1,282 ORN bodies.
  One sample of odor o: each ORN of a panel type is on with probability 0.5 at (r / SCALE) * U(0.75, 1.25), r in
  spikes/s, negative values kept; each of the other 1,357 olfactory ORNs is on with probability 0.05 at U(0, BG).
  SCALE = 163, BG = 0.5 (LK-11): SCALE matches the odor panel's median input L2 norm to step 2a's taste samples
  (3.43), chosen on input alone after the drafted DN-based rule had no solution (SCALE 100 moved 38 DNs against
  taste's 171). BG is the largest of 0.5, 0.1, 0.02 whose background norm (2.38) stays below 3.43.

Splits (seeds as in LK-11, which reproduced them):
  Odors: 30% held out (seed 101), stratified by DoOR chemical class: 77 train, 33 held out.
  ORNs: per panel type 30% held out (seed 102), the rest split into halves h1, h2 (seed 103). Smallest pool 5.
  Frame A is always drawn on h1; frame B on h2 or on the held-out ORNs, so A and B never share a neuron.
  Label: even trials same odor, odd trials a different odor, the ordered pair drawn uniformly within the set.
    train    train odors,    B on h2           800 trials   seed 111
    test A   train odors,    B on h2           400 trials   seed 112   seen odors, seen neurons
    test C   held-out odors, B on h2          1000 trials   seed 113   new odors, seen neurons
    test B   held-out odors, B on held-out    1000 trials   seed 114   new odors, new neurons (the verdicts)
  WHY TWO HELD-OUT AXES. Held-out odors alone hand the memory-only arm the win: the raw ORN vector is the
  receptor code, and processing can only lose information about it. Held-out neurons alone test little beyond
  glomerular convergence. Test C separates the odor axis. 1000 trials: a paired gap of 0.05 needs 290-934 (LK-9).
  Delays: training at D = 1 and D = 4; every test trial scored at D = 1, 4, 8.

Encoder: h = ReLU(0.9 W h + 0.1 + u), float32, tol 1e-6, x = h(u) - h(0).
  Readout: lateral horn output neurons, types matching ^LH(AV|AD|PV|PD), 1,929 bodies.
  WHY NOT DNs (LK-11): at SCALE 163 only 9 DNs pass 1e-3 on the real wiring against 754 and 728 on shuffled seeds.
  LH output: 1,648 past 1e-3 real against 894 and 877 shuffled, with shuffled responses about 5.7x smaller rather
  than saturated. MBON (97) barely separated from shuffled and is not used.
Memory, identical for every arm (the step 9 fix):
  input features kept at max |x| >= 1e-4 over train samples and variance > 0, divided by one scalar (the std of all
  kept train entries); then a FIXED random projection to 64 dimensions, P ~ N(0, 1 / n_in) at seed 140, not
  trained, and divided by the std of the projected train entries. GRUCell(64 -> 16), answer sigmoid(v . m + c)
  after the B frame. BCE, full batch, AdamW lr 1e-2, weight decay 1e-3, 1000 steps, no early stopping, nothing
  selected on any test set. Init seeds 0, 1, 2; a trial's score is its correctness averaged over the three.
  The 64-dimensional projection and the weight decay were fixed before any arm was trained and were NOT checked
  on accuracy: the gate G below is what catches them failing.

Arms:
  REAL      baseline k>=3
  SHUF      presynaptic partners permuted, seeds 1000..1004 (LK-11: 4 min per seed)
  MEM-ONLY  the stimulus itself over all 2,639 olfactory ORNs in place of x
  POOLED    the stimulus summed within each of the 23 panel types: an ORACLE for glomerular convergence given
            the type labels the wiring has to discover. A reading, never a verdict.

Verdicts, fixed before running. Lower bound = 2.5% percentile of 2000 bootstrap resamples of trials, seed 7; paired
differences resample the same trials for both arms.
  G  arm is valid          an arm's test A accuracy at D = 4 has lower bound > 0.5. An arm that cannot do the
                           task on what it was trained on cannot be compared on anything else.
  V0 encoder as measured   REAL's median over LH output neurons of max |x| on LK-11's 160-sample batch (seed 130)
                           is 4.818e-3 +- 5%. LK-11 found the drafted share-based V0 useless here: 1,925 of
                           1,929 neurons pass 1e-4 for the real wiring and for both shuffled seeds. Else stop.
  V1 learnable             G holds for REAL.
  V2 generalises           REAL test B at D = 4: lower bound > 0.5.
  V3 wiring earns it       REAL test B at D = 4 above every SHUF seed (one-sided p ~ 1/6).
                           VOID if any SHUF seed fails G.
  V4 beats memory alone    REAL minus MEM-ONLY, test B at D = 4: paired lower bound > 0. VOID if MEM-ONLY fails G.
  V5 holds over delay      REAL test B at D = 8: lower bound > 0.5.
  V6 new odors alone       REAL test C at D = 4: lower bound > 0.5.
  WIRING HELPS MEMORY = V2 and V3 and V4; VOID if V3 or V4 is VOID and neither is False.
Readings, not verdicts: POOLED against REAL; on different-odor test B trials, error rate against the Hallem
  response correlation of the pair.
Saved for figures: step10_odor_delay.npz holds, per arm, set and delay, each trial's mean probability and
  seed-averaged correctness; the odor pair and label of every trial; and for the first 200 test B trials at D = 8,
  init seed 0, the memory state at every frame.

Run from data/ after retention.py and step2a_taste.py (for nothing but the file layout):
  step10_odor_delay.py [--smoke]
--smoke: 24 trials per set, 20 steps, one SHUF seed; prints V0's value, timing and shapes, no scores.
DoOR files are fetched into data/door/ from ropensci/DoOR.data at tag v2.0.1 if absent.
"""
import csv, json, os, resource, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import scipy.sparse as sp
import torch
import fpmodel as fm

SMOKE = "--smoke" in sys.argv
N_TR, N_A, N_C, N_B = (24, 24, 24, 24) if SMOKE else (800, 400, 1000, 1000)
STEPS = 20 if SMOKE else 1000
SHUF = [1000] if SMOKE else [1000, 1001, 1002, 1003, 1004]
SCALE, BG, K, PROJ, LR, WD, MIN_RESP, BOOT, CHUNK = 163, 0.5, 16, 64, 1e-2, 1e-3, 1e-4, 2000, 160
TRAIN_D, TEST_D, INITS = [1, 4], [1, 4, 8], [0, 1, 2]
V0_REF = 4.818327724933624e-3
UNITS = ["Or10a", "Or19a", "Or22a", "Or23a", "Or2a", "Or35a", "Or43a", "Or43b", "Or47a", "Or47b", "Or49b", "Or59b",
         "Or65a", "Or67a", "Or67c", "Or7a", "Or82a", "Or85a", "Or85b", "Or85f", "Or88a", "Or98a", "Or9a"]
RECEPTOR_TYPE = {"Or10a": "DL1", "Or19a": "DC1", "Or22a": "DM2", "Or23a": "DA3", "Or2a": "DA4m", "Or35a": "VC3",
                 "Or43a": "DA4l", "Or43b": "VM2", "Or47a": "DM3", "Or47b": "VA1v", "Or49b": "VA5", "Or59b": "DM4",
                 "Or65a": "DL3", "Or67a": "DM6", "Or67c": "VC4", "Or7a": "DL5", "Or82a": "VA6", "Or85a": "DM5",
                 "Or85b": "VM5d", "Or85f": "DL4", "Or88a": "VA1d", "Or98a": "VM5v", "Or9a": "VM3"}
torch.set_num_threads(max(1, os.cpu_count() - 1))
gb = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9

# ---- Hallem 2006 panel
os.makedirs("door", exist_ok=True)
resp, cls = {}, {}
for u in UNITS:
    f = f"door/{u}.csv"
    if not os.path.exists(f):
        subprocess.run(["curl", "-sfL", "-o", f, f"https://raw.githubusercontent.com/ropensci/DoOR.data/v2.0.1/data/{u}.csv"], check=True)
    r = list(csv.reader(open(f), delimiter=";")); i = r[0].index("Hallem.2006.EN") + 1
    for x in r[1:]:
        if x[i] in ("NA", "") or x[3] == "SFR": continue
        resp.setdefault(x[3], {})[u] = float(x[i]); cls[x[3]] = x[1]
ODORS = sorted(o for o in resp if len(resp[o]) == len(UNITS))
R = np.array([[resp[o][u] for u in UNITS] for o in ODORS])
assert R.shape == (110, 23), R.shape

rng = np.random.default_rng(101)
by_cls = {}
for j, o in enumerate(ODORS): by_cls.setdefault(cls[o], []).append(j)
held = []
for c, js in sorted(by_cls.items()):
    js = rng.permutation(js); held += list(js[: int(round(0.3 * len(js)))])
HELD_ODOR = np.sort(held); TRAIN_ODOR = np.setdiff1d(np.arange(len(ODORS)), HELD_ODOR)

net = fm.load_network("baseline", torch.float32)
N = net.N
typ, klass = net.ann.type.astype(str), net.ann["class"].astype(str)
r102, r103 = np.random.default_rng(102), np.random.default_rng(103)
POOLS = {}
for u in UNITS:
    p = r102.permutation(np.flatnonzero(typ.eq("ORN_" + RECEPTOR_TYPE[u]).to_numpy())); k = int(round(0.3 * len(p)))
    rest = r103.permutation(p[k:])
    POOLS[u] = {"held": np.sort(p[:k]), "h1": np.sort(rest[: len(rest) // 2]), "h2": np.sort(rest[len(rest) // 2:])}
OLF = np.flatnonzero(klass.eq("olfactory").to_numpy())
PANEL = np.concatenate([np.concatenate(list(v.values())) for v in POOLS.values()])
BGN = np.setdiff1d(OLF, PANEL)
LH = np.flatnonzero(typ.str.match(r"^LH(AV|AD|PV|PD)").to_numpy())
assert (len(TRAIN_ODOR), len(HELD_ODOR), len(PANEL), len(BGN), len(LH)) == (77, 33, 1282, 1357, 1929)


def odor_sample(j, half, rng):
    rows, vals = [], []
    for ui, u in enumerate(UNITS):
        p = POOLS[u][half]; on = p[rng.random(len(p)) < 0.5]
        rows.append(on); vals.append(np.full(len(on), R[j, ui] / SCALE) * rng.uniform(0.75, 1.25, len(on)))
    b = BGN[rng.random(len(BGN)) < 0.05]
    rows.append(b); vals.append(rng.uniform(0, BG, len(b)))
    return np.concatenate(rows), np.concatenate(vals)


def trials(n, odors, b_half, seed):
    rng = np.random.default_rng(seed)
    A, B, pairs, y = [], [], [], []
    for t in range(n):
        if t % 2 == 0:
            a = b = int(rng.choice(odors))
        else:
            a, b = (int(x) for x in rng.choice(odors, 2, replace=False))
        A.append(odor_sample(a, "h1", rng)); B.append(odor_sample(b, b_half, rng)); pairs.append((a, b)); y.append(float(a == b))
    return A, B, np.array(pairs), np.array(y, np.float32)


SETS = {"train": trials(N_TR, TRAIN_ODOR, "h2", 111), "test A": trials(N_A, TRAIN_ODOR, "h2", 112),
        "test C": trials(N_C, HELD_ODOR, "h2", 113), "test B": trials(N_B, HELD_ODOR, "held", 114)}
TESTS = ["test A", "test C", "test B"]
print(f"odors {len(TRAIN_ODOR)} train / {len(HELD_ODOR)} held; panel {len(PANEL)} bg {len(BGN)} LH {len(LH)}; "
      f"trials {[len(s[3]) for s in SETS.values()]}", flush=True)


def dense(cols):
    U = torch.zeros(N, len(cols))
    for k, (r, v) in enumerate(cols): U[torch.from_numpy(r.astype(np.int64)), k] = torch.from_numpy(v.astype(np.float32))
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


def lh_features(W, cols, h0):
    out, its = [], []
    for j in range(0, len(cols), CHUNK):
        H, it = solve(W, dense(cols[j:j + CHUNK])); its.append(it)
        out.append((H[LH] - h0[LH]).T.clone())
    return torch.cat(out), max(its)


def stim(cols, pooled):
    if pooled:
        X = torch.zeros(len(cols), len(UNITS))
        where = {int(r): ui for ui, u in enumerate(UNITS) for r in np.concatenate(list(POOLS[u].values()))}
        for k, (r, v) in enumerate(cols):
            for i, x in zip(r, v):
                if int(i) in where: X[k, where[int(i)]] += float(x)
        return X
    idx = {int(r): i for i, r in enumerate(OLF)}
    X = torch.zeros(len(cols), len(OLF))
    for k, (r, v) in enumerate(cols): X[k, [idx[int(i)] for i in r]] = torch.from_numpy(v.astype(np.float32))
    return X


def raw_inputs(arm):
    t0 = time.time(); info = {}
    if arm in ("MEM-ONLY", "POOLED"):
        F = {s: (stim(v[0], arm == "POOLED"), stim(v[1], arm == "POOLED")) for s, v in SETS.items()}
        return F, {"sec": time.time() - t0}
    W = encoder(None if arm == "REAL" else int(arm.split()[-1]))
    h0, _ = solve(W, torch.zeros(N, 1))
    if arm == "REAL":
        r_ = np.random.default_rng(130)
        cols = [odor_sample(j, "h1", r_) for j in r_.choice(TRAIN_ODOR, 160)]
        X0, _ = lh_features(W, cols, h0)
        info["V0 median LH max |x|"] = float(X0.abs().max(0).values.median())
    F, its = {}, []
    for s, (A, B, _, _) in SETS.items():
        XA, i1 = lh_features(W, A, h0); XB, i2 = lh_features(W, B, h0); its += [i1, i2]
        F[s] = (XA, XB)
    del W
    info.update({"max_iters": max(its), "sec": time.time() - t0})
    return F, info


def prepare(F):
    tr = torch.cat(F["train"])
    keep = (tr.abs().max(0).values >= MIN_RESP) & (tr.var(0) > 0)
    s1 = tr[:, keep].std()
    P = torch.from_numpy(np.random.default_rng(140).normal(0, 1 / np.sqrt(int(keep.sum())), (int(keep.sum()), PROJ)).astype(np.float32))
    proj = lambda X: (X[:, keep] / s1) @ P
    s2 = torch.cat([proj(a) for a in F["train"]]).std()
    return {s: (proj(a) / s2, proj(b) / s2) for s, (a, b) in F.items()}, int(keep.sum())


class Memory(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cell, self.out = torch.nn.GRUCell(PROJ, K), torch.nn.Linear(K, 1)

    def forward(self, XA, XB, D, states=False):
        m = self.cell(XA, torch.zeros(len(XA), K)); traj = [m]
        blank = torch.zeros_like(XA)
        for _ in range(D):
            m = self.cell(blank, m); traj.append(m)
        m = self.cell(XB, m); traj.append(m)
        logit = self.out(m).squeeze(1)
        return (logit, torch.stack(traj, 1)) if states else logit


def run_arm(F):
    y = {s: torch.from_numpy(SETS[s][3]) for s in SETS}
    prob = {s: {D: [] for D in TEST_D} for s in TESTS}
    losses, traj = [], None
    for seed in INITS:
        torch.manual_seed(seed)
        M = Memory()
        opt = torch.optim.AdamW(M.parameters(), lr=LR, weight_decay=WD)
        for _ in range(STEPS):
            opt.zero_grad()
            loss = sum(torch.nn.functional.binary_cross_entropy_with_logits(M(*F["train"], D), y["train"])
                       for D in TRAIN_D) / len(TRAIN_D)
            loss.backward(); opt.step()
        losses.append(float(loss.detach()))
        with torch.no_grad():
            for s in TESTS:
                for D in TEST_D:
                    prob[s][D].append(torch.sigmoid(M(*F[s], D)).numpy())
            if seed == 0:
                XA, XB = F["test B"]
                traj = M(XA[:200], XB[:200], 8, states=True)[1].numpy()
    p = {s: {D: np.mean(v, 0) for D, v in d.items()} for s, d in prob.items()}
    c = {s: {D: np.mean([(q > 0.5) == SETS[s][3] for q in v], 0) for D, v in d.items()} for s, d in prob.items()}
    return p, c, losses, traj


def lower(x, seed=7):
    idx = np.random.default_rng(seed).integers(0, len(x), (BOOT, len(x)))
    return float(np.percentile(x[idx].mean(1), 2.5))


ARMS = ["REAL"] + [f"SHUF {s}" for s in SHUF] + ["MEM-ONLY", "POOLED"]
out = {"rules": __doc__, "odors": ODORS, "held_odors": HELD_ODOR.tolist(), "arms": {}}
save = {f"pairs {s}": v[2] for s, v in SETS.items()} | {f"label {s}": v[3] for s, v in SETS.items()}
save["hallem"] = R
C = {}
for arm in ARMS:
    F, info = raw_inputs(arm)
    F, nf = prepare(F)
    t0 = time.time()
    p, C[arm], losses, traj = run_arm(F)
    del F
    info.update({"features": nf, "final_train_loss": losses, "train_sec": time.time() - t0, "peak_GB": gb()})
    out["arms"][arm] = info
    print(f"{arm}: {json.dumps(info)}", flush=True)
    if arm == "REAL" and not SMOKE and abs(info["V0 median LH max |x|"] / V0_REF - 1) > 0.05:
        out["verdicts"] = {"V0 encoder as measured": False}
        json.dump(out, open("step10_odor_delay.json", "w"), indent=1)
        sys.exit("V0 failed: the encoder does not match LK-11; nothing below is interpretable")
    if SMOKE: continue
    info["acc"] = {s: {str(D): float(v.mean()) for D, v in d.items()} for s, d in C[arm].items()}
    print(f"  acc {info['acc']}", flush=True)
    for s in TESTS:
        for D in TEST_D:
            save[f"{arm}|{s}|D{D}|prob"] = p[s][D]; save[f"{arm}|{s}|D{D}|correct"] = C[arm][s][D]
    save[f"{arm}|traj test B D8 seed0 first200"] = traj
    json.dump(out, open("step10_odor_delay.json", "w"), indent=1)
    np.savez_compressed("step10_odor_delay.npz", **save)

if SMOKE:
    print("smoke: pipeline ran end to end; scores withheld"); sys.exit(0)

gate = {a: lower(C[a]["test A"][4]) for a in ARMS}
G = {a: g > 0.5 for a, g in gate.items()}
RB = C["REAL"]["test B"]
b = {"G lower bounds (test A, D 4)": gate,
     "V2 REAL test B D4 lower": lower(RB[4]),
     "V4 REAL - MEM-ONLY test B D4 lower": lower(RB[4] - C["MEM-ONLY"]["test B"][4]),
     "V5 REAL test B D8 lower": lower(RB[8]),
     "V6 REAL test C D4 lower": lower(C["REAL"]["test C"][4])}
shuf = [a for a in ARMS if a.startswith("SHUF")]
v = {"V0 encoder as measured": True, "V1 learnable": G["REAL"],
     "V2 generalises": b["V2 REAL test B D4 lower"] > 0.5,
     "V3 wiring earns it": "VOID" if not all(G[a] for a in shuf) else bool(all(RB[4].mean() > C[a]["test B"][4].mean() for a in shuf)),
     "V4 beats memory alone": "VOID" if not G["MEM-ONLY"] else b["V4 REAL - MEM-ONLY test B D4 lower"] > 0,
     "V5 holds over delay": b["V5 REAL test B D8 lower"] > 0.5,
     "V6 new odors alone": b["V6 REAL test C D4 lower"] > 0.5}
parts = [v["V2 generalises"], v["V3 wiring earns it"], v["V4 beats memory alone"]]
v["WIRING HELPS MEMORY"] = False if False in parts else ("VOID" if "VOID" in parts else True)
out.update({"bounds": b, "verdicts": v})
json.dump(out, open("step10_odor_delay.json", "w"), indent=1)
for k, x in {**b, **v}.items(): print(f"{k}: {x}")
