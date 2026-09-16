"""Step 9 (taste pilot): delayed match-to-sample with a memory box outside the connectome.

PILOT STATUS. This is a small run on taste to check the design before the same question is asked on olfaction,
where the input pools are large enough for a confirmatory test. Its verdicts are fixed below and stand as
recorded, but its numbers are exploratory for the olfactory step and are never cited as evidence for it.

Question. The steady-state model has no time axis. A small trainable memory carried between frames gives it one
from outside. Does the real wiring let that memory solve a delayed comparison on neurons the memory was never
trained on, better than the memory alone and better than shuffled wiring? This leaves biology on purpose: the
memory is not a fly structure. Only the open loop (9a) is run here: the connectome is a fixed per-frame encoder
and every bit of the time dependence is the memory's.

Task (step 2a's classes and 30% held-out split, read from step2a_taste_task.npz):
  trial = frame 1: sample A -> D blank frames -> frame D+2: sample B -> answer: same class?
  Each class's TRAIN pool is split in two halves (seed 91): A is always drawn from half 1. B is drawn from
  half 2 (train, test A) or from the HELD-OUT pool (test B). So A and B never share a neuron, in any set.
  WHY: the LK-9 lookup found same-class pairs drawn from one pool share 34-46% of their active neurons
  (Jaccard), and 9.6% of held-out water pairs are the identical subset, while different-class pairs share none.
  Neuron overlap alone would solve the task without any taste identity.
  Sample rule is step 2a's: 50% of the pool (at least 3), amplitude U(0.5, 1), background gustatory noise
  p 0.05 at U(0, 0.5). Labels exactly half same; different pairs cycle over the 12 ordered class pairs.
  Trials: train 800 (seed 92), test A 400 (seed 93), test B 1000 (seed 94).
  WHY 1000: LK-9's arithmetic, a paired accuracy gap of 0.05 needs 290-934 trials at 80% power for discordant
  shares 0.1-0.3; a lower bound above 0.5 needs 93 at accuracy 0.6.
  Delays: every training trial is presented at D = 1 and at D = 4. Every test trial is scored at D = 1, 4, 8,
  the same trials at each D, so delays are compared paired. D = 8 is never trained.

Encoder: step 2a's model, h = ReLU(0.9 W h + 0.1 + u), float32, tol 1e-6. x = DN responses h(u) - h(0) for
  the 1,314 descending neurons; a blank frame is x = 0 exactly. Features kept: largest |x| over the train
  samples >= 1e-4 and variance > 0; x is divided by one scalar, the std of all kept train entries.
Memory: GRUCell(n_features -> k = 16), m_0 = 0, answer = sigmoid(v . m + c) after the B frame. k = 16 was fixed
  before LK-9 and is not tuned. Binary cross-entropy, full batch, Adam lr 1e-2, 1000 steps, no weight decay,
  no early stopping; the final step is the result. Nothing is selected on test A or B.
  Init seeds 0, 1, 2 per arm; a trial's score is its correctness averaged over the three.

Arms:
  REAL      baseline k>=3 encoder + memory
  SHUF      shuffled encoder, presynaptic partners permuted, seed 1000 (step 2a's shuffled seed 0) + memory.
            LK-9 found this a WEAK null: 97.5% of DNs respond against 31.4% for the real wiring, and it converges
            in 10 iterations against 62. No gain in 0.2-0.9 matches both responsive share and response size.
            V3 is therefore expected to be easy and carries little weight; V4 is the arm that decides.
  MEM-ONLY  the stimulus itself (the 1,428 gustatory sensory neurons) in place of x, same memory and training.
            Its test B input dimensions (held-out neurons) were never active in training.

Verdicts, fixed before running. "acc" is test-trial accuracy; "lower bound" is the 2.5% percentile over 2000
bootstrap resamples of trials, seed 7; paired differences resample the same trials for both arms.
  V0 encoder as measured   REAL's DN responsive share on the first 160 step 2a train samples is 0.314 +- 0.005,
                           LK-9's value. Otherwise the run stops.
  V1 learnable             REAL test A at D = 4: lower bound > 0.5
  V2 generalises           REAL test B at D = 4: lower bound > 0.5
  V3 wiring earns it       REAL test B at D = 4 above SHUF test B at D = 4 (one seed; see SHUF above)
  V4 beats memory alone    REAL minus MEM-ONLY, test B at D = 4: paired lower bound > 0
  V5 holds over delay      REAL test B at D = 8: lower bound > 0.5
  WIRING HELPS MEMORY = V2 and V3 and V4.

Run from data/ after step2a_taste.py:  step9_delay.py [--smoke]
--smoke: 24 trials per set, 20 training steps, prints timing and shapes but no scores.
"""
import json, os, resource, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import fpmodel as fm

SMOKE = "--smoke" in sys.argv
N_TR, N_A, N_B = (24, 24, 24) if SMOKE else (800, 400, 1000)
STEPS = 20 if SMOKE else 1000
K, LR, MIN_RESP, BOOT, CHUNK = 16, 1e-2, 1e-4, 2000, 160
ACTIVE, BG_P, BG_AMP = 0.5, 0.05, 0.5
TRAIN_D, TEST_D, INITS = [1, 4], [1, 4, 8], [0, 1, 2]
torch.set_num_threads(max(1, os.cpu_count() - 1))
gb = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9

z = np.load("step2a_taste_task.npz")
CLASSES, bg = [str(c) for c in z["classes"]], z["bg"]
net = fm.load_network("baseline", torch.float32)
N = net.N
sc = net.ann.superclass.astype(str)
dn = torch.from_numpy(np.flatnonzero(sc.eq("descending_neuron").to_numpy()))
gust = np.flatnonzero((net.ann["class"].astype(str).eq("gustatory") & sc.str.contains("sensory")).to_numpy())

rng = np.random.default_rng(91)
half1, half2 = {}, {}
for c in CLASSES:
    p = rng.permutation(z[f"pool_{c}_train"])
    half1[c], half2[c] = np.sort(p[: len(p) // 2]), np.sort(p[len(p) // 2:])
heldout = {c: z[f"pool_{c}_heldout"] for c in CLASSES}


def sample(pool, rng):
    on = rng.choice(pool, max(3, int(round(ACTIVE * len(pool)))), replace=False)
    b_on = bg[rng.random(len(bg)) < BG_P]
    return np.concatenate([on, b_on]), np.concatenate([rng.uniform(0.5, 1.0, len(on)), rng.uniform(0.0, BG_AMP, len(b_on))])


def trials(n, b_pools, seed):
    rng = np.random.default_rng(seed)
    pairs = [(i, j) for i in range(4) for j in range(4) if i != j]
    A, B, y = [], [], []
    for t in range(n):
        if t % 2 == 0:
            ca = cb = t // 2 % 4
        else:
            ca, cb = pairs[t // 2 % 12]
        A.append(sample(half1[CLASSES[ca]], rng)); B.append(sample(b_pools[CLASSES[cb]], rng)); y.append(float(ca == cb))
    return A, B, np.array(y, np.float32)


SETS = {"train": trials(N_TR, half2, 92), "test A": trials(N_A, half2, 93), "test B": trials(N_B, heldout, 94)}
print(f"half1 {[len(half1[c]) for c in CLASSES]} half2 {[len(half2[c]) for c in CLASSES]} "
      f"heldout {[len(heldout[c]) for c in CLASSES]}; trials {[len(s[2]) for s in SETS.values()]}", flush=True)


def dense_u(cols):
    U = torch.zeros(N, len(cols))
    for j, (r, v) in enumerate(cols): U[torch.from_numpy(r.astype(np.int64)), j] = torch.from_numpy(v.astype(np.float32))
    return U


def solve(W, U, tol=1e-6, maxit=3000):
    H = torch.zeros_like(U)
    for it in range(1, maxit + 1):
        Hn = torch.relu(W @ H + 0.1 + U); d = (Hn - H).abs().max().item(); H = Hn
        if d < tol: return H, it
    raise RuntimeError(f"no convergence in {maxit} iterations (last change {d:.3g})")


def encoder(which):
    s_neuron = np.ones(N); s_neuron[net.pre] = np.sign(net.val)
    pre = net.pre if which == "REAL" else np.random.default_rng(1000).permutation(net.pre)
    val = s_neuron[pre] * np.abs(net.val)
    A = fm.sp.csr_matrix(((0.9 * val).astype(np.float32), (net.post, pre)), shape=(N, N))
    return fm._torch_csr(A, torch.float32)


def dn_features(W, cols):
    h0, _ = solve(W, torch.zeros(N, 1))
    out, its = [], []
    for j in range(0, len(cols), CHUNK):
        H, it = solve(W, dense_u(cols[j:j + CHUNK])); its.append(it)
        out.append((H[dn] - h0[dn]).T.clone())
    return torch.cat(out), max(its)


def stim_features(cols):
    idx = {int(r): i for i, r in enumerate(gust)}
    X = torch.zeros(len(cols), len(gust))
    for j, (r, v) in enumerate(cols): X[j, [idx[int(i)] for i in r]] = torch.from_numpy(v.astype(np.float32))
    return X


def raw_inputs(arm):
    t0 = time.time()
    if arm == "MEM-ONLY":
        F = {s: (stim_features(A), stim_features(B)) for s, (A, B, _) in SETS.items()}
        return F, {"sec": time.time() - t0}
    W = encoder(arm)
    info = {}
    if arm == "REAL":
        cols = [(z["train_row"][z["train_col"] == j], z["train_val"][z["train_col"] == j]) for j in range(160)]
        X0, _ = dn_features(W, cols)
        info["V0 DN responsive share"] = float((X0.abs().max(0).values >= MIN_RESP).float().mean())
    F, its = {}, []
    for s, (A, B, _) in SETS.items():
        XA, i1 = dn_features(W, A); XB, i2 = dn_features(W, B); its += [i1, i2]
        F[s] = (XA, XB)
    del W
    info.update({"max_iters": max(its), "sec": time.time() - t0})
    return F, info


def standardise(F):
    tr = torch.cat(F["train"])
    keep = (tr.abs().max(0).values >= MIN_RESP) & (tr.var(0) > 0)
    scale = tr[:, keep].std()
    return {s: (a[:, keep] / scale, b[:, keep] / scale) for s, (a, b) in F.items()}, int(keep.sum())


class Memory(torch.nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.cell, self.out = torch.nn.GRUCell(n_in, K), torch.nn.Linear(K, 1)

    def forward(self, XA, XB, D):
        m = self.cell(XA, torch.zeros(len(XA), K))
        blank = torch.zeros_like(XA)
        for _ in range(D): m = self.cell(blank, m)
        return self.out(self.cell(XB, m)).squeeze(1)


def run_arm(F):
    y = {s: torch.from_numpy(SETS[s][2]) for s in SETS}
    correct = {s: {D: [] for D in TEST_D} for s in ("test A", "test B")}
    losses = []
    for seed in INITS:
        torch.manual_seed(seed)
        M = Memory(F["train"][0].shape[1])
        opt = torch.optim.Adam(M.parameters(), lr=LR)
        for _ in range(STEPS):
            opt.zero_grad()
            loss = sum(torch.nn.functional.binary_cross_entropy_with_logits(M(*F["train"], D), y["train"])
                       for D in TRAIN_D) / len(TRAIN_D)
            loss.backward(); opt.step()
        losses.append(float(loss))
        with torch.no_grad():
            for s in correct:
                for D in TEST_D:
                    correct[s][D].append(((M(*F[s], D) > 0).float() == y[s]).numpy())
    return {s: {D: np.mean(v, 0) for D, v in d.items()} for s, d in correct.items()}, losses


def lower(x, seed=7):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (BOOT, len(x)))
    return float(np.percentile(x[idx].mean(1), 2.5))


out = {"rules": __doc__, "trials": {s: len(v[2]) for s, v in SETS.items()}, "arms": {}}
C = {}
for arm in ["REAL", "SHUF", "MEM-ONLY"]:
    F, info = raw_inputs(arm)
    F, nf = standardise(F)
    t0 = time.time()
    C[arm], losses = run_arm(F)
    info.update({"features": nf, "final_train_loss": losses, "train_sec": time.time() - t0, "peak_GB": gb()})
    if not SMOKE:
        info["acc"] = {s: {str(D): float(v.mean()) for D, v in d.items()} for s, d in C[arm].items()}
    out["arms"][arm] = info
    print(f"{arm}: {json.dumps({k: v for k, v in info.items() if k != 'acc'})}", flush=True)
    if SMOKE:
        continue
    print(f"  acc {info['acc']}", flush=True)
    if arm == "REAL" and abs(info["V0 DN responsive share"] - 0.314) > 0.005:
        out["verdicts"] = {"V0 encoder as measured": False}
        json.dump(out, open("step9_delay.json", "w"), indent=1)
        sys.exit("V0 failed: encoder does not match LK-9; nothing below is interpretable")
    json.dump(out, open("step9_delay.json", "w"), indent=1)

if SMOKE:
    print("smoke: pipeline ran end to end; scores withheld")
    sys.exit(0)

R, S, M_ = C["REAL"]["test B"], C["SHUF"]["test B"], C["MEM-ONLY"]["test B"]
out["bounds"] = {"V1 REAL test A D4 lower": lower(C["REAL"]["test A"][4]), "V2 REAL test B D4 lower": lower(R[4]),
                 "V4 REAL - MEM-ONLY test B D4 lower": lower(R[4] - M_[4]), "V5 REAL test B D8 lower": lower(R[8])}
b = out["bounds"]
out["verdicts"] = {
    "V0 encoder as measured": True,
    "V1 learnable": b["V1 REAL test A D4 lower"] > 0.5,
    "V2 generalises": b["V2 REAL test B D4 lower"] > 0.5,
    "V3 wiring earns it": bool(R[4].mean() > S[4].mean()),
    "V4 beats memory alone": b["V4 REAL - MEM-ONLY test B D4 lower"] > 0,
    "V5 holds over delay": b["V5 REAL test B D8 lower"] > 0.5,
}
v = out["verdicts"]
v["WIRING HELPS MEMORY"] = v["V2 generalises"] and v["V3 wiring earns it"] and v["V4 beats memory alone"]
json.dump(out, open("step9_delay.json", "w"), indent=1)
for k, x in {**b, **v}.items(): print(f"{k}: {x}")
