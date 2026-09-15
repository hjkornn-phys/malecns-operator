"""Step 2b: does training the network's gains through the fixed point improve taste classification on held-out
neurons?

Task: step2a_taste.py's, read from step2a_taste_task.npz (same pools, samples and labels; run step 2a first).
Model: fpmodel.py, h* = ReLU(gamma[sc(post)] * (W0 @ (alpha[nt(pre)] * h*)) + s * u + b), trainable
  alpha (5 transmitter classes), gamma (one per superclass), s, b: 34 parameters, float32, tol 1e-6.
  Start equals step 2a's untrained model: alpha = 0.9 (a = ln 10), gamma = sigmoid(10), s = 1, b = 0.1.
Features: descending-neuron responses h(sample) - h(no stimulus). The feature set is fixed at the start
  (largest |response| over step 2a's train set >= 1e-4, variance > 0); features are centred by the mean of
  whichever set the readout is fitted on, not scaled (per-feature scaling divides by near-silent neurons and
  blew up in the smoke run).
Readout: ridge regression on one-hot labels, solved in closed form (dual), lambda = 0.1 x mean diagonal of the
  Gram matrix, so the readout is invariant to the overall response scale; the prediction is the argmax. Closed
  form, so the gradient through the readout is exact.

Training objective (fixed before running). Step 2a's train set is linearly separable (test A 1.000 for every
network), so a readout loss on it carries no signal. Instead, like test B, the loss asks for generalisation to
neurons the readout was not fitted on:
  each class's TRAIN pool is split in two halves (seed 21); 60 samples per class are drawn from each half
  (seeds 22, 23; step 2a's rule: 50% of the half, at least 3, amplitude U(0.5, 1), same background);
  loss = mean of [ridge fitted on half 1, squared error on half 2] and [fitted on half 2, error on half 1].
  Held-out (test B) neurons are never used. Adam, lr 0.05, STEPS steps (default 40); no early stopping,
  nothing selected on test A or B; the final step is the result.
Evaluation, at the start and at the end: ridge fitted on step 2a's train set, balanced accuracy on test A, B.
Networks (argv[1]): baseline k>=3; shuffled = baseline with presynaptic partners permuted, seed 1000
  (step 2a's shuffled seed 0).

Verdicts, fixed before running:
  T1 training works:         cross-fitted loss at the end <= 0.8 x at the start
  T2 held-out gain:          test B balanced accuracy end - start > 0, lower 95% bootstrap bound > 0
                             (paired over test B samples, 2000 resamples, seed 7)
  T3 gain needs the wiring:  trained baseline test B above trained shuffled test B (needs both runs)
Run from data/ after step2a_taste.py:  step2b_taste.py baseline|shuffled [steps] [--smoke]
--smoke: 6,000-neuron subnetwork downstream of the task neurons, 2 samples per class per set, 10 steps,
  non-sensory features, float64 at tol 1e-12; prints timing, gradient sizes and a directional
  finite-difference check of the gradient (limit 1e-4 relative), but no scores.
"""
import json, math, os, resource, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import fpmodel as fm

SMOKE = "--smoke" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
WHICH = args[0]
assert WHICH in ("baseline", "shuffled")
STEPS = 10 if SMOKE else (int(args[1]) if len(args) > 1 else 40)
DTYPE, TOL, CHUNK = (torch.float64, 1e-12, 160) if SMOKE else (torch.float32, 1e-6, 160)
RIDGE, MIN_RESP, LR, BOOT, N_HALF = 0.1, 1e-4, 0.05, 2000, 60
ACTIVE, BG_P, BG_AMP = 0.5, 0.05, 0.5
torch.manual_seed(0)

net = fm.load_network("baseline", DTYPE)
if WHICH == "shuffled":
    s_neuron = np.ones(net.N); s_neuron[net.pre] = np.sign(net.val)
    p = np.random.default_rng(1000).permutation(net.pre)
    net = fm.Net(net.ann, p, net.post, s_neuron[p] * np.abs(net.val), net.ntc, DTYPE)

z = np.load("step2a_taste_task.npz")
CLASSES, bg = [str(c) for c in z["classes"]], z["bg"]
SETS = {s: {"row": z[f"{k}_row"], "col": z[f"{k}_col"], "val": z[f"{k}_val"], "y": z[f"{k}_y"]}
        for s, k in [("train", "train"), ("test A", "testA"), ("test B", "testB")]}

rng = np.random.default_rng(21)
halves = []
for c in CLASSES:
    p_ = rng.permutation(z[f"pool_{c}_train"])
    halves.append((np.sort(p_[: len(p_) // 2]), np.sort(p_[len(p_) // 2:])))


def draw(h, n, seed):
    rng = np.random.default_rng(seed)
    rows, cols, vals, y = [], [], [], []
    for ci in range(len(CLASSES)):
        pool = halves[ci][h]
        for _ in range(n):
            on = rng.choice(pool, max(3, int(round(ACTIVE * len(pool)))), replace=False)
            b_on = bg[rng.random(len(bg)) < BG_P]
            r = np.concatenate([on, b_on])
            rows.append(r); cols.append(np.full(len(r), len(y)))
            vals.append(np.concatenate([rng.uniform(0.5, 1.0, len(on)), rng.uniform(0.0, BG_AMP, len(b_on))]))
            y.append(ci)
    return {"row": np.concatenate(rows), "col": np.concatenate(cols), "val": np.concatenate(vals), "y": np.array(y)}


SETS["half 1"], SETS["half 2"] = draw(0, N_HALF, 22), draw(1, N_HALF, 23)

sc = net.ann.superclass.astype(str)
feat = np.flatnonzero(sc.eq("descending_neuron").to_numpy())
if SMOKE:
    rows = np.unique(np.concatenate([SETS[s]["row"] for s in SETS]))
    keep, frontier, seen = list(rows), list(rows), set(int(r) for r in rows)
    while len(keep) < 6000 and frontier:
        nxt = [int(j) for j in np.unique(net.AT_sp[frontier].indices) if int(j) not in seen][: 6000 - len(keep)]
        keep += nxt; seen.update(nxt); frontier = nxt
    keep = np.sort(np.asarray(keep))
    net = net.sub(keep)
    feat = np.flatnonzero(~net.ann.superclass.astype(str).str.contains("sensory").to_numpy())
    for s in SETS:                                      # first 2 samples of each class, rows re-indexed
        st = SETS[s]
        first = np.concatenate([np.flatnonzero(st["y"] == c)[:2] for c in range(len(CLASSES))])
        new = np.full(len(st["y"]), -1); new[first] = np.arange(len(first))
        m = new[st["col"]] >= 0
        SETS[s] = {"row": np.searchsorted(keep, st["row"][m]), "col": new[st["col"][m]], "val": st["val"][m],
                   "y": st["y"][first]}
feat_t = torch.from_numpy(feat)


def dense_u(st, lo, hi):
    m = (st["col"] >= lo) & (st["col"] < hi)
    U = torch.zeros(net.N, hi - lo, dtype=DTYPE)
    U[torch.from_numpy(st["row"][m]), torch.from_numpy(st["col"][m] - lo)] = torch.from_numpy(st["val"][m]).to(DTYPE)
    return U


P = fm.Params(len(net.sc_names), per_sc=True, dtype=DTYPE)
with torch.no_grad():
    P.a.fill_(math.log(10.0)); P.c.fill_(10.0); P.s_raw.fill_(math.log(math.e - 1)); P.b.fill_(0.1)


def features(names, grad):
    """DN responses (samples x features, float64) for each named set; with grad=True the graph reaches P."""
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        h0 = fm.solve(torch.zeros(net.N, 1, dtype=DTYPE), P, net, TOL)[feat_t]
        out = []
        for s in names:
            K = len(SETS[s]["y"])
            out.append(torch.cat([(fm.solve(dense_u(SETS[s], lo, min(K, lo + CHUNK)), P, net, TOL)[feat_t] - h0).T
                                  for lo in range(0, K, CHUNK)]).double())
        return out


X0, = features(["train"], grad=False)
keep_t = torch.from_numpy(np.flatnonzero(((X0.abs().max(0).values >= MIN_RESP) & (X0.std(0) > 0)).numpy()))
onehot = lambda s: torch.nn.functional.one_hot(torch.from_numpy(SETS[s]["y"]), len(CLASSES)).double()


def ridge(Xf, Yf, Xp):
    """Fit on (Xf, Yf), predict Xp; features centred by Xf's mean. Differentiable."""
    Xf, Xp = Xf[:, keep_t], Xp[:, keep_t]
    mu = Xf.mean(0)
    Zf, Zp = Xf - mu, Xp - mu
    Ym = Yf.mean(0)
    Kf = Zf @ Zf.T
    lam = RIDGE * Kf.diagonal().mean() + 1e-12
    A = torch.linalg.solve(Kf + lam * torch.eye(len(Zf), dtype=Kf.dtype), Yf - Ym)
    return Zp @ (Zf.T @ A) + Ym


def cross_loss(X1, X2):
    Y1, Y2 = onehot("half 1"), onehot("half 2")
    e12 = ((ridge(X1, Y1, X2) - Y2) ** 2).sum(1).mean()
    e21 = ((ridge(X2, Y2, X1) - Y1) ** 2).sum(1).mean()
    return 0.5 * (e12 + e21)


def bal(y, p): return float(np.mean([(p[y == c] == c).mean() for c in range(len(CLASSES))]))


def evaluate():
    Xtr, XA, XB, X1, X2 = features(["train", "test A", "test B", "half 1", "half 2"], grad=False)
    out = {"cross loss": float(cross_loss(X1, X2))}
    for s, X in [("test A", XA), ("test B", XB)]:
        p = ridge(Xtr, onehot("train"), X).argmax(1).numpy()
        y = SETS[s]["y"]
        out[s] = {"balanced": bal(y, p), "recall": {CLASSES[c]: float((p[y == c] == c).mean()) for c in range(len(CLASSES))},
                  "pred": p.tolist()}
    return out


def peak_gb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9   # bytes on macOS


def check(where, unroll=400):
    """Smoke only: the implicit gradient of the cross loss along a random direction v, against backprop through
    the unrolled iteration (exact, limit 1e-6 relative) and central finite differences at three step sizes
    (ReLU kinks make these approximate)."""
    gen = torch.Generator().manual_seed(5)
    v = [torch.randn(q.shape, generator=gen, dtype=q.dtype) for q in P.parameters()]
    dot = lambda: float(sum((q.grad * vq).sum() for q, vq in zip(P.parameters(), v)))
    P.zero_grad(); cross_loss(*features(["half 1", "half 2"], grad=True)).backward(); g_imp = dot()

    def unrolled_features(names):
        h0 = fm.unrolled(torch.zeros(net.N, 1, dtype=DTYPE), P, net, unroll)[feat_t]
        return [(fm.unrolled(dense_u(SETS[s], 0, len(SETS[s]["y"])), P, net, unroll)[feat_t] - h0).T.double()
                for s in names]
    P.zero_grad(); cross_loss(*unrolled_features(["half 1", "half 2"])).backward(); g_unr = dot()
    P.zero_grad()
    fds = {}
    for eps in (1e-4, 1e-5, 1e-6):
        fd = []
        for sgn in (1, -1):
            with torch.no_grad():
                for q, vq in zip(P.parameters(), v): q.add_(sgn * eps * vq)
            fd.append(float(cross_loss(*features(["half 1", "half 2"], grad=False))))
            with torch.no_grad():
                for q, vq in zip(P.parameters(), v): q.sub_(sgn * eps * vq)
        fds[eps] = (fd[0] - fd[1]) / (2 * eps)
    rel = lambda a, b: abs(a - b) / max(abs(b), 1e-30)
    b_now = float(P.values()[3])
    print(f"gradient check at {where} (b {b_now:.3f}): implicit {g_imp:.9e} unrolled {g_unr:.9e} rel err "
          f"{rel(g_imp, g_unr):.2e} (limit 1e-6: {rel(g_imp, g_unr) < 1e-6}) | finite differences " +
          " ".join(f"eps {e:.0e}: rel err {rel(g_imp, f):.2e}" for e, f in fds.items()), flush=True)


t0 = time.time()
if SMOKE: check("start")
start = evaluate()
if not SMOKE:
    print(f"{WHICH} start: cross loss {start['cross loss']:.4f} | test A {start['test A']['balanced']:.3f} "
          f"test B {start['test B']['balanced']:.3f} {start['test B']['recall']} | {time.time() - t0:.0f}s", flush=True)
adam = torch.optim.Adam(P.parameters(), lr=LR)
hist = []
for step in range(1, STEPS + 1):
    adam.zero_grad()
    l = cross_loss(*features(["half 1", "half 2"], grad=True))
    l.backward()
    gnorm = float(torch.cat([q.grad.reshape(-1) for q in P.parameters()]).norm())
    adam.step()
    a, g, s, b = (v.detach().double() for v in P.values())
    hist.append(dict(step=step, loss=l.item(), grad_norm=gnorm, s=float(s), b=float(b), alpha=a.tolist(),
                     fwd=fm.STATS["fwd_iters"], bwd=fm.STATS["bwd_iters"], sec=time.time() - t0))
    print(f"step {step:3d} cross loss {l.item():.4f} |grad| {gnorm:.2e} s {float(s):.3f} b {float(b):.3f} "
          f"iters {fm.STATS['fwd_iters']}/{fm.STATS['bwd_iters']} | {time.time() - t0:.0f}s peak {peak_gb():.2f} GB",
          flush=True)
if SMOKE:
    check("end")
    print(f"smoke: {net.N} neurons, {net.edges} edges, {len(keep_t)} features; scores withheld"); sys.exit(0)

end = evaluate()
yB = SETS["test B"]["y"]
p0, p1 = np.array(start["test B"]["pred"]), np.array(end["test B"]["pred"])
boot = np.random.default_rng(7).integers(0, len(yB), (BOOT, len(yB)))
gain = [bal(yB[i], p1[i]) - bal(yB[i], p0[i]) for i in boot]
out = {"rules": __doc__, "network": WHICH, "steps": STEPS, "parameters": P.names(net.sc_names),
       "features_used": len(keep_t), "start": start, "end": end, "history": hist, "final_raw": P.flat().tolist(),
       "test_B_gain": {"value": end["test B"]["balanced"] - start["test B"]["balanced"],
                       "ci": [float(np.quantile(gain, 0.025)), float(np.quantile(gain, 0.975))]}}
out["verdicts"] = {"T1 training works": bool(end["cross loss"] <= 0.8 * start["cross loss"]),
                   "T2 held-out gain": bool(out["test_B_gain"]["value"] > 0 and out["test_B_gain"]["ci"][0] > 0)}
other = f"step2b_taste_{'shuffled' if WHICH == 'baseline' else 'baseline'}.json"
if os.path.exists(other):
    o = json.load(open(other))
    mine, theirs = end["test B"]["balanced"], o["end"]["test B"]["balanced"]
    out["verdicts"]["T3 gain needs the wiring"] = bool(mine > theirs if WHICH == "baseline" else theirs > mine)
json.dump(out, open(f"step2b_taste_{WHICH}.json", "w"), indent=1)
print(f"\n{WHICH} end: cross loss {start['cross loss']:.4f} -> {end['cross loss']:.4f} | test A "
      f"{start['test A']['balanced']:.3f} -> {end['test A']['balanced']:.3f} | test B {start['test B']['balanced']:.3f} -> "
      f"{end['test B']['balanced']:.3f} [gain CI {out['test_B_gain']['ci'][0]:+.3f}, {out['test_B_gain']['ci'][1]:+.3f}]")
print("test B recall start", start["test B"]["recall"], "\n              end  ", end["test B"]["recall"])
for k, v in out["verdicts"].items(): print(f"{k}: {v}")
