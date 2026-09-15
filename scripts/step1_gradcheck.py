"""Step 1a: are the implicit fixed-point gradients in fpmodel.py correct?

Pass criteria, fixed before running:
  A. Subnetwork (6,000 neurons downstream of sensory seeds), float64, every parameter:
     implicit vs central finite differences   max relative error < 1e-4
     implicit vs backprop through the unrolled iteration   max relative error < 1e-6
  B. Full baseline network (k>=3), float64, 4 stimuli, 10 parameters (5 alpha, s, b, 3 largest-superclass gamma):
     implicit vs finite differences   max relative error < 1e-3
  C. Full baseline, float32, 29 stimuli: forward + backward time and peak memory (reported, no threshold).
Relative error = |x - y| / max(|x|, |y|, 1e-10).  Loss L = sum(C * h*) with fixed random C.
Run from data/ after retention.py.
"""
import json, os, resource, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import fpmodel as fm

torch.manual_seed(0)
rng = np.random.default_rng(0)
out = {}


def rel(x, y): return np.abs(x - y) / np.maximum(np.maximum(np.abs(x), np.abs(y)), 1e-10)


def init_params(net, dtype):
    P = fm.Params(len(net.sc_names), per_sc=True, dtype=dtype)
    with torch.no_grad():
        P.a.copy_(torch.randn(5, dtype=dtype) * 0.5 + 1.0)       # alpha ~ 0.73
        P.c.copy_(torch.randn(len(net.sc_names), dtype=dtype) * 0.5 + 1.5)
        P.s_raw.fill_(0.5); P.b.fill_(0.1)
    return P


def implicit_grad(net, P, U, C, tol):
    P.zero_grad()
    t = time.time()
    h = fm.solve(U, P, net, tol=tol)
    (C * h).sum().backward()
    g = torch.cat([p.grad.reshape(-1) for p in P.parameters()]).numpy().copy()
    return g, h, time.time() - t, fm.STATS["fwd_iters"], fm.STATS["bwd_iters"]


def fd_grad(net, P, U, C, tol, eps, idx):
    params = list(P.parameters())
    sizes = [p.numel() for p in params]
    out = []
    for i in idx:
        k = int(np.searchsorted(np.cumsum(sizes), i, side="right")); j = i - int(sum(sizes[:k]))
        vals = []
        for sgn in (1, -1):
            with torch.no_grad():
                params[k].view(-1)[j] += sgn * eps
                vals.append(float((C * fm.solve(U, P, net, tol=tol)).sum()))
                params[k].view(-1)[j] -= sgn * eps
        out.append((vals[0] - vals[1]) / (2 * eps))
    return np.array(out)


def kink_share(net, P, U, h, thr):
    with torch.no_grad():
        z = fm.drive(h, P.values(), net, U)
    return float((z.abs() < thr).double().mean())


# ---------- A: subnetwork, every parameter
full64 = fm.load_network("baseline", torch.float64)
print(f"full baseline: {full64.N} neurons, {full64.edges} edges", flush=True)
sens = np.flatnonzero(full64.ann.superclass.astype(str).str.contains("sensory").to_numpy())
net = full64.downstream(rng.choice(sens, 300, replace=False), 6000)
K, TOL, EPS = 8, 1e-13, 1e-6
U = fm.sensory_stimuli(net, K, seed=1)
P = init_params(net, torch.float64)
C = torch.randn(net.N, K, dtype=torch.float64)
names = P.names(net.sc_names)
g_imp, h, t_imp, fi, bi = implicit_grad(net, P, U, C, TOL)
iters = fi + 50
P.zero_grad(); (C * fm.unrolled(U, P, net, iters)).sum().backward()
g_unr = torch.cat([p.grad.reshape(-1) for p in P.parameters()]).numpy()
g_fd = fd_grad(net, P, U, C, TOL, EPS, range(len(g_imp)))
eA_fd, eA_unr = rel(g_imp, g_fd), rel(g_imp, g_unr)
okA = bool(eA_fd.max() < 1e-4 and eA_unr.max() < 1e-6)
print(f"\nA. subnetwork {net.N} neurons, {net.edges} edges, {len(g_imp)} parameters, fwd {fi} / adjoint {bi} iters")
print(f"   |z| < 1e-8 share (ReLU kinks): {kink_share(net, P, U, h, 1e-8):.2e}")
print(f"   implicit vs FD:       max rel err {eA_fd.max():.2e}  (worst {names[int(eA_fd.argmax())]})")
print(f"   implicit vs unrolled: max rel err {eA_unr.max():.2e}  (worst {names[int(eA_unr.argmax())]})")
for i in np.argsort(-np.abs(g_imp))[:8]:
    print(f"     {names[i]:34s} implicit {g_imp[i]: .6e}  FD {g_fd[i]: .6e}  unrolled {g_unr[i]: .6e}")
print(f"   PASS A: {okA}", flush=True)
out["A"] = dict(neurons=net.N, edges=net.edges, params=len(g_imp), fwd_iters=fi, bwd_iters=bi,
                max_rel_fd=float(eA_fd.max()), max_rel_unrolled=float(eA_unr.max()), passed=okA,
                grads={n: [float(a), float(b), float(c)] for n, a, b, c in zip(names, g_imp, g_fd, g_unr)})

# ---------- B: full network, float64, 10 parameters by finite differences
K, TOL, EPS = 4, 1e-12, 1e-5
U = fm.sensory_stimuli(full64, K, seed=1)
P = init_params(full64, torch.float64)
C = torch.randn(full64.N, K, dtype=torch.float64)
names = P.names(full64.sc_names)
g_imp, h, t_imp, fi, bi = implicit_grad(full64, P, U, C, TOL)
counts = np.bincount(full64.sc.numpy(), minlength=len(full64.sc_names))
idx = list(range(5)) + [5 + int(i) for i in np.argsort(-counts)[:3]] + [len(names) - 2, len(names) - 1]
t = time.time(); g_fd = fd_grad(full64, P, U, C, TOL, EPS, idx); t_fd = time.time() - t
eB = rel(g_imp[idx], g_fd)
okB = bool(eB.max() < 1e-3)
print(f"\nB. full baseline {full64.N} neurons, {full64.edges} edges; implicit {t_imp:.1f}s "
      f"(fwd {fi} / adjoint {bi} iters); {2 * len(idx)} FD solves {t_fd:.0f}s")
print(f"   |z| < 1e-8 share: {kink_share(full64, P, U, h, 1e-8):.2e}")
for i, e, f in zip(idx, eB, g_fd):
    print(f"     {names[i]:34s} implicit {g_imp[i]: .6e}  FD {f: .6e}  rel err {e:.1e}")
print(f"   PASS B: {okB}", flush=True)
out["B"] = dict(fwd_iters=fi, bwd_iters=bi, seconds_implicit=t_imp, max_rel_fd=float(eB.max()), passed=okB,
                grads={names[i]: [float(g_imp[i]), float(f)] for i, f in zip(idx, g_fd)})
del full64, net, h, U, C, P

# ---------- C: full network, float32, realistic batch: time and memory
full32 = fm.load_network("baseline", torch.float32)
U = fm.sensory_stimuli(full32, 29, seed=1)
P = init_params(full32, torch.float32)
C = torch.randn(full32.N, 29)
rows = []
for rep in range(2):
    g, _, t, fi, bi = implicit_grad(full32, P, U, C, 1e-6)
    rows.append((t, fi, bi))
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30      # bytes on macOS
print(f"\nC. full baseline float32, 29 stimuli, torch {torch.__version__}, {torch.get_num_threads()} threads: "
      + "; ".join(f"fwd+bwd {t:.1f}s ({fi} fwd / {bi} adjoint iters)" for t, fi, bi in rows)
      + f"; peak RSS of this process {rss:.2f} GB")
out["C"] = dict(runs=[dict(seconds=t, fwd_iters=fi, bwd_iters=bi) for t, fi, bi in rows], peak_rss_gb=rss,
                grad_finite=bool(np.isfinite(g).all()))
out["passed"] = okA and okB
json.dump(out, open("step1_gradcheck.json", "w"), indent=1)
print(f"\nOVERALL PASS: {out['passed']}")
