"""Step 1c: when recovery misses the parameters, is it the optimizer or identifiability?

Same teacher, stimuli, loss neurons and student start as step1_recovery.py (same seeds, same draw order).
  1. Sensitivity at the teacher: J = d(loss-neuron responses)/d(trainable raw params) by central finite
     differences (float64), eigen-decomposition of J^T J (normalised by the output norm).
  2. Levenberg-Marquardt from the recovery run's student start, using the same J by finite differences.

Interpretation, fixed before running:
  LM loss < 1e-12 and parameters pass step1_recovery's criteria   -> identifiable; Adam was the problem.
  LM loss < 1e-12 but parameters fail, smallest eigenvalues tiny  -> not identifiable from these data:
                                                                     judge recovery by held-out predictions.
  LM loss stays well above 1e-12                                  -> neither answer; report as is.
Run from data/:   step1_identify.py few|tens sub|full [lm_iters]
"full" = baseline k>=3, loss on readout neurons, as in step1_recovery.py; only "few" is affordable there
(2 x 7 full-network float64 solves per Jacobian).
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import fpmodel as fm

MODE, WHERE = sys.argv[1], sys.argv[2]
LM_ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 40
assert WHERE == "sub" or MODE == "few", "finite-difference Jacobians on the full network: few mode only"
DTYPE, TOL, EPS, K_TRAIN, K_TEST = torch.float64, 1e-12, 1e-6, 16, 8
torch.manual_seed(0)
rng = np.random.default_rng(0)

# ---- identical setup to step1_recovery.py
net = fm.load_network("baseline", DTYPE)
if WHERE == "sub":
    sens = np.flatnonzero(net.ann.superclass.astype(str).str.contains("sensory").to_numpy())
    net = net.downstream(rng.choice(sens, 300, replace=False), 6000)
sc = net.ann.superclass.astype(str)
if WHERE == "sub":
    target_rows = torch.from_numpy(np.flatnonzero(~sc.str.contains("sensory").to_numpy()))
else:
    target_rows = torch.from_numpy(np.flatnonzero((sc.eq("descending_neuron") | sc.str.endswith("_motor")
                                                   | sc.str.contains("efferent") | sc.str.endswith("_endocrine")).to_numpy()))
U_tr = fm.sensory_stimuli(net, K_TRAIN, seed=11)
U_te = fm.sensory_stimuli(net, K_TEST, seed=12)
per_sc = MODE == "tens"
n_sc = len(net.sc_names)
teacher = fm.Params(n_sc, per_sc=per_sc, dtype=DTYPE)
with torch.no_grad():
    teacher.a.copy_(torch.tensor(rng.normal(1.0, 0.6, 5)))
    teacher.c.copy_(torch.tensor(rng.normal(1.5, 0.6, teacher.c.numel())))
    teacher.s_raw.fill_(float(rng.normal(0.5, 0.3))); teacher.b.fill_(float(rng.uniform(0.05, 0.2)))
    Y_tr = fm.solve(U_tr, teacher, net, TOL)[target_rows]
    Y_te = fm.solve(U_te, teacher, net, TOL)[target_rows]
counts = np.bincount(net.sc.numpy(), minlength=n_sc)
pin = int(np.argmax(counts)) if per_sc else 0
student = fm.Params(n_sc, per_sc=per_sc, dtype=DTYPE)
with torch.no_grad():
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        ps.copy_(pt + torch.randn_like(pt) * 0.7)
    student.c.view(-1)[pin] = teacher.c.view(-1)[pin]
# ----

names_all = teacher.names(net.sc_names)
sizes = [p.numel() for p in teacher.parameters()]
pin_flat = sizes[0] + pin
train_idx = [i for i in range(sum(sizes)) if i != pin_flat]
names = [names_all[i] for i in train_idx]


def get_raw(P): return P.flat().numpy().copy()


def set_raw(P, x):
    with torch.no_grad():
        o = 0
        for p in P.parameters():
            p.copy_(torch.from_numpy(x[o:o + p.numel()]).reshape(p.shape)); o += p.numel()


def outputs(P, U):
    with torch.no_grad():
        return fm.solve(U, P, net, TOL)[target_rows].reshape(-1).numpy().copy()


y_tr, y_te = Y_tr.reshape(-1).numpy(), Y_te.reshape(-1).numpy()
norm_tr, norm_te = float((y_tr ** 2).mean()), float((y_te ** 2).mean())
work = fm.Params(n_sc, per_sc=per_sc, dtype=DTYPE)


def jacobian(x):
    J = np.zeros((len(y_tr), len(train_idx)))
    for k, i in enumerate(train_idx):
        xp, xm = x.copy(), x.copy(); xp[i] += EPS; xm[i] -= EPS
        set_raw(work, xp); fp = outputs(work, U_tr)
        set_raw(work, xm); fm_ = outputs(work, U_tr)
        J[:, k] = (fp - fm_) / (2 * EPS)
    return J


def loss_at(x, U, y, norm):
    set_raw(work, x); return float(((outputs(work, U) - y) ** 2).mean() / norm)


def phys(x):
    set_raw(work, x)
    a, g, s, b = (v.detach().numpy() for v in work.values())
    return a, g, float(s), float(b)


# 1. sensitivity spectrum at the teacher
t0 = time.time()
x_t = get_raw(teacher)
J = jacobian(x_t) / np.sqrt(len(y_tr) * norm_tr)
H = J.T @ J
ev, evec = np.linalg.eigh(H)
print(f"{MODE}/{WHERE}: {len(train_idx)} trainable params, {len(y_tr)} outputs; Jacobian {time.time() - t0:.0f}s")
print("J^T J eigenvalues (ascending):", np.array2string(ev, precision=2, separator=", "))
print(f"condition number {ev[-1] / max(ev[0], 1e-300):.2e}")
for q in range(min(3, len(ev))):
    comp = sorted(zip(np.abs(evec[:, q]), names, evec[:, q]), reverse=True)[:4]
    print(f"  softest direction {q} (eig {ev[q]:.2e}): " + ", ".join(f"{n} {c:+.2f}" for _, n, c in comp))

# 2. Levenberg-Marquardt from the recovery start
x = get_raw(student)
lam = 1e-3
at, gt, st, bt = phys(x_t)
hist = []
l = loss_at(x, U_tr, y_tr, norm_tr)
l0_te = loss_at(x, U_te, y_te, norm_te)
for it in range(1, LM_ITERS + 1):
    set_raw(work, x)
    r = (outputs(work, U_tr) - y_tr) / np.sqrt(len(y_tr) * norm_tr)
    Jx = jacobian(x) / np.sqrt(len(y_tr) * norm_tr)
    A, gvec = Jx.T @ Jx, Jx.T @ r
    while True:
        step = -np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), gvec)
        xn = x.copy(); xn[train_idx] += step
        ln = loss_at(xn, U_tr, y_tr, norm_tr)
        if ln < l:
            x, l, lam = xn, ln, max(lam / 3, 1e-12); break
        lam *= 4
        if lam > 1e12: break
    a, g, s, b = phys(x)
    hist.append(dict(it=it, loss=l, lam=lam, alpha_err=float(np.abs(a - at).max()), s_rel=abs(s - st) / st, b_err=abs(b - bt)))
    print(f"LM {it:3d} loss {l:.3e} lam {lam:.1e} | max|dalpha| {hist[-1]['alpha_err']:.2e} s rel {hist[-1]['s_rel']:.2e} "
          f"|db| {hist[-1]['b_err']:.2e} | {time.time() - t0:.0f}s", flush=True)
    if l < 1e-20 or lam > 1e12: break

a, g, s, b = phys(x)
l_te = loss_at(x, U_te, y_te, norm_te)
ok_params = bool(np.all(np.abs(a - at) < 0.01) and abs(s - st) / st < 0.01 and abs(b - bt) < 0.005)
if per_sc:
    gerr = np.abs(g - gt)
    print("gamma err (teacher order by size):", {net.sc_names[i]: round(float(gerr[i]), 4) for i in np.argsort(-counts)})
verdict = ("identifiable; Adam was the problem" if l < 1e-12 and ok_params else
           "not identifiable from these data" if l < 1e-12 else "undecided: LM did not reach 1e-12")
print(f"\nLM final train loss {l:.3e}; held-out {l0_te:.3e} -> {l_te:.3e}; params pass {ok_params}")
print("alpha teacher", np.round(at, 4), "LM", np.round(a, 4), f"| s {st:.4f} / {s:.4f} | b {bt:.4f} / {b:.4f}")
print("VERDICT:", verdict)
json.dump(dict(mode=MODE, where=WHERE, eigenvalues=ev.tolist(), names=names, softest=evec[:, :3].tolist(),
               lm=hist, heldout=[l0_te, l_te], params_pass=ok_params, verdict=verdict,
               alpha=[at.tolist(), a.tolist()], s=[st, s], b=[bt, b],
               gamma=[gt.tolist(), g.tolist()] if per_sc else None),
          open(f"step1_identify_{MODE}_{WHERE}.json", "w"), indent=1)
