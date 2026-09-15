"""Step 1b: can training through the fixed point recover parameters we set ourselves?

Teacher: fpmodel with known parameters generates responses to sensory stimuli. Student: same model, raw
parameters perturbed by N(0, 0.7), trained with Adam on the teacher's responses. Training and held-out
stimuli are different random sensory groups.

Modes (argv[1]):
  few   alpha x5, s, b trained; one gamma pinned to the teacher (alpha * gamma share a scale)      7 params
  tens  alpha x5, gamma per superclass, s, b; gamma of the largest superclass pinned (same reason)
Network (argv[2]): "sub" = 6,000 neurons downstream of sensory seeds, loss on all non-sensory neurons;
                   "full" = baseline k>=3, loss on readout neurons (descending, motor, efferent, endocrine).

Pass criteria, fixed before running:
  held-out loss (normalised MSE) <= 1e-3 x its value at initialisation, and
  few:  |alpha - alpha*| < 0.01 each, |s - s*| / s* < 0.01, |b - b*| < 0.005
  tens: the same for alpha, s, b; gamma reported per superclass with its loss sensitivity, and gammas whose
        superclass carries under 1e-6 of the loss gradient at the teacher are labelled not identifiable
        instead of counted.
Run from data/ after retention.py:   step1_recovery.py few|tens sub|full [steps]
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import fpmodel as fm

MODE, WHERE = sys.argv[1], sys.argv[2]
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else (300 if WHERE == "sub" else 60)
DTYPE = torch.float64 if WHERE == "sub" else torch.float32
TOL = 1e-10 if WHERE == "sub" else 1e-6
K_TRAIN, K_TEST, LR = 16, 8, 0.05
torch.manual_seed(0)
rng = np.random.default_rng(0)

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


def loss(P, U, Y):
    h = fm.solve(U, P, net, TOL)[target_rows]
    return ((h - Y) ** 2).mean() / (Y ** 2).mean()


def values(P):
    a, g, s, b = (v.detach().double().numpy() for v in P.values())
    return a, g, float(s), float(b)


# which gammas can the loss see at all: the loss gradient is zero at the teacher, so use a random
# projection of the loss neurons' responses, averaged over a few projections (|d<r, h>/dc|)
sens_c = np.zeros(teacher.c.numel())
for _ in range(4):
    teacher.zero_grad()
    h_t = fm.solve(U_tr, teacher, net, TOL)[target_rows]
    (h_t * torch.randn_like(h_t)).sum().backward()
    sens_c += teacher.c.grad.abs().double().numpy()
sens_share = sens_c / max(sens_c.sum(), 1e-30)

with torch.no_grad():
    l0_te = float(loss(student, U_te, Y_te))
opt = torch.optim.Adam(student.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
hist = []
t0 = time.time()
for step in range(1, STEPS + 1):
    opt.zero_grad()
    l = loss(student, U_tr, Y_tr)
    l.backward()
    student.c.grad.view(-1)[pin] = 0.0
    opt.step(); sched.step()
    a, g, s, b = values(student); at, gt, st, bt = values(teacher)
    rec = dict(step=step, train=float(l), alpha_err=float(np.abs(a - at).max()), s_rel=abs(s - st) / st,
               b_err=abs(b - bt), fwd=fm.STATS["fwd_iters"], bwd=fm.STATS["bwd_iters"], sec=time.time() - t0)
    hist.append(rec)
    if step % max(1, STEPS // 15) == 0 or step == 1:
        print(f"step {step:4d} train {rec['train']:.3e} | max|dalpha| {rec['alpha_err']:.4f} s rel {rec['s_rel']:.4f} "
              f"|db| {rec['b_err']:.4f} | iters {rec['fwd']}/{rec['bwd']} | {rec['sec']:.0f}s", flush=True)

with torch.no_grad():
    l_te = float(loss(student, U_te, Y_te))
a, g, s, b = values(student); at, gt, st, bt = values(teacher)
ok_loss = l_te <= 1e-3 * l0_te
ok_params = bool(np.all(np.abs(a - at) < 0.01) and abs(s - st) / st < 0.01 and abs(b - bt) < 0.005)
print(f"\n{MODE} / {WHERE}: {net.N} neurons, {net.edges} edges, {len(target_rows)} loss neurons, {STEPS} steps, "
      f"{time.time() - t0:.0f}s")
print(f"held-out loss {l0_te:.3e} -> {l_te:.3e}  (ratio {l_te / l0_te:.1e}; pass <= 1e-3: {ok_loss})")
print("alpha  teacher", np.round(at, 4), "\n       student", np.round(a, 4))
print(f"s teacher {st:.4f} student {s:.4f} | b teacher {bt:.4f} student {b:.4f} | params pass: {ok_params}")
gam = []
if per_sc:
    for i in np.argsort(-counts):
        ident = sens_share[i] >= 1e-6 or i == pin
        gam.append(dict(superclass=net.sc_names[i], neurons=int(counts[i]), teacher=float(gt[i]), student=float(g[i]),
                        err=float(abs(g[i] - gt[i])), grad_share=float(sens_share[i]), pinned=bool(i == pin),
                        identifiable=bool(ident)))
        print(f"  gamma {net.sc_names[i]:24s} n={counts[i]:6d} teacher {gt[i]:.4f} student {g[i]:.4f} "
              f"err {abs(g[i] - gt[i]):.4f} grad share {sens_share[i]:.1e}{' PINNED' if i == pin else ''}"
              f"{'' if ident else ' (not identifiable)'}")
    idg = [x for x in gam if x["identifiable"] and not x["pinned"]]
    print(f"identifiable gammas within 0.01: {sum(x['err'] < 0.01 for x in idg)}/{len(idg)}")
passed = ok_loss and ok_params
print(f"PASS: {passed}")
json.dump(dict(mode=MODE, where=WHERE, neurons=net.N, edges=net.edges, loss_neurons=len(target_rows), steps=STEPS,
               heldout_init=l0_te, heldout_final=l_te, alpha_teacher=at.tolist(), alpha_student=a.tolist(),
               s=[st, s], b=[bt, b], gamma=gam, history=hist, passed=passed),
          open(f"step1_recovery_{MODE}_{WHERE}.json", "w"), indent=1)
