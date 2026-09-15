"""Steady-state rate model with trainable parameters and implicit gradients at the fixed point.

    h* = ReLU(z),   z = F(h*) = gamma[sc(post)] * (W0 @ (alpha[nt(pre)] * h*)) + s * U + b
    W0[post, pre] = sign(pre) * w / in_tot_full[post]        (count-fraction weights; |W0| row sums <= 1)

    alpha = 0.99 * sigmoid(a)   one gain per presynaptic transmitter class (NT_CLASSES)
    gamma = sigmoid(c)          one gain per postsynaptic superclass (or a single one)
    s = softplus(s_raw)         input gain;   b   tonic drive
Every effective gain alpha * gamma is below 0.99, so the iteration contracts for any parameter values.
A global gain g is absorbed into alpha * gamma, and a threshold theta would only enter as b - theta, so
neither is a separate parameter: they are not identifiable next to these.

Backward, no unrolling: with D = 1[z > 0] and v = dL/dh*, solve
    nu = D v + D J^T nu,     J^T nu = alpha[nt] * (W0^T @ (gamma[sc] * nu)),
then dL/dp = nu^T dF/dp evaluated at h* held fixed.
"""
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as tnf

warnings.filterwarnings("ignore", message=".*Sparse.*")

NT_CLASSES = ["acetylcholine", "gaba", "glutamate", "histamine", "other"]   # other: DA, 5-HT, OA, unclear, missing
SIGN = {"acetylcholine": 1, "dopamine": 1, "serotonin": 1, "octopamine": 1,
        "gaba": -1, "glutamate": -1, "histamine": -1}
STATS = {"fwd_iters": 0, "bwd_iters": 0}


def _torch_csr(A, dtype):
    A = A.tocsr(); A.sort_indices()
    return torch.sparse_csr_tensor(torch.from_numpy(A.indptr.astype(np.int64)),
                                   torch.from_numpy(A.indices.astype(np.int64)),
                                   torch.from_numpy(A.data).to(dtype), size=A.shape)


class Net:
    def __init__(self, ann, pre, post, val, ntc, dtype):
        self.ann, self.N, self.dtype = ann.reset_index(drop=True), len(ann), dtype
        self.pre, self.post, self.val, self.ntc = pre, post, val, ntc
        cat = pd.Categorical(self.ann.superclass.astype(str))
        self.sc_names = list(cat.categories)
        self.sc = torch.from_numpy(cat.codes.astype(np.int64))
        self.nt = torch.from_numpy(ntc.astype(np.int64))
        A = sp.csr_matrix((val, (post, pre)), shape=(self.N, self.N))
        self.AT_sp = A.T.tocsr()
        self.W0, self.W0T = _torch_csr(A, dtype), _torch_csr(self.AT_sp, dtype)
        self.edges = int(A.nnz)

    def sub(self, keep):
        keep = np.sort(np.asarray(keep))
        new = np.full(self.N, -1); new[keep] = np.arange(len(keep))
        m = (new[self.pre] >= 0) & (new[self.post] >= 0)
        return Net(self.ann.iloc[keep], new[self.pre[m]], new[self.post[m]], self.val[m], self.ntc[keep], self.dtype)

    def downstream(self, seeds, n_keep):
        keep, frontier = list(dict.fromkeys(int(s) for s in seeds)), list(seeds)
        seen = set(keep)
        while len(keep) < n_keep and frontier:
            nxt = [int(j) for j in np.unique(self.AT_sp[frontier].indices) if int(j) not in seen]
            nxt = nxt[: n_keep - len(keep)]
            keep += nxt; seen.update(nxt); frontier = nxt
        return self.sub(keep)


def load_network(rule="baseline", dtype=torch.float64, data="."):
    ann = pd.read_parquet(f"{data}/annotations.parquet", columns=["bodyId", "superclass", "class", "type"])
    ann = ann[ann.superclass.notna()].sort_values("bodyId").reset_index(drop=True)
    N = len(ann)
    nt = pd.read_parquet(f"{data}/nt.parquet", columns=["body", "consensus_nt"]).set_index("body").consensus_nt
    cons = ann.bodyId.map(nt).fillna("missing").astype(str).to_numpy()
    sign = np.array([SIGN.get(c, 1) for c in cons], np.float64)
    ntc = np.array([NT_CLASSES.index(c) if c in NT_CLASSES[:4] else 4 for c in cons])
    d = np.load(f"{data}/nn_edges.npz"); pre, post, w = d["pre"], d["post"], d["w"].astype(np.float64)
    in_tot = np.bincount(post, weights=w, minlength=N)
    out_tot = np.bincount(pre, weights=w, minlength=N)
    m = w >= 3
    if rule == "OR 1%": m &= (w / in_tot[post] >= 0.01) | (w / out_tot[pre] >= 0.01)
    elif rule != "baseline": raise ValueError(rule)
    return Net(ann, pre[m], post[m], sign[pre[m]] * w[m] / in_tot[post[m]], ntc, dtype)


def sensory_stimuli(net, K, seed=0, frac=0.02):
    rng = np.random.default_rng(seed)
    sens = np.flatnonzero(net.ann.superclass.astype(str).str.contains("sensory").to_numpy())
    U = torch.zeros(net.N, K, dtype=net.dtype)
    for j in range(K):
        U[rng.choice(sens, min(len(sens), max(10, int(len(sens) * frac))), replace=False), j] = 1.0
    return U


class Params(torch.nn.Module):
    def __init__(self, n_sc, per_sc=True, dtype=torch.float64):
        super().__init__()
        self.a = torch.nn.Parameter(torch.zeros(len(NT_CLASSES), dtype=dtype))
        self.c = torch.nn.Parameter(torch.zeros(n_sc if per_sc else 1, dtype=dtype))
        self.s_raw = torch.nn.Parameter(torch.zeros((), dtype=dtype))
        self.b = torch.nn.Parameter(torch.zeros((), dtype=dtype))

    def values(self):
        return 0.99 * torch.sigmoid(self.a), torch.sigmoid(self.c), tnf.softplus(self.s_raw), self.b

    def flat(self):
        return torch.cat([p.detach().reshape(-1) for p in self.parameters()])

    def names(self, sc_names):
        return ([f"alpha:{n}" for n in NT_CLASSES] + [f"gamma:{n}" for n in (sc_names if self.c.numel() > 1 else ["all"])]
                + ["s", "b"])


class SpMV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, A, AT):
        ctx.AT = AT
        return A @ x

    @staticmethod
    def backward(ctx, g):
        return ctx.AT @ g, None, None


def drive(h, pv, net, U):
    alpha, gamma, s, b = pv
    y = SpMV.apply(alpha[net.nt].unsqueeze(1) * h, net.W0, net.W0T)
    g = gamma[net.sc].unsqueeze(1) if gamma.numel() > 1 else gamma
    return g * y + s * U + b


class FixedPoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, U, net, tol, maxit, *pv):
        with torch.no_grad():
            h = torch.zeros_like(U)
            for it in range(1, maxit + 1):
                hn = torch.relu(drive(h, pv, net, U))
                delta = (hn - h).abs().max().item(); h = hn
                if delta < tol: break
            else:
                raise RuntimeError(f"forward did not converge in {maxit} iterations (last change {delta:.3g})")
        STATS["fwd_iters"] = it
        ctx.save_for_backward(h, U, *pv)
        ctx.net, ctx.tol, ctx.maxit = net, tol, maxit
        return h

    @staticmethod
    def backward(ctx, v):
        h, U, *pv = ctx.saved_tensors
        net = ctx.net
        with torch.no_grad():
            D = (drive(h, pv, net, U) > 0).to(v.dtype)
            alpha, gamma = pv[0], pv[1]
            a = alpha[net.nt].unsqueeze(1)
            g = gamma[net.sc].unsqueeze(1) if gamma.numel() > 1 else gamma
            Dv = D * v
            nu, scale = Dv.clone(), max(Dv.abs().max().item(), 1e-30)
            for it in range(1, ctx.maxit + 1):
                nn_ = Dv + D * (a * (net.W0T @ (g * nu)))
                delta = (nn_ - nu).abs().max().item(); nu = nn_
                if delta < ctx.tol * scale: break
            else:
                raise RuntimeError(f"adjoint did not converge in {ctx.maxit} iterations")
        STATS["bwd_iters"] = it
        with torch.enable_grad():
            leaves = [p.detach().requires_grad_(True) for p in pv]
            z = drive(h, leaves, net, U)
            grads = torch.autograd.grad(z, leaves, grad_outputs=nu, allow_unused=True)
        return (None, None, None, None, *grads)


def solve(U, P, net, tol=1e-6, maxit=5000):
    return FixedPoint.apply(U, net, tol, maxit, *P.values())


def unrolled(U, P, net, iters):
    h = torch.zeros_like(U)
    for _ in range(iters):
        h = torch.relu(drive(h, P.values(), net, U))
    return h
