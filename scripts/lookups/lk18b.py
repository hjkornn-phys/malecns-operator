"""LK-18 probe, fixed null: degree-preserving rewiring WITHIN the 152-neuron subnetwork."""
import sys, os, re
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, pandas as pd, scipy.sparse as sp, torch
import fpmodel as fm

net = fm.load_network("baseline", torch.float64); N = net.N
ann = pd.read_parquet("annotations.parquet", columns=["bodyId","superclass","instance","type"])
ann = ann[ann.superclass.notna()].sort_values("bodyId").reset_index(drop=True)
typ = net.ann.type.astype(str); inst = ann.instance.astype(str)
HD = np.flatnonzero(typ.str.match(r"^(EPG|EPGt|PEG|Delta7)$|^PEN_[ab]\(").to_numpy())
n = len(HD)
s_all = np.ones(N); s_all[net.pre] = np.sign(net.val)
A = sp.csr_matrix((s_all[net.pre]*np.abs(net.val), (net.post, net.pre)), shape=(N,N))
W = np.asarray(A[HD][:, HD].todense())
post_i, pre_i = np.nonzero(W)
print(f"{n} neurons, {len(post_i)} internal edges, density {len(post_i)/n**2:.3f}")

# ring coordinate from `instance`, held out from the test itself
def glom(s):
    m = re.search(r"_([LR])(\d)", s)
    return (m.group(1), int(m.group(2))) if m else None
lab = [glom(inst.iloc[i]) for i in HD]
ang_true = np.array([np.nan if l is None else 2*np.pi*((l[1]-1) % 8)/8 for l in lab])

def rewire(seed, sweeps=20):
    """Directed double-edge swaps: in- and out-degree exactly preserved, higher-order structure destroyed."""
    rng = np.random.default_rng(seed)
    po, pr, va = post_i.copy(), pre_i.copy(), W[post_i, pre_i].copy()
    E = len(po); present = set(zip(po.tolist(), pr.tolist()))
    for _ in range(sweeps*E):
        i, j = rng.integers(0, E, 2)
        if i == j: continue
        a, b, c, d = po[i], pr[i], po[j], pr[j]
        if (a, d) in present or (c, b) in present: continue
        present.discard((a,b)); present.discard((c,d))
        pr[i], pr[j] = d, b
        present.add((a,d)); present.add((c,b))
    M = np.zeros_like(W); M[po, pr] = va
    return M

def scores(M):
    S = (M + M.T)/2
    w, V = np.linalg.eigh(S); o = np.argsort(-np.abs(w))
    emb = V[:, o[1:3]]
    r = np.linalg.norm(emb, axis=1); r = r[r > np.percentile(r, 10)]
    annulus = float(r.mean()/(r.std()+1e-12))
    # label-based: is connection strength a cosine function of angular difference?
    ok = ~np.isnan(ang_true)
    d = ang_true[:, None] - ang_true[None, :]
    mask = ok[:, None] & ok[None, :]
    x, y = np.cos(d[mask]), S[mask]
    fourier = float(abs(np.corrcoef(x, y)[0, 1]))
    return annulus, fourier, float(np.abs(w[o[0]]))

a0, f0, e0 = scores(W)
print(f"\nREAL           annulus {a0:6.3f}   cos-Fourier corr {f0:6.3f}   |eig|max {e0:.4f}")
res = []
for sd in range(2000, 2010):
    a, f, e = scores(rewire(sd)); res.append((a, f, e))
    print(f"REWIRE-{sd}   annulus {a:6.3f}   cos-Fourier corr {f:6.3f}   |eig|max {e:.4f}")
A_, F_, E_ = map(np.array, zip(*res))
print(f"\nnull annulus {A_.min():.3f}-{A_.max():.3f} | null Fourier {F_.min():.3f}-{F_.max():.3f} "
      f"| null |eig|max {E_.min():.4f}-{E_.max():.4f}")
print(f"REAL beats all 10 seeds:  annulus {bool((a0>A_).all())}   Fourier {bool((f0>F_).all())}")
