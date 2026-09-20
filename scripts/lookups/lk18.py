"""LK-18 probe: is there a ring in the connectivity itself? Labels held out as validation."""
import sys, os, re
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, pandas as pd, scipy.sparse as sp, torch
import fpmodel as fm

net = fm.load_network("baseline", torch.float64); N = net.N
ann = pd.read_parquet("annotations.parquet", columns=["bodyId", "superclass", "instance", "type"])
ann = ann[ann.superclass.notna()].sort_values("bodyId").reset_index(drop=True)
assert (ann.bodyId.to_numpy() == net.ann.bodyId.to_numpy()).all()
typ = net.ann.type.astype(str); inst = ann.instance.astype(str)
HD = np.flatnonzero(typ.str.match(r"^(EPG|EPGt|PEN_|PEG|Delta7)$|^PEN_[ab]\(").to_numpy())
print(f"head-direction subnetwork: {len(HD)} neurons")

# held-out validation labels: PB glomerulus index from `instance`, never used to build the test
def glom(s):
    m = re.search(r"_([LR])(\d)(?:[LR]\d)*(?:_[LR])?$", s)
    if not m: return None
    side, n = m.group(1), int(m.group(2))
    return (side, n)
lab = [glom(inst.iloc[i]) for i in HD]
ok = np.array([l is not None for l in lab])
print(f"glomerulus label parsed for {ok.sum()}/{len(HD)}")

s_all = np.ones(N); s_all[net.pre] = np.sign(net.val)

def submatrix(seed):
    pre = net.pre if seed is None else np.random.default_rng(seed).permutation(net.pre)
    A = sp.csr_matrix((s_all[pre] * np.abs(net.val), (net.post, pre)), shape=(N, N))
    return np.asarray(A[HD][:, HD].todense())

def ring_score(W):
    """No labels used. Symmetrise, take the top two non-trivial eigenvectors, embed, and ask how
    circular the cloud is: a ring gives a thin annulus, a blob gives a filled disc."""
    S = (W + W.T) / 2
    w, V = np.linalg.eigh(S)
    order = np.argsort(-np.abs(w))
    emb = V[:, order[1:3]]                       # skip the leading (mean) mode
    emb = emb / (np.linalg.norm(emb, axis=0, keepdims=True) + 1e-12)
    r = np.linalg.norm(emb, axis=1)
    r = r[r > np.percentile(r, 10)]              # drop the few near-origin points
    annulus = float(r.mean() / (r.std() + 1e-12))   # high = points sit at one radius = a ring
    ang = np.arctan2(emb[:, 1], emb[:, 0])
    gaps = np.diff(np.sort(ang)); gaps = np.append(gaps, 2*np.pi - gaps.sum())
    uniform = float(gaps.mean() / (gaps.std() + 1e-12))  # high = evenly spread around the circle
    return annulus, uniform, emb, np.abs(w[order[:6]])

W = submatrix(None)
a0, u0, emb0, ev0 = ring_score(W)
print(f"\nREAL       annulus {a0:6.3f}   angular-uniformity {u0:6.3f}   |eig| top6 {np.round(ev0,4)}")
for sd in (1000, 1001, 1002, 1003, 1004):
    a, u, _, ev = ring_score(submatrix(sd))
    print(f"SHUF-{sd}  annulus {a:6.3f}   angular-uniformity {u:6.3f}   |eig| top6 {np.round(ev,4)}")

# validation, only now: does the embedding angle track the real glomerulus index?
ang = np.arctan2(emb0[:, 1], emb0[:, 0])
sub = [(l, ang[i]) for i, l in enumerate(lab) if l is not None]
for side in "LR":
    xs = [(n, a) for (s, n), a in sub if s == side]
    if len(xs) < 4: continue
    n_, a_ = np.array([x[0] for x in xs]), np.array([x[1] for x in xs])
    c = np.abs(np.corrcoef(np.cos(a_), np.cos(2*np.pi*n_/9))[0, 1])
    print(f"validation: side {side} n={len(xs)}  |corr(embedding angle, glomerulus index)| = {c:.3f}")
