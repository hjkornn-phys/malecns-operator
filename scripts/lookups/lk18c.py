"""LK-18, second pass. No angle assumption: is connectivity a function of glomerulus index difference?
The PB glomeruli are physically ordered L9..L1 | R1..R9, so the index IS the ring coordinate."""
import sys, os, re
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, pandas as pd, scipy.sparse as sp, torch
import fpmodel as fm

net = fm.load_network("baseline", torch.float64); N = net.N
ann = pd.read_parquet("annotations.parquet", columns=["bodyId","superclass","instance","type"])
ann = ann[ann.superclass.notna()].sort_values("bodyId").reset_index(drop=True)
typ = net.ann.type.astype(str); inst = ann.instance.astype(str)
HD = np.flatnonzero(typ.str.match(r"^(EPG|EPGt|PEG|Delta7)$|^PEN_[ab]\(").to_numpy())

# parse EVERY glomerulus in the instance, not just the first (Delta7 spans several)
def gloms(s):
    tail = s.split(")_")[-1]
    return re.findall(r"([LR])(\d)", tail)
G = [gloms(inst.iloc[i]) for i in HD]
kind = [typ.iloc[i] for i in HD]
print("glomeruli parsed per cell type:")
for k in sorted(set(kind)):
    c = [len(G[i]) for i in range(len(HD)) if kind[i] == k]
    print(f"  {k:14s} n={len(c):3d}  glomeruli/cell: min {min(c)} max {max(c)} mean {np.mean(c):.2f}")

s_all = np.ones(N); s_all[net.pre] = np.sign(net.val)
A = sp.csr_matrix((s_all[net.pre]*np.abs(net.val), (net.post, net.pre)), shape=(N,N))
W = np.asarray(A[HD][:, HD].todense())
post_i, pre_i = np.nonzero(W)

def rewire(seed, sweeps=20):
    rng = np.random.default_rng(seed)
    po, pr, va = post_i.copy(), pre_i.copy(), W[post_i, pre_i].copy()
    E = len(po); pres = set(zip(po.tolist(), pr.tolist()))
    for _ in range(sweeps*E):
        i, j = rng.integers(0, E, 2)
        if i == j: continue
        a, b, c, d = po[i], pr[i], po[j], pr[j]
        if (a,d) in pres or (c,b) in pres: continue
        pres.discard((a,b)); pres.discard((c,d)); pr[i], pr[j] = d, b
        pres.add((a,d)); pres.add((c,b))
    M = np.zeros_like(W); M[po, pr] = va
    return M

# EPG only, one glomerulus each, within one hemisphere: is W a function of index difference?
EPG = [i for i in range(len(HD)) if kind[i] == "EPG" and len(G[i]) == 1]
print(f"\nEPG with exactly one glomerulus: {len(EPG)}")
for side in "LR":
    idx = [i for i in EPG if G[i][0][0] == side]
    num = np.array([int(G[i][0][1]) for i in idx])
    print(f"  side {side}: {len(idx)} cells, glomeruli {sorted(set(num.tolist()))}")

def profile(M):
    """Mean connection strength as a function of |glomerulus difference|, within hemisphere, EPG->EPG."""
    out = {}
    for side in "LR":
        idx = [i for i in EPG if G[i][0][0] == side]
        num = np.array([int(G[i][0][1]) for i in idx])
        for a in range(len(idx)):
            for b in range(len(idx)):
                if a == b: continue
                d = abs(int(num[a]) - int(num[b]))
                out.setdefault(d, []).append(M[idx[a], idx[b]])
    return {d: float(np.mean(v)) for d, v in sorted(out.items())}

def modulation(M):
    p = profile(M)
    v = np.array([p[d] for d in sorted(p)])
    return float(v.std() / (np.abs(v).mean() + 1e-12)), p

m0, p0 = modulation(W)
print(f"\nREAL  EPG->EPG mean weight by |glomerulus difference|:")
print("   " + "  ".join(f"d{d}:{p0[d]:+.4f}" for d in sorted(p0)))
print(f"   modulation (sd/mean) = {m0:.3f}")
nulls = []
for sd in range(2000, 2010):
    m, _ = modulation(rewire(sd)); nulls.append(m)
nulls = np.array(nulls)
print(f"\nREWIRE x10 modulation: {nulls.min():.3f}-{nulls.max():.3f}  (mean {nulls.mean():.3f})")
print(f"REAL beats all 10: {bool((m0 > nulls).all())}")
