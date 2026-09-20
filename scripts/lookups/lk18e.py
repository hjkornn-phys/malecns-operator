"""The actual ring machinery: PEN->EPG offset (rotation) and Delta7 anti-phase reach (global inhibition)."""
import sys, os, re
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, pandas as pd, scipy.sparse as sp, torch
import fpmodel as fm

net = fm.load_network("baseline", torch.float64); N = net.N
ann = pd.read_parquet("annotations.parquet", columns=["bodyId","superclass","instance","type"])
ann = ann[ann.superclass.notna()].sort_values("bodyId").reset_index(drop=True)
typ = net.ann.type.astype(str); inst = ann.instance.astype(str)
HD = np.flatnonzero(typ.str.match(r"^(EPG|EPGt|PEG|Delta7)$|^PEN_[ab]\(").to_numpy())
gloms = lambda s: re.findall(r"([LR])(\d)", s.split(")_")[-1])
G = [gloms(inst.iloc[i]) for i in HD]; kind = [typ.iloc[i] for i in HD]
s_all = np.ones(N); s_all[net.pre] = np.sign(net.val)
A = sp.csr_matrix((s_all[net.pre]*np.abs(net.val), (net.post, net.pre)), shape=(N,N))
W = np.asarray(A[HD][:, HD].todense()); post_i, pre_i = np.nonzero(W)

def rewire(seed, sweeps=20):
    rng = np.random.default_rng(seed)
    po, pr, va = post_i.copy(), pre_i.copy(), W[post_i, pre_i].copy()
    E = len(po); pres = set(zip(po.tolist(), pr.tolist()))
    for _ in range(sweeps*E):
        i, j = rng.integers(0, E, 2)
        if i == j: continue
        a,b,c,d = po[i], pr[i], po[j], pr[j]
        if (a,d) in pres or (c,b) in pres: continue
        pres.discard((a,b)); pres.discard((c,d)); pr[i], pr[j] = d, b
        pres.add((a,d)); pres.add((c,b))
    M = np.zeros_like(W); M[po, pr] = va
    return M

sel = lambda k: [i for i in range(len(HD)) if kind[i].startswith(k) and len(G[i]) == 1]
EPG = sel("EPG")
def signed_offset(M, pre_kind):
    """For each PEN->EPG edge within a hemisphere, the signed glomerulus offset (EPG index - PEN index).
    A rotation mechanism puts PEN_a and PEN_b on OPPOSITE signs."""
    hist = {}
sel = lambda k: [i for i in range(len(HD)) if kind[i].startswith(k) and len(G[i]) == 1]
EPG = sel("EPG")

def com_by_side(M, pre_kind, side):
    h = {}
    for p in sel(pre_kind):
        sp_, np_ = G[p][0]
        if sp_ != side: continue
        for e in EPG:
            se, ne = G[e][0]
            if se != side: continue
            w = M[e, p]
            if w == 0: continue
            h.setdefault(int(ne) - int(np_), []).append(w)
    if not h: return float("nan")
    tot = sum(abs(np.sum(v)) for v in h.values()) + 1e-12
    return sum(d * abs(np.sum(v)) for d, v in h.items()) / tot

print("PEN->EPG offset centre of mass, BY HEMISPHERE.")
print("L and R are mirror images: the same index sign is OPPOSITE rotation in heading space,")
print("so a rotation mechanism needs the two hemispheres to DIFFER.\n")
for name in ("PEN_a", "PEN_b"):
    l, r = com_by_side(W, name, "L"), com_by_side(W, name, "R")
    print(f"  REAL   {name:6s}  L {l:+.3f}   R {r:+.3f}   |L-R| = {abs(l-r):.3f}")
print()
gaps = []
for sd in range(2000, 2010):
    M = rewire(sd); row, g = [], []
    for name in ("PEN_a", "PEN_b"):
        l, r = com_by_side(M, name, "L"), com_by_side(M, name, "R")
        row.append(f"{name} |L-R| {abs(l-r):.3f}"); g.append(abs(l - r))
    gaps.append(g); print(f"  REWIRE-{sd}  " + "   ".join(row))
gaps = np.array(gaps)
for k, name in enumerate(("PEN_a", "PEN_b")):
    l, r = com_by_side(W, name, "L"), com_by_side(W, name, "R")
    print(f"\n{name}: REAL |L-R| {abs(l-r):.3f} vs null {gaps[:,k].min():.3f}-{gaps[:,k].max():.3f}"
          f"  -> beats all 10: {bool(abs(l-r) > gaps[:,k].max())}")
