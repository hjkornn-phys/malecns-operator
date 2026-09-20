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
    for p in sel(pre_kind):
        sp_, np_ = G[p][0]
        for e in EPG:
            se, ne = G[e][0]
            if se != sp_: continue
            w = M[e, p]
            if w == 0: continue
            hist.setdefault(int(ne) - int(np_), []).append(w)
    return {d: float(np.sum(v)) for d, v in sorted(hist.items())}

def d7_reach(M):
    """Delta7 spans several glomeruli; the spread between the ones it touches, in glomerulus units."""
    sp_ = []
    for i in range(len(HD)):
        if not kind[i].startswith("Delta7") or len(G[i]) < 2: continue
        pos = [(0 if s == "L" else 1) * 9 + int(n) for s, n in G[i]]
        sp_.append(max(pos) - min(pos))
    return float(np.mean(sp_)), float(np.std(sp_))

for name in ("PEN_a", "PEN_b"):
    h = signed_offset(W, name)
    tot = sum(abs(v) for v in h.values()) + 1e-12
    print(f"REAL {name}->EPG signed offset (share of weight):")
    print("   " + "  ".join(f"{d:+d}:{v/tot:5.2f}" for d, v in h.items() if abs(v)/tot > .01))
    com = sum(d*abs(v) for d, v in h.items())/tot
    print(f"   centre of mass {com:+.3f}")
m, s = d7_reach(W)
print(f"\nDelta7 glomerulus spread: mean {m:.2f} +- {s:.2f}  (anti-phase would be ~8-9)")

print("\n--- nulls ---")
for sd in range(2000, 2005):
    M = rewire(sd)
    coms = []
    for name in ("PEN_a", "PEN_b"):
        h = signed_offset(M, name); tot = sum(abs(v) for v in h.values()) + 1e-12
        coms.append(sum(d*abs(v) for d, v in h.items())/tot)
    print(f"REWIRE-{sd}  PEN_a com {coms[0]:+.3f}  PEN_b com {coms[1]:+.3f}")
