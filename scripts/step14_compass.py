"""Step 14: does the wiring alternate hemispheres around the compass ring?

Not "does a bump persist". That question cannot be asked at this project's settings: the skill fixes g < 1
so the iteration is a CONTRACTION, which by Banach has exactly one fixed point, while a ring attractor needs
a continuum of them. The property that makes steps 11 to 13 well posed is the one that forbids a compass
(LK-18). This step asks a structural question instead, where no dynamics and no g are involved.

WHAT IS TESTED, AND WHERE THE DIRECTION COMES FROM. Hulse et al.'s central-complex connectome paper states:
    "The 16 EPG wedges in the EB ALTERNATE so that half go to the right PB while half go to the left."
So EPG neurons adjacent in the ellipsoid body should sit in OPPOSITE halves of the protocerebral bridge.
The direction of this test is taken from that sentence, not from the data. This matters because two earlier
attempts at a compass test had their direction chosen after seeing results, and were discarded (LK-18).

WHAT IS *NOT* TESTED HERE, AND WHY. The PEN_a hemisphere-offset symmetry was measured before its direction
was fixed and is therefore EXPLORATORY per METHOD.md: the numbers that generated a hypothesis are never
later cited as evidence for it. They are reported as a reading and carry no verdict. Hulse's Figure 17
settles that direction afterwards -- left-PB PEN_a shifts counterclockwise, right-PB clockwise, so a
mirror-symmetric bridge requires the SAME index-offset sign on both sides, which is what was seen -- but
seeing it second is the whole problem, and the reading says so.

Subnetwork: EPG 46, EPGt 4, PEG 18, PEN_a 20, PEN_b 22, Delta7 42 = 152 neurons, 7,390 internal edges at
weight >= 3. Glomerulus labels are parsed from `instance` (Delta7 spans 2-3, everything else exactly 1).

Statistic, zero parameters. PEN and PEG contact EPG in the EB, so two EPG neurons at nearby EB positions
share PEN/PEG partners. For each EPG, similarity to every other EPG is the cosine of their incoming
PEN/PEG weight vectors, and its NEAREST NEIGHBOUR is the most similar other EPG.
    ALT = the fraction of EPG neurons whose nearest neighbour lies in the OPPOSITE PB half.
Chance is the contralateral share of the pool, 0.5 here since the panel is 23 per side.

Null: degree-preserving double-edge swaps WITHIN the subnetwork, 20 sweeps, seeds 3000..3009. In- and
out-degree are exactly preserved AND the same weight values are reused, so total drive is identical and the
comparison isolates who-connects-to-whom. This null is strictly stronger than the one steps 11 to 13 use,
where shuffled arms are also 6-9x less driven.
WHY NOT THE GLOBAL SHUFFLE (LK-18): permuting presynaptic partners across all 166,700 neurons leaves almost
no edge inside a 152-neuron subnetwork, so it deletes the subnetwork rather than its structure and would
pass anything. Seeds 2000-2009 were used during the lookup and are NOT reused here.

Verdicts, fixed before running.
  C0 pool is balanced   the EPG panel is within 2 of even between PB halves. Else the chance level is not
                        0.5 and ALT means something else. Stop if it fails.
  C1 alternation        REAL's ALT > 0.5.
  C2 wiring earns it    REAL's ALT is above every one of the ten rewired seeds (one-sided p ~ 1/11).
  C3 holds at k = 3     the same, counting the three nearest neighbours instead of one.
  THE RING ALTERNATES = C1 and C2. C3 is read beside them.
Readings, not verdicts: the PEN_a and PEN_b hemisphere offsets (exploratory, see above); Delta7 glomerulus
  spread; the leading eigenvalue of the subnetwork against the same ten seeds.

Run from data/:  step14_compass.py
"""
import json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import fpmodel as fm

SEEDS = list(range(3000, 3010))
SWEEPS = 20
t0 = time.time()

net = fm.load_network("baseline", torch.float64); N = net.N
ann = pd.read_parquet("annotations.parquet", columns=["bodyId", "superclass", "instance", "type"])
ann = ann[ann.superclass.notna()].sort_values("bodyId").reset_index(drop=True)
assert (ann.bodyId.to_numpy() == net.ann.bodyId.to_numpy()).all(), "row order"
typ = net.ann.type.astype(str); inst = ann.instance.astype(str)
HD = np.flatnonzero(typ.str.match(r"^(EPG|EPGt|PEG|Delta7)$|^PEN_[ab]\(").to_numpy())
kind = [typ.iloc[i] for i in HD]
gloms = lambda s: re.findall(r"([LR])(\d)", s.split(")_")[-1])
G = [gloms(inst.iloc[i]) for i in HD]

s_all = np.ones(N); s_all[net.pre] = np.sign(net.val)
A = sp.csr_matrix((s_all[net.pre] * np.abs(net.val), (net.post, net.pre)), shape=(N, N))
W = np.asarray(A[HD][:, HD].todense())
post_i, pre_i = np.nonzero(W)
EPG = [i for i in range(len(HD)) if kind[i].startswith("EPG") and len(G[i]) == 1]
PP = [i for i in range(len(HD)) if kind[i].startswith(("PEN", "PEG")) and len(G[i]) == 1]
side = np.array([G[i][0][0] for i in EPG])
nL, nR = int((side == "L").sum()), int((side == "R").sum())
print(f"{len(HD)} neurons, {len(post_i)} internal edges; EPG {len(EPG)} = {nL}L + {nR}R; PEN/PEG {len(PP)}")


def rewire(seed):
    rng = np.random.default_rng(seed)
    po, pr, va = post_i.copy(), pre_i.copy(), W[post_i, pre_i].copy()
    E = len(po); pres = set(zip(po.tolist(), pr.tolist()))
    for _ in range(SWEEPS * E):
        i, j = rng.integers(0, E, 2)
        if i == j: continue
        a, b, c, d = po[i], pr[i], po[j], pr[j]
        if (a, d) in pres or (c, b) in pres: continue
        pres.discard((a, b)); pres.discard((c, d)); pr[i], pr[j] = d, b
        pres.add((a, d)); pres.add((c, b))
    M = np.zeros_like(W); M[po, pr] = va
    return M


def alt(M, k=1):
    """Fraction of EPG whose k nearest EPG neighbours (by incoming PEN/PEG profile) are contralateral."""
    V = M[np.ix_(EPG, PP)]
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    S = V @ V.T
    np.fill_diagonal(S, -np.inf)
    hits = 0.0
    for a in range(len(EPG)):
        nb = np.argsort(-S[a])[:k]
        hits += float(np.mean([side[b] != side[a] for b in nb]))
    return hits / len(EPG)


V = {}
V["C0 pool is balanced"] = abs(nL - nR) <= 2
if not V["C0 pool is balanced"]:
    print(f"C0 FAILED: EPG pool {nL}L/{nR}R is not balanced; chance is not 0.5."); sys.exit(1)

a1, a3 = alt(W, 1), alt(W, 3)
nulls1, nulls3, lead = [], [], []
for sd in SEEDS:
    M = rewire(sd)
    nulls1.append(alt(M, 1)); nulls3.append(alt(M, 3))
    lead.append(float(np.abs(np.linalg.eigvalsh((M + M.T) / 2)).max()))
    print(f"  REWIRE-{sd}  ALT(k=1) {nulls1[-1]:.3f}  ALT(k=3) {nulls3[-1]:.3f}", flush=True)
nulls1, nulls3 = np.array(nulls1), np.array(nulls3)
lead0 = float(np.abs(np.linalg.eigvalsh((W + W.T) / 2)).max())

V["C1 alternation"] = a1 > 0.5
V["C2 wiring earns it"] = bool((a1 > nulls1).all())
V["C3 holds at k = 3"] = bool((a3 > nulls3).all())
V["THE RING ALTERNATES"] = V["C1 alternation"] and V["C2 wiring earns it"]
print(f"\nREAL  ALT(k=1) {a1:.3f}   ALT(k=3) {a3:.3f}   leading |eig| {lead0:.4f}")
print(f"null  ALT(k=1) {nulls1.min():.3f}-{nulls1.max():.3f}   ALT(k=3) {nulls3.min():.3f}-{nulls3.max():.3f}"
      f"   leading |eig| {min(lead):.4f}-{max(lead):.4f}")
print("\n-- verdicts --")
for k, v in V.items(): print(f"  {k:24s} {'PASS' if v else 'FAIL'}")
json.dump({"ALT_k1": a1, "ALT_k3": a3, "null_k1": nulls1.tolist(), "null_k3": nulls3.tolist(),
           "leading_real": lead0, "leading_null": lead,
           "verdicts": {k: bool(v) for k, v in V.items()},
           "epg": {"L": nL, "R": nR}}, open("step14_compass.json", "w"), indent=1)
print(f"\nwrote step14_compass.json ({time.time() - t0:.0f}s)")
