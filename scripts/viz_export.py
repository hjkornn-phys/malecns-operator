"""Export neuron positions and taste responses for a 3D viewer.

This script only measures and writes; it draws nothing. A viewer (web or notebook) reads viz_taste.npz
and needs no connectome data of its own.

Model: compare_steady.py's, h = ReLU(g W h + b + u), g = 0.9, b = 0.1, float32, tol 1e-6, W[post, pre] =
  sign(pre) * w / in_tot_full[post]. Response = h(stimulus) - h(no stimulus), as everywhere else in this repo.
Stimuli: the four labellar GRN groups of shiu_tasks.json (sugar LB3b+c, water LB3a, bitter LB1a-d,
  ir94e LB1e/LB2a-c), each driven at u = 1 on every body of the group, one column per group.
Positions: `somaLocation` from annotations.parquet, MaleCNS voxel coordinates, parsed as three ints.
  141,781 of 166,700 neurons have one. Neurons without a position are still simulated (they carry the
  dynamics) but are not exported, since a viewer cannot place them.
Frames (optional): the fixed-point iteration is solved from h = 0, so its iterates are a propagation
  sequence from the stimulated neurons outward. --frames=N stores N evenly spaced iterates of ONE stimulus,
  including the first and the converged last, for an animation.

Run from data/ after retention.py and shiu_tasks.py:
  viz_export.py [--rule=baseline|or1] [--gain=0.9] [--frames=N] [--frame-stim=sugar] [--out=viz_taste.npz]
"""
import json, sys, time
import numpy as np
import pandas as pd
import scipy.sparse as sp

arg = lambda k, d: next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith(f"--{k}=")), d)
RULE, G = arg("rule", "baseline"), float(arg("gain", 0.9))
FRAMES, FRAME_STIM, OUT = int(arg("frames", 0)), arg("frame-stim", "sugar"), arg("out", "viz_taste.npz")
B, TOL, MAXIT = 0.1, 1e-6, 2000
STIMULI = ["sugar", "water", "bitter", "ir94e"]
assert RULE in ("baseline", "or1")
assert FRAME_STIM in STIMULI

T = json.load(open("shiu_tasks.json"))
ann = pd.read_parquet("annotations.parquet",
                      columns=["bodyId", "superclass", "class", "type", "somaLocation", "somaSide"])
ann = ann[ann.superclass.notna()].sort_values("bodyId").reset_index(drop=True)
N = len(ann)
assert N == T["N"], f"{N} neurons here, {T['N']} in shiu_tasks.json — rebuild one of them"
nt = pd.read_parquet("nt.parquet", columns=["body", "consensus_nt"]).set_index("body").consensus_nt
cons = ann.bodyId.map(nt).fillna("missing").astype(str).to_numpy()
SIGN = {"acetylcholine": 1, "dopamine": 1, "serotonin": 1, "octopamine": 1,
        "gaba": -1, "glutamate": -1, "histamine": -1}
sign = np.array([SIGN.get(c, 1) for c in cons], np.float32)

d = np.load("nn_edges.npz"); pre, post, w = d["pre"], d["post"], d["w"].astype(np.float64)
in_tot = np.bincount(post, weights=w, minlength=N)
out_tot = np.bincount(pre, weights=w, minlength=N)
m = w >= 3
if RULE == "or1": m &= (w / in_tot[post] >= 0.01) | (w / out_tot[pre] >= 0.01)
W = sp.csr_matrix(((G * sign[pre[m]] * w[m] / in_tot[post[m]]).astype(np.float32), (post[m], pre[m])),
                  shape=(N, N))

rows = {k: np.asarray(T["stimuli"][k]["rows"], np.int64) for k in STIMULI}
U = np.zeros((N, len(STIMULI) + 1), np.float32)          # last column: no stimulus
for j, k in enumerate(STIMULI): U[rows[k], j] = 1.0


def solve(U, keep_every=0):
    H, snaps = np.zeros_like(U), []
    for it in range(1, MAXIT + 1):
        Hn = np.maximum(W @ H + B + U, 0.0)
        delta = np.abs(Hn - H).max(); H = Hn
        if keep_every: snaps.append(H.copy())
        if delta < TOL: return H, it, snaps
    raise RuntimeError(f"no convergence in {MAXIT} iterations (last change {delta:.3g})")


t0 = time.time()
H, iters, _ = solve(U)
resp = H[:, :-1] - H[:, -1:]                             # N x 4
print(f"{RULE} g={G}: {int(m.sum())} edges, {iters} iterations, {time.time() - t0:.0f}s", flush=True)

# positions: "[37124 22258 36274]" -> three ints; neurons without one are dropped from the export
loc = ann.somaLocation.fillna("").astype(str)
has = loc.str.startswith("[").to_numpy()
xyz = np.full((N, 3), -1, np.int32)
xyz[has] = np.array([np.fromstring(s.strip("[]"), sep=" ", dtype=np.float64)
                     for s in loc[has]], np.float64).astype(np.int32)
keep = np.flatnonzero(has)

sc = ann.superclass.astype(str)
dn = sc.eq("descending_neuron").to_numpy()
readout = (sc.eq("descending_neuron") | sc.str.endswith("_motor") | sc.str.contains("efferent")
           | sc.str.endswith("_endocrine")).to_numpy()
stim_of = np.zeros(N, np.int8)                            # 0 = not stimulated, else 1-based stimulus index
for j, k in enumerate(STIMULI): stim_of[rows[k]] = j + 1

cat = lambda s: (lambda c: (c.codes.astype(np.int16), np.array(list(c.categories), object)))(
    pd.Categorical(ann[s].fillna("?").astype(str)))
sc_codes, sc_names = cat("superclass")
cl_codes, cl_names = cat("class")
ty_codes, ty_names = cat("type")

out = dict(xyz=xyz[keep], row=keep.astype(np.int32), bodyId=ann.bodyId.to_numpy()[keep],
           resp=resp[keep].astype(np.float32), baseline_h=H[keep, -1].astype(np.float32),
           superclass=sc_codes[keep], superclass_names=sc_names, klass=cl_codes[keep], klass_names=cl_names,
           type=ty_codes[keep], type_names=ty_names, side=ann.somaSide.fillna("?").astype(str).to_numpy()[keep],
           is_dn=dn[keep], is_readout=readout[keep], stim_of=stim_of[keep], stimuli=np.array(STIMULI, object))

if FRAMES:
    j = STIMULI.index(FRAME_STIM)
    Hf, it_f, snaps = solve(U[:, [j, -1]], keep_every=1)
    pick = np.unique(np.linspace(0, len(snaps) - 1, FRAMES).round().astype(int))
    out["frames"] = np.stack([(snaps[i][:, 0] - snaps[i][:, 1])[keep] for i in pick]).astype(np.float32)
    out["frame_iters"] = (pick + 1).astype(np.int32)
    out["frame_stim"] = FRAME_STIM
    print(f"frames: {FRAME_STIM}, {len(pick)} of {it_f} iterations {out['frame_iters'][:6]}...", flush=True)

out["meta"] = json.dumps({"rule": RULE, "gain": G, "b": B, "tol": TOL, "iterations": iters,
                          "edges": int(m.sum()), "neurons_total": N, "neurons_exported": len(keep),
                          "coordinates": "MaleCNS voxel, somaLocation",
                          "response": "h(stimulus) - h(no stimulus)",
                          "stimulus_sizes": {k: int(len(v)) for k, v in rows.items()},
                          "source": "MaleCNS v1.0 (CC-BY 4.0); stimulus groups from shiu_tasks.json"})
np.savez_compressed(OUT, **out)

import os
print(f"{OUT}: {len(keep)} of {N} neurons ({100 * len(keep) / N:.1f}%), {os.path.getsize(OUT) / 1e6:.1f} MB")
print("stimulus bodies: " + ", ".join(f"{k} {len(v)}" for k, v in rows.items()) +
      f" | DN {int(dn[keep].sum())} readout {int(readout[keep].sum())}")
print("xyz range: " + " ".join(f"{a}[{out['xyz'][:, i].min()}, {out['xyz'][:, i].max()}]"
                               for i, a in enumerate("xyz")))
for j, k in enumerate(STIMULI):
    r = out["resp"][:, j]
    print(f"  {k:7s} response max {r.max():.3f} | responding (>1e-3 of max) "
          f"{int((np.abs(r) > 1e-3 * np.abs(r).max()).sum()):6d} | DN responding "
          f"{int((np.abs(r[out['is_dn']]) > 1e-3 * np.abs(r).max()).sum()):4d}")
