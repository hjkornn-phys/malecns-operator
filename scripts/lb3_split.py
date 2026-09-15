"""Prototype: can connectivity split MaleCNS LB3 GRNs into Shiu's sugar and water sets?

1. FlyWire (v783): fingerprint each of Shiu's sugar (21) and water (18) GRNs by its synapse fractions over
   partner cell types, outputs and inputs kept apart.
2. Check the fingerprint separates them in FlyWire: leave-one-out nearest centroid (cosine), against
   label permutations.
3. MaleCNS: fingerprint every LB3* body over the same partner types (via flywireType) and score it
   cos(sugar centroid) - cos(water centroid). Cross-tabulate with MaleCNS type LB3a-d.
Only partner types present in both datasets are used. Run from data/ after shiu_tasks.py's downloads.
"""
import json, re
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

rng = np.random.default_rng(0)
src = "\n".join("".join(c["source"]) for c in json.load(open("shiu/figures.ipynb"))["cells"])
ids_of = lambda v: [int(x) for x in re.findall(r"\d{18}", re.search(r"\b" + v + r"\s*=\s*\[(.*?)\]", src, re.S).group(1))]
sugar, water = ids_of("neu_sugar"), ids_of("neu_water")

fw = pd.read_csv("shiu/flywire_neuron_annotations.tsv", sep="\t", dtype=str, low_memory=False,
                 usecols=["root_id", "cell_type"])
fw_type = dict(zip(fw.root_id.astype("int64"), fw.cell_type))

ann = pd.read_parquet("annotations.parquet")
ann = ann[ann.superclass.notna()].sort_values("bodyId").reset_index(drop=True)
fwt = ann.flywireType.fillna("").astype(str).str.split(",").map(lambda l: [s.strip() for s in l if s.strip()])
mcns_types = {t for l in fwt for t in l}

# --- FlyWire fingerprints
grn = np.array(sugar + water, np.int64)
t = pq.read_table("shiu/Connectivity_783.parquet", columns=["Presynaptic_ID", "Postsynaptic_ID", "Connectivity"])
sel = pc.or_(pc.is_in(t["Presynaptic_ID"], pa_grn := __import__("pyarrow").array(grn)),
             pc.is_in(t["Postsynaptic_ID"], pa_grn))
e = t.filter(sel).to_pandas()
e.columns = ["pre", "post", "w"]
print("FlyWire edges touching the 39 GRNs:", len(e))


def fingerprints_fw():
    rows = {}
    for g in grn:
        o = e[e.pre == g]; i = e[e.post == g]
        f = {}
        for df, col, tag in [(o, "post", "out"), (i, "pre", "in")]:
            ty = df[col].map(fw_type)
            s = df.w.groupby(ty).sum()
            s = s[[k in mcns_types for k in s.index]]            # shared vocabulary only
            tot = df.w.sum()
            for k, v in s.items(): f[f"{tag}:{k}"] = v / tot if tot else 0.0
        rows[g] = f
    return pd.DataFrame.from_dict(rows, orient="index").fillna(0.0)


F = fingerprints_fw().reindex(grn).fillna(0.0)
y = np.array([1] * len(sugar) + [0] * len(water))                  # 1 = sugar
missing = F.sum(axis=1).eq(0).to_numpy()
print("GRNs with no shared-type synapses (dropped):", int(missing.sum()), list(grn[missing]))
F, y = F[~missing], y[~missing]
import sys
if len(sys.argv) > 1:                                                 # ablation: drop partner types matching a regex
    drop = F.columns.str.contains(sys.argv[1], regex=True)
    print(f"dropping {int(drop.sum())} feature columns matching {sys.argv[1]!r}:", list(F.columns[drop]))
    F = F.loc[:, ~drop]
X = F.to_numpy()
unit = lambda A: A / np.maximum(np.linalg.norm(A, axis=-1, keepdims=True), 1e-12)


def loo_acc(X, y):
    ok = 0
    for k in range(len(y)):
        m = np.ones(len(y), bool); m[k] = False
        cs, cw = X[m & (y == 1)].mean(0), X[m & (y == 0)].mean(0)
        ok += int((unit(X[k]) @ unit(cs) > unit(X[k]) @ unit(cw)) == bool(y[k]))
    return ok / len(y)


acc = loo_acc(X, y)
perm = np.array([loo_acc(X, rng.permutation(y)) for _ in range(500)])
print(f"FlyWire leave-one-out accuracy {acc:.3f} (n={len(y)}); permutation mean {perm.mean():.3f}, "
      f"p = {(1 + (perm >= acc).sum()) / (1 + len(perm)):.4f}")

cs, cw = X[y == 1].mean(0), X[y == 0].mean(0)
diff = pd.Series(cs - cw, index=F.columns).sort_values()
print("partner types most WATER-biased:", diff.head(8).round(3).to_dict())
print("partner types most SUGAR-biased:", diff.tail(8).round(3).to_dict())

# --- MaleCNS fingerprints over the same columns
d = np.load("nn_edges.npz"); pre, post, w = d["pre"], d["post"], d["w"].astype(np.float64)
lb3 = np.flatnonzero(ann.type.fillna("").astype(str).str.startswith("LB3") |
                     fwt.map(lambda l: "LB3" in l).to_numpy())
first_fwt = fwt.map(lambda l: l[0] if len(l) == 1 else ",".join(l)).to_numpy()
in_lb3 = np.isin(pre, lb3) | np.isin(post, lb3)
p_, q_, w_ = pre[in_lb3], post[in_lb3], w[in_lb3]
col = {c: j for j, c in enumerate(F.columns)}
M = np.zeros((len(lb3), len(col)))
pos = {b: k for k, b in enumerate(lb3)}
for tag, me, other in [("out", p_, q_), ("in", q_, p_)]:
    for b in lb3:
        m = me == b
        tot = w_[m].sum()
        if not tot: continue
        s = pd.Series(w_[m]).groupby(first_fwt[other[m]]).sum()
        for k, v in s.items():
            for part in str(k).split(","):                       # a multi-type partner counts toward each
                j = col.get(f"{tag}:{part.strip()}")
                if j is not None: M[pos[b], j] += v / tot

Mu = unit(M)
score = Mu @ unit(cs) - Mu @ unit(cw)
res = pd.DataFrame({"bodyId": ann.bodyId.iloc[lb3].to_numpy(), "type": ann.type.iloc[lb3].fillna("?").to_numpy(),
                    "side": ann.rootSide.iloc[lb3].fillna("?").to_numpy(), "sugar_minus_water": score,
                    "call": np.where(score > 0, "sugar", "water")})
res.to_csv("lb3_split.csv", index=False)
print("\nMaleCNS LB3 bodies:", len(res))
print(pd.crosstab(res.type, res.call, margins=True))
print(res.groupby("type").sugar_minus_water.describe().round(3))

# FlyWire self-consistency: same score on the training GRNs
sf = unit(X) @ unit(cs) - unit(X) @ unit(cw)
print("\nFlyWire score, sugar GRNs:", np.round(np.percentile(sf[y == 1], [0, 50, 100]), 3),
      "water GRNs:", np.round(np.percentile(sf[y == 0], [0, 50, 100]), 3))
