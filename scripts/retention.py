"""Synapse-mass retention curves for MaleCNS v1.0 (minconf 0.5), streamed per batch.

Universe: bodies with a non-null `superclass` (neurons). Edge = neuron->neuron row.
Criterion A: absolute weight >= k.
Criterion B: input fraction w / (total neuron input of post) >= r.
"""
import json
import numpy as np
import pandas as pd
import pyarrow.ipc as ipc

ann = pd.read_parquet("annotations.parquet", columns=["bodyId", "superclass"])
neurons = np.sort(ann.loc[ann.superclass.notna(), "bodyId"].to_numpy(np.int64))
N = len(neurons)


def index_of(ids):
    pos = np.searchsorted(neurons, ids)
    pos = np.minimum(pos, N - 1)
    ok = neurons[pos] == ids
    return pos, ok


CAP = 100_000
hist_all = np.zeros(CAP + 1, np.int64)       # edge counts by weight, all bodies
tot_rows = tot_mass = 0
pre_l, post_l, w_l = [], [], []
in_all = np.zeros(N, np.int64)               # post input incl. non-neuron partners

with ipc.open_file("weights.feather") as r:
    for i in range(r.num_record_batches):
        b = r.get_batch(i)
        pre = b.column("body_pre").to_numpy()
        post = b.column("body_post").to_numpy()
        w = b.column("weight").to_numpy()
        tot_rows += len(w)
        tot_mass += int(w.sum())
        ip, okp = index_of(pre)
        iq, okq = index_of(post)
        in_all += np.bincount(iq[okq], weights=w[okq], minlength=N).astype(np.int64)
        m = okp & okq
        pre_l.append(ip[m].astype(np.int32))
        post_l.append(iq[m].astype(np.int32))
        w_l.append(w[m].astype(np.int32))

pre = np.concatenate(pre_l); post = np.concatenate(post_l); w = np.concatenate(w_l)
del pre_l, post_l, w_l
np.savez("nn_edges.npz", pre=pre, post=post, w=w)

E, M = len(w), int(w.sum())
in_nn = np.bincount(post, weights=w, minlength=N)
out_deg0 = np.bincount(pre, minlength=N); in_deg0 = np.bincount(post, minlength=N)
frac = w / in_nn[post]


def row(mask):
    kept_w = w[mask]
    indeg = np.bincount(post[mask], minlength=N)
    outdeg = np.bincount(pre[mask], minlength=N)
    return {
        "edges": int(mask.sum()),
        "edge_pct": 100 * mask.sum() / E,
        "mass_pct": 100 * kept_w.sum() / M,
        "no_input": int((indeg == 0).sum()),
        "no_output": int((outdeg == 0).sum()),
        "isolated": int(((indeg == 0) & (outdeg == 0)).sum()),
    }


out = {
    "neurons": N,
    "all_rows": tot_rows, "all_mass": tot_mass,
    "nn_edges": E, "nn_mass": M,
    "nn_mass_pct_of_all": 100 * M / tot_mass,
    "baseline_no_input": int((in_deg0 == 0).sum()),
    "baseline_no_output": int((out_deg0 == 0).sum()),
    "neuron_input_share_median_pct": float(100 * np.median((in_nn / np.maximum(in_all, 1))[in_all > 0])),
    "A": {}, "B": {}, "AB": {},
}
for k in [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 50, 100]:
    out["A"][k] = row(w >= k)
for rr in [0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10]:
    out["B"][rr] = row(frac >= rr)
for k in [1, 3, 5, 10]:
    for rr in [0.001, 0.005, 0.01, 0.02]:
        out["AB"][f"{k}|{rr}"] = row((w >= k) & (frac >= rr))
json.dump(out, open("retention.json", "w"), indent=1)
print(json.dumps(out, indent=1))
