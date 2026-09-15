import json, numpy as np
d = np.load("nn_edges.npz"); pre, post, w = d["pre"], d["post"], d["w"]
N = 166700; E = len(w); M = int(w.sum())
in_tot = np.bincount(post, weights=w, minlength=N)
out_tot = np.bincount(pre, weights=w, minlength=N)
fin = w / in_tot[post]; fout = w / out_tot[pre]
def row(name, m):
    indeg = np.bincount(post[m], minlength=N); outdeg = np.bincount(pre[m], minlength=N)
    return dict(rule=name, edges=int(m.sum()), edge_pct=round(100*m.sum()/E, 1),
                mass_pct=round(100*w[m].sum()/M, 1), no_input=int((indeg==0).sum()),
                no_output=int((outdeg==0).sum()), isolated=int(((indeg==0)&(outdeg==0)).sum()))
k3 = w >= 3
rows = [row("k>=3", k3)]
for r in [0.002, 0.005, 0.01, 0.02]:
    p = f"{100*r:g}%"
    rows += [row(f"k>=3 & in>={p}", k3 & (fin >= r)),
             row(f"k>=3 & out>={p}", k3 & (fout >= r)),
             row(f"k>=3 & (in OR out)>={p}", k3 & ((fin >= r) | (fout >= r)))]
json.dump(rows, open("retention_or.json", "w"), indent=1)
for x in rows: print(x)
