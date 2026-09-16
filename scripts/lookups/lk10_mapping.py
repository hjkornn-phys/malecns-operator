"""DoOR 2.0 unit -> MaleCNS ORN type mapping used by LK-10 (PREDICTIONS.md). Table work only, no network.

Fetches DoOR.data v2.0.1 (CC-BY 4.0; Münch & Galizia 2016) into data/door/ and writes door/door_to_malecns_DRAFT.csv.
Rules: a DoOR unit maps through its `glomerulus` field; summed-sensillum recordings, larval receptors and units whose
glomerulus is not a MaleCNS ORN type (VP*, or two glomeruli such as Or33b's DM5+DM3) are excluded; DL2d and DL2v
both take ac3A, which DoOR does not separate; where a type has several units the one with the most measured
odorants is primary and the rest are listed as alternates. MaleCNS VC3/VC5 are hemibrain VC3l/VC3m.
Run from data/:  uv run --project .. python ../scripts/lookups/lk10_mapping.py
"""
import collections, csv, os, subprocess
import pandas as pd

RAW = "https://raw.githubusercontent.com/ropensci/DoOR.data/v2.0.1/data/"
os.makedirs("door", exist_ok=True)
for f in ("door_mappings.csv", "door_response_matrix.csv"):
    if not os.path.exists(f"door/{f}"): subprocess.run(["curl", "-sfL", "-o", f"door/{f}", RAW + f], check=True)

a = pd.read_parquet("annotations.parquet"); a = a[a["class"].astype(str).eq("olfactory")]
nmc = a.type.value_counts().rename(lambda t: t.replace("ORN_", "")).to_dict()

r = list(csv.reader(open("door/door_mappings.csv"), delimiter=";"))
m = [dict(zip(r[0], x[1:])) for x in r[1:]]                      # rows carry an unheaded row-number column
rows = list(csv.reader(open("door/door_response_matrix.csv"), delimiter=";"))
hdr, data = rows[0], [x for x in rows[1:] if x[0] != "SFR"]
col = {h: {x[0] for x in data if x[i + 1] not in ("NA", "")} for i, h in enumerate(hdr)}
SUMMED = {"ac1", "ac1BC", "ac2", "ac2BC", "ac3_noOr35a", "ac4", "ac1A", "ac1B", "ac2A", "ac2B"}
cand = collections.defaultdict(list)
for x in m:
    rec, g = x["receptor"], x["glomerulus"]
    if rec not in col or rec in SUMMED or g in ("?", "") or x["adult"] == "FALSE": continue
    gs = ["DL2d", "DL2v"] if g == "DL2d/v" else [g]
    if any(t not in nmc for t in gs): continue
    for t in gs: cand[t].append((rec, len(col[rec])))
NOTES = {"DA1": "DoOR maps Or67d here but the normalized matrix holds no data for it; no usable unit",
         "VA7m": "DoOR lists VA7m with an unknown receptor and no data",
         "DL2d": "ac3A (Ir75b/Ir75c) maps to DL2d and DL2v without separation; both get it",
         "DL2v": "ac3A (Ir75b/Ir75c) maps to DL2d and DL2v without separation; both get it",
         "VC5": "DoOR VC5=Ir41a; MaleCNS VC5 = hemibrain VC3m (renamed, Task 2022)",
         "VC3": "MaleCNS VC3 = hemibrain VC3l"}
out = []
for t in sorted(nmc):
    c = sorted(cand.get(t, []), key=lambda z: -z[1])
    note = NOTES.get(t, "split of old VC5, no DoOR unit" if t.startswith("VM6") else "")
    if len(c) > 1 and t not in ("DL2d", "DL2v"):
        note = "; ".join(filter(None, [note, "alternates: " + ", ".join(f"{u}({n})" for u, n in c[1:])]))
    out.append({"malecns_type": "ORN_" + t, "orns": nmc[t], "door_unit": c[0][0] if c else ("Or67d" if t == "DA1" else ""),
                "odorants": c[0][1] if c else 0, "notes": note})
with open("door/door_to_malecns_DRAFT.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=out[0].keys(), lineterminator="\n"); w.writeheader(); w.writerows(out)
print(len(out), "types;", sum(1 for o in out if o["odorants"]), "with data")
