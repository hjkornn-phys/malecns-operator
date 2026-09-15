"""Build a ground-truth task set for MaleCNS from Shiu et al. 2024 (Nature 634:210) experiments.

Sources (downloaded into data/shiu/, not committed):
  41586_2024_7763_MOESM2_ESM.xlsx          supplementary tables (Europe PMC, PMC11446845)
  figures.ipynb, sez_neurons.pickle        stimulus / readout FlyWire IDs (github philshiu/Drosophila_brain_model)
  flywire_neuron_annotations.tsv           FlyWire root_id -> cell_type (github flyconnectome/flywire_annotations)

Chain: FlyWire ID (Shiu used snapshot 630) -> FlyWire cell_type (783 table) -> MaleCNS bodies whose
flywireType lists that type (else MaleCNS type equals it). Named neurons also match MaleCNS `synonyms`
("Shiu 2022: <name>"). IDs absent from the 783 table are counted, not guessed.

Writes shiu_tasks.json (run from data/). Outcomes are experimental results, not Shiu's model predictions.
"""
import json, re, pickle
import numpy as np
import pandas as pd

X = "shiu/41586_2024_7763_MOESM2_ESM.xlsx"
sheets = pd.read_excel(X, sheet_name=None, header=None)
sheet = lambda prefix: sheets[next(k for k in sheets if k.startswith(prefix))]

fw = pd.read_csv("shiu/flywire_neuron_annotations.tsv", sep="\t", dtype=str, low_memory=False).set_index("root_id")
ann = pd.read_parquet("annotations.parquet")
ann = ann[ann.superclass.notna()].sort_values("bodyId").reset_index(drop=True)   # same order as nn_edges.npz

fwt_sets = ann.flywireType.fillna("").astype(str).str.split(",").map(lambda l: {s.strip() for s in l if s.strip()})
by_fwt = {}
for i, s in enumerate(fwt_sets):
    for t in s: by_fwt.setdefault(t, []).append(i)
by_type = ann.groupby(ann.type.astype(str)).indices
syn = ann.synonyms.fillna("").astype(str)


def fw_types(ids):
    hit = fw.reindex([str(i) for i in ids])
    ok = hit.cell_type.notna()
    return hit.cell_type[ok].tolist(), int((~ok).sum())


def mcns_rows(types):
    rows, unmatched = set(), []
    for t in dict.fromkeys(types):
        parts = [p.strip() for p in t.split(",")]
        got = [i for p in parts for i in by_fwt.get(p, [])] or [i for p in parts for i in by_type.get(p, [])]
        rows.update(got)
        if not got: unmatched.append(t)
    return sorted(rows), unmatched


def by_synonym(name):
    pat = re.compile(r"Shiu 2022: " + re.escape(name) + r"(\s|;|$)", re.I)
    return [int(i) for i in np.flatnonzero(syn.map(lambda s: bool(pat.search(s))))]


def group(ids, name=None):
    types, missing = fw_types(ids)
    rows, unmatched = mcns_rows(types)
    s = by_synonym(name) if name else []
    rows = sorted(int(i) for i in set(rows) | set(s))
    return {"flywire_ids": len(ids), "flywire_missing_783": missing, "flywire_types": sorted(set(types)),
            "unmatched_types": unmatched, "synonym_rows": len(s), "rows": rows,
            "mcns_types": sorted(set(ann.type.fillna("?").astype(str).iloc[rows].astype(str)))}


nb = json.load(open("shiu/figures.ipynb"))
src = "\n".join("".join(c["source"]) for c in nb["cells"])
def ids_of(var):
    m = re.search(r"\b" + var + r"\s*=\s*\[(.*?)\]", src, re.S)
    return [int(x) for x in re.findall(r"\d{18}", m.group(1))]

stimuli = {k: group(ids_of("neu_" + k)) for k in ["sugar", "water", "bitter", "ir94e", "JON_CE", "JON_F", "JON_D_m"]}
# FlyWire types Shiu's sugar and water GRNs both as LB3. Split by MaleCNS subtype per Tastekin et al.
# (bioRxiv 10.1101/2025.08.25.671814): LB3a ppk28 water; LB3b, LB3c Gr64f sugar; LB3d salt-aversive and
# LB4b (FlyWire's LB3) left out. scripts/lb3_split.py independently calls every LB3a body water.
typ = ann.type.fillna("").astype(str)
for k, types in [("sugar", ["LB3b", "LB3c"]), ("water", ["LB3a"])]:
    stimuli[k].update(rows=[int(i) for i in np.flatnonzero(typ.isin(types))], mcns_types=types,
                      rule="MaleCNS type " + "+".join(types) + " (Tastekin et al.)")

readouts = {"MN9": group([720575940660219265, 720575940645521262]),
            "aBN1": group([720575940630907434]), "aDN1": group([720575940616185531]),
            "aDN2": group([720575940629806974])}

# Supp Table 3: 106 SEZ types, optogenetic activation rate of proboscis extension (fraction of flies)
t3 = sheet("Sup Table 3")
sez = {k.lower(): v for k, v in pickle.load(open("shiu/sez_neurons.pickle", "rb")).items()}
alias = {"g2n_1": "G2N-1", "th_vum": "TH-VUM", "sink_sync": "sink"}
sez_tasks = []
for _, r in t3.iloc[1:].iterrows():
    name = str(r[0])
    ids = sez.get(name.lower(), [])
    g = group(ids, alias.get(name.lower(), name))
    sez_tasks.append({"name": name, "opto_pe_rate": float(r[1]), "shortest_path_to_mn9": str(r[3]), **g})

# Supp Tables 2 and 5: named neurons, yes/no outcomes
def yesno(v):
    s = str(v).strip().lower()
    return True if s.startswith("yes") else False if s.startswith("no") else None

named = {}
for prefix, taste in [("Supplemental Table 2 ", "sugar"), ("Supplemental Table 5 ", "water")]:
    t = sheet(prefix); hdr = [str(h) for h in t.iloc[0]]
    for _, r in t.iloc[1:].iterrows():
        n = str(r[0]); e = named.setdefault(n, {})
        e[f"responds_to_{taste}"] = yesno(r[1])
        e["sufficient_for_pe"] = yesno(r[3]) if e.get("sufficient_for_pe") is None else e["sufficient_for_pe"]
        e[f"required_for_{taste}_pe"] = yesno(r[5])
for n, e in named.items():
    ids = sez.get(n.lower(), []) or ([720575940660219265, 720575940645521262] if n == "MN9" else [])
    e.update(group(ids, n) if ids else {"rows": by_synonym(n), "synonym_rows": len(by_synonym(n))})
    e["mcns_types"] = sorted(set(ann.type.fillna("?").astype(str).iloc[e["rows"]].astype(str)))

out = {"source": "Shiu et al. 2024 Nature 634:210; supplementary tables 2, 3, 5; figures.ipynb",
       "N": len(ann), "stimuli": stimuli, "readouts": readouts, "named_neurons": named, "sez_activation": sez_tasks,
       "table10": sheet("Supp Table 10").fillna("").astype(str).values.tolist()}
json.dump(out, open("shiu_tasks.json", "w"), indent=1)

print("stimuli:")
for k, g in stimuli.items():
    print(f"  {k:8s} fw ids {g['flywire_ids']:3d} (missing {g['flywire_missing_783']}) types {g['flywire_types']} "
          f"-> MaleCNS {len(g['rows'])} bodies {g['mcns_types'][:8]} unmatched {g['unmatched_types']}")
print("readouts:")
for k, g in readouts.items():
    print(f"  {k:5s} fw {g['flywire_types']} -> MaleCNS {len(g['rows'])} {g['mcns_types']}")
print("named neurons:")
for n, e in named.items():
    print(f"  {n:9s} -> {len(e['rows'])} bodies {e['mcns_types']} | " +
          " ".join(f"{k}={v}" for k, v in e.items() if k.startswith(("responds", "sufficient", "required"))))
m = [t for t in sez_tasks if t["rows"]]
pos = [t for t in sez_tasks if t["opto_pe_rate"] > 0]
print(f"SEZ activation: {len(sez_tasks)} types, {len(m)} matched in MaleCNS, "
      f"{len(pos)} with PE rate > 0 ({sum(1 for t in pos if t['rows'])} matched)")
print("  unmatched:", [t["name"] for t in sez_tasks if not t["rows"]])
print("  multi-type:", {t["name"]: t["mcns_types"] for t in sez_tasks if len(t["mcns_types"]) > 3})
