"""Build the "Does the wiring matter?" page (docs/surge/index.html, ko.html) from step 13's results.

Illustration, not a result. Every track drawn is one of the SCORED episodes of step 13's primary condition:
step13_surge.py --export=N re-runs that condition with the same seed and the same 200-episode batch and
records the first N tracks, so nothing shown was selected for looking good. The arrival table is read
straight out of step13_surge.json. The arena, the glow and the fly markers are illustration.

The page states in its own text what the run cannot claim: that this is not how a fly navigates (odour
updates a goal, and steering compares that goal against a compass this model does not have), that it holds
for smooth gradients only, that excitatory/inhibitory signs are predicted rather than measured, that there
is no body, that "turn toward the stronger side" is an assumption given equally to every arm, and that five
shuffled seeds buy only a one-sided p of about 1/6.

Run from data/ after:  step13_surge.py  and  step13_surge.py --export=12
    python3 ../scripts/viz/export_surge.py
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "docs", "surge")
COLORS = {"REAL": "#F2A03D", "SHUF-1000": "#7FA8D4", "RANDOM-WALK": "#8A8A94", "ORACLE": "#6FD497"}
ORDER = ["REAL", "SHUF-1000", "RANDOM-WALK", "ORACLE"]

paths = json.load(open("step13_paths.json"))
run = json.load(open("step13_surge.json"))
P, R0 = paths["condition"]["p"], paths["condition"]["r0"]
g = paths["geometry"]

arms = []
for k in ORDER:
    a = paths["arms"][k]
    arms.append({"key": k, "color": COLORS[k], "arrival": round(a["arrival"], 3),
                 "paths": [[[round(x, 3), round(y, 3)] for x, y in ep] for ep in a["paths"]],
                 "arrived": a["arrived"]})

shuf = [k for k in run["arms"] if k.startswith("SHUF-")]
conds, seen = [], []
for key in run["arms"]["REAL"]:
    if key == "m" or key in seen: continue
    seen.append(key)
    p_s, r_s = key.split("|")
    sv = [run["arms"][s][key]["arrival"] for s in shuf]
    conds.append({"p": float(p_s), "r0": float(r_s), "primary": float(p_s) == P and float(r_s) == R0,
                  "real": f'{run["arms"]["REAL"][key]["arrival"]:.3f}',
                  "shuf_lo": f"{min(sv):.3f}", "shuf_hi": f"{max(sv):.3f}",
                  "rand": f'{run["arms"]["RANDOM-WALK"][key]["arrival"]:.3f}',
                  "oracle": f'{run["arms"]["ORACLE"][key]["arrival"]:.3f}'})
conds.sort(key=lambda c: (not c["primary"], c["r0"], c["p"]))

data = {"meta": {"p": P, "r0": R0, "n_shown": paths["condition"]["n_shown"],
                 "n_scored": paths["condition"]["n_scored"], "arrive": g["arrive"], "step": g["step"],
                 "turn_deg": g["turn_deg"], "frames": g["frames"], "theta_min_deg": g["theta_min_deg"]},
        "arms": arms, "conditions": conds, "verdicts": run.get("verdicts", {})}

tpl = open(os.path.join(HERE, "surge_template.html")).read()
strings = json.load(open(os.path.join(HERE, "surge_strings.json")))
os.makedirs(OUT, exist_ok=True)
for lang, name in (("en", "index.html"), ("ko", "ko.html")):
    S = strings[lang]; page = tpl
    for k, v in S.items():
        if k != "js": page = page.replace("{{%s}}" % k, v)
    assert "{{" not in page, "unfilled token"
    page = page.replace("__FLYDATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    page = page.replace("__I18N__", json.dumps(S["js"], ensure_ascii=False, separators=(",", ":")))
    head = (f'<!doctype html>\n<html lang="{lang}">\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n')
    open(os.path.join(OUT, name), "w").write(head + page + "\n</html>\n")
kb = os.path.getsize(os.path.join(OUT, "index.html")) / 1024
print(f'{len(arms)} arms, {data["meta"]["n_shown"]} episodes shown of {data["meta"]["n_scored"]} scored, '
      f'{len(conds)} conditions -> {os.path.normpath(OUT)} ({kb:.0f} KB)')
