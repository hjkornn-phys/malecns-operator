"""Build the Fly Memory Match page (docs/memory-match/index.html in English, ko.html in Korean) from step 10's results.

Illustration, not a result. Every comparison shown is one real test B item of step 10 (odor A on h1 neurons, one blank
frame, odor B on held-out neurons), answered with the saved D = 1 probabilities, mean over init seeds. The card stream
is stitched from items whose odor A equals the previous item's odor B, balancing same and different; the model drew
fresh neurons and reset its memory for every item. Card bars are Hallem & Carlson 2006 responses from DoOR.data v2.0.1
(CC BY-SA 4.0), so the generated pages carry that license for those values.

Run from data/ after step10_odor_delay.py:  python3 ../scripts/viz/export_memory_match.py
"""
import csv, json, os, random
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "docs", "memory-match")
z = np.load("step10_odor_delay.npz"); res = json.load(open("step10_odor_delay.json"))
rows = [x for x in list(csv.reader(open("door/Or10a.csv"), delimiter=";"))[1:] if x[3] != "SFR"]
meta = {x[3]: (x[2], x[1]) for x in rows}
keys, R = res["odors"], z["hallem"]
pairs = [tuple(map(int, p)) for p in z["pairs test B"]]
ARMS = ["REAL", "SHUF 1000", "MEM-ONLY"]
prob = {k: z[f"{k}|test B|D1|prob"] for k in ARMS}
acc = {k: float(z[f"{k}|test B|D1|correct"].mean()) for k in ARMS}

by_a = {}
for i, (a, b) in enumerate(pairs): by_a.setdefault(a, []).append(i)

def chain(seed, target=160):
    rng = random.Random(seed); used = set()
    cur = pairs[rng.randrange(len(pairs))][0]; cards, steps = [cur], []
    while len(steps) < target:
        opts = [i for i in by_a[cur] if i not in used]
        if not opts: break
        same_n = sum(s["same"] for s in steps)
        pref = [i for i in opts if (pairs[i][0] == pairs[i][1]) == (same_n < len(steps) - same_n)]
        i = rng.choice(pref or opts); used.add(i)
        steps.append({"t": i, "same": pairs[i][0] == pairs[i][1]}); cur = pairs[i][1]; cards.append(cur)
    return cards, steps

cards, steps = max((chain(s) for s in range(200)), key=lambda c: len(c[1]))
used = sorted(set(cards)); idx = {o: k for k, o in enumerate(used)}
data = {"odors": [{"name": meta[keys[o]][0], "cls": meta[keys[o]][1], "fp": [round(float(v)) for v in R[o]]} for o in used],
        "cards": [idx[c] for c in cards],
        "steps": [{"trial": s["t"] + 1, "same": s["same"], "p": [round(float(prob[k][s["t"]]), 3) for k in ARMS]} for s in steps],
        "arms": [{"key": k, "acc": acc[k]} for k in ARMS], "verdicts": res.get("verdicts", {})}
tpl = open(os.path.join(HERE, "memory_match_template.html")).read()
strings = json.load(open(os.path.join(HERE, "memory_match_strings.json")))
os.makedirs(OUT, exist_ok=True)
for lang, name in (("en", "index.html"), ("ko", "ko.html")):
    S = strings[lang]; page = tpl
    for k, v in S.items():
        if k != "js": page = page.replace("{{%s}}" % k, v)
    assert "{{" not in page, "unfilled token"
    page = page.replace("__FLYDATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    page = page.replace("__I18N__", json.dumps(S["js"], ensure_ascii=False, separators=(",", ":")))
    head = f'<!doctype html>\n<html lang="{lang}">\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
    open(os.path.join(OUT, name), "w").write(head + page + "\n</html>\n")
print(f"{len(cards)} cards, {len(steps)} comparisons, {sum(s['same'] for s in steps)} same, {len(used)} odors -> {os.path.normpath(OUT)}")
