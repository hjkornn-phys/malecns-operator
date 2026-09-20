"""Step 13: close the loop. Does the wiring walk to the smell?

Steps 11 and 12 measured two ingredients on held-out odors and neurons: which antenna was stimulated more
(0.715, ten null seeds 0.441-0.620), and whether concentration is rising or falling (HYST 0.460 against
0.191-0.197). This step spends them. A fly is placed in an odor gradient and asked to reach the source,
with NOTHING TRAINED and no controller that could do the job by itself.

WHY THIS IS NOT ANOTHER CONNECTOME GAME (LK-15). About forty exist and at least four already do odor
navigation. The audit found none of them trains weights -- and that the hazard was never training. One
states outright that its plume input "bypasses native circuit computation"; another that "the sign
convention is ours", chosen because the connectome's laterality was ambiguous; a third ships a motor
adapter and weight calibration. NONE runs a control network. This step differs on exactly three checkable
points: the steering sign is MEASURED, not chosen (step 11, on held-out odors AND neurons); the odor
reaches the readout only through the circuit; and the shuffled arms run beside the real one.

THE ONE BEHAVIOURAL ASSUMPTION, DECLARED. "Turn toward the more stimulated side" is an assumption about
attraction versus aversion. It is not a neural sign and the wiring did not supply it. It is identical in
every arm, including the nulls, so it cannot produce a difference between them. Named here so it is never
mistaken for a result.

WHAT REGIME THIS MODELS (LK-15). A SMOOTH gradient, which is where bilateral antennal comparison is the
documented mechanism. Walking flies in complex turbulent plumes navigate by the TIMING of odor encounters
instead; this step says nothing about that case and must not be quoted for it.

Geometry, declared before the run and identical in every arm. Chosen from fly anatomy and LK-14, never
tuned on performance:
    antenna separation  0.30 mm    (Drosophila head width ~0.7 mm)
    falloff             c(d) = (1 mm / d)^p, p = 2 primary; p = 1 and p = 3 reported as sensitivity
    start distance      5.0 mm primary; 2.0 and 10.0 reported
    step                0.25 mm per frame,  turn 30 deg per frame,  400 frames
    arrival             within 0.5 mm of the source
    start heading       uniform, but REJECTED if within 90 deg of facing the source, so not one episode
                        begins already pointed at the smell (LK-16)
WHY THE HEADING IS REJECTED, AND WHAT IT IS WORTH (LK-16). An episode that starts pointing at the source
tests nothing. Excluding the whole forward hemisphere costs the real wiring 0.682 -> 0.631 in the geometry
model and drops the chance floor 0.164 -> 0.137, moving the ratio 4.17x -> 4.62x. So free arrivals were
real but NOT dominant -- 30 deg of turn per frame washes the initial heading out within 20-30 frames. It
is adopted anyway: knowing the effect is small because it was measured is not the same as not having
looked, and it buys the plain sentence that no fly ever started facing the smell.

WHY p = 2 AND NOT A PLUME MODEL: a smooth point-source falloff is the regime the bilateral mechanism is
documented in, and it has one parameter instead of a turbulence model's many. LK-14 measured what this
geometry delivers: contrast 0.89 at 5 mm and 0.94 at 10 mm, which step 11's curve reads as per-step
accuracy 0.551 and 0.526 -- barely above chance. It works anyway because 400 steps integrate it: at a
FIXED per-step accuracy of 0.600, arrival from 5 mm is 0.968, while chance (0.500) gives 0.154.
That floor of 0.154 is why RANDOM-WALK is an arm and not an assumption.

Network. Step 12's dynamics at lam = 1, ONE Euler step per movement frame, state carried across frames:
    h <- h + 1.0 * (-h + ReLU(0.9 W h + 0.1 + u(t)))
which step 12 verified is step 11's fixed point to 2.98e-8. The network's own state is the ONLY memory in
the system; there is no stored variable outside it. The fly does not re-equilibrate its brain each step,
which is both cheaper and the less absurd of the two options.

Odor input: step 11's BALANCED panel, 402 ORNs per side cut to min(nL, nR) per receptor type from 1,282,
because an unbalanced panel tilts a left-minus-right readout before any odor arrives. ORN side from
`rootSide`, readout side from `somaSide` -- the two ends need different fields (LK-12). Left-side ORNs are
driven at c(dL), right-side at c(dR). Background is constant and does not scale. Held-out ORNs and
held-out odors only (step 11's test B split, seeds 101/202), so this is the closed-loop form of the exact
condition step 11 reported.

Readout and steering, zero parameters beyond one centering scalar per arm:
    d = (sum_L x - sum_R x) / (sum_L x + sum_R x),  x = h - h_blank
    turn LEFT iff d > m, else RIGHT.
m is estimated per arm on STATIC calibration presentations, half left-stronger and half right-stronger,
exactly as step 11 estimated it -- NOT inside the loop, which would be circular since the trajectory
depends on m.

Arms:
  REAL        baseline k>=3
  SHUF        presynaptic partners permuted, seeds 1000..1004
  RANDOM-WALK turns at random each frame. The floor, and a real one: LK-14 says it arrives 0.154 of the
              time from 5 mm, so "the fly reached the source" is not by itself evidence of anything.
  ORACLE      steers on the true concentrations cL, cR. The ceiling and the harness gate.

Verdicts, fixed before running. Rank tests against the five shuffled seeds, one-sided p ~ 1/6, the
convention steps 10 to 12 used. No absolute thresholds except N0's, which gates the harness.
  N0 harness works     ORACLE arrival at the primary condition > 0.90. Else STOP and read nothing else.
  N1 better than luck  REAL arrival above RANDOM-WALK's.
  N2 wiring earns it   REAL arrival above EVERY shuffled seed.
  N3 not one geometry  N2 also holds at start distances 2.0 and 10.0 mm.
  N4 makes progress    REAL's median PROGRESS, (start distance - final distance) / path length, is above
                       every shuffled seed. Defined for EVERY episode, arrived or not.
                       WHY NOT PATH EFFICIENCY AMONG ARRIVALS (found at smoke, before the run). The first
                       draft compared path length over straight-line distance among episodes that arrived,
                       and it is SELECTION-BIASED: an arm that rarely arrives only arrives when it happened
                       to start pointing at the source, so its few successes are straight by construction.
                       In the smoke SHUF arrived 0.125 of the time and scored 1.30 against REAL's 1.75,
                       i.e. the statistic REWARDED FAILING. It is still printed, labelled, as a reading.
                       This is the same species of defect as step 12's RATEACC and step 8's D3: a number
                       that looks like it measures quality while measuring something else.
  THE WIRING WALKS TO THE SMELL = N1 and N2 and N4.  N3 is read beside them.
Readings, not verdicts: per-step decision accuracy measured inside the loop, against step 11's static
  curve at the matched contrast -- does a one-shot number predict closed-loop behaviour at all; arrival
  under p = 1 and p = 3; time to arrival.

Run from data/ after retention.py (and step10_odor_delay.py for door/):
  step13_surge.py [--smoke] [--export=N]
--export=N: record the xy tracks of the first N episodes at the primary condition for the figure, with the
  same seed and the same 200-episode batch as the verdict run, so every track drawn is a SCORED episode.
  Computes no verdict and writes step13_paths.json.
"""
import csv, json, os, resource, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import fpmodel as fm

SMOKE = "--smoke" in sys.argv
EXPORT = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--export=")), 0))
UNITS = ["Or10a", "Or19a", "Or22a", "Or23a", "Or2a", "Or35a", "Or43a", "Or43b", "Or47a", "Or47b", "Or49b",
         "Or59b", "Or65a", "Or67a", "Or67c", "Or7a", "Or82a", "Or85a", "Or85b", "Or85f", "Or88a", "Or98a", "Or9a"]
RECEPTOR_TYPE = {"Or10a": "DL1", "Or19a": "DC1", "Or22a": "DM2", "Or23a": "DA3", "Or2a": "DA4m", "Or35a": "VC3",
                 "Or43a": "DA4l", "Or43b": "VM2", "Or47a": "DM3", "Or47b": "VA1v", "Or49b": "VA5", "Or59b": "DM4",
                 "Or65a": "DL3", "Or67a": "DM6", "Or67c": "VC4", "Or7a": "DL5", "Or82a": "VA6", "Or85a": "DM5",
                 "Or85b": "VM5d", "Or85f": "DL4", "Or88a": "VA1d", "Or98a": "VM5v", "Or9a": "VM3"}
SCALE, BG = 163, 0.5
S_ANT, STEP, TURN, FRAMES, ARRIVE, D0 = 0.30, 0.25, np.deg2rad(30), 400, 0.5, 1.0
THETA_MIN = np.deg2rad(90)        # no episode may start already pointing at the source
P_PRIMARY, R0_PRIMARY = 2.0, 5.0
P_ALL = [2.0] if SMOKE else [1.0, 2.0, 3.0]
R0_ALL = [5.0] if SMOKE else [2.0, 5.0, 10.0]
N_EP = 24 if SMOKE else 200
N_CAL = 24 if SMOKE else 400
SHUF_SEEDS = [1000] if SMOKE else [1000, 1001, 1002, 1003, 1004]
torch.set_num_threads(max(1, os.cpu_count() - 1))
gb = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
t0 = time.time()

os.makedirs("door", exist_ok=True)
resp, cls = {}, {}
for u in UNITS:
    f = f"door/{u}.csv"
    if not os.path.exists(f):
        subprocess.run(["curl", "-sfL", "-o", f,
                        f"https://raw.githubusercontent.com/ropensci/DoOR.data/v2.0.1/data/{u}.csv"], check=True)
    r = list(csv.reader(open(f), delimiter=";")); i = r[0].index("Hallem.2006.EN") + 1
    for x in r[1:]:
        if x[i] in ("NA", "") or x[3] == "SFR": continue
        resp.setdefault(x[3], {})[u] = float(x[i]); cls[x[3]] = x[1]
ODORS = sorted(o for o in resp if len(resp[o]) == len(UNITS))
R = np.array([[resp[o][u] for u in UNITS] for o in ODORS])

rng101 = np.random.default_rng(101)
by_cls = {}
for j, o in enumerate(ODORS): by_cls.setdefault(cls[o], []).append(j)
held = []
for c, js in sorted(by_cls.items()):
    js = rng101.permutation(js); held += list(js[: int(round(0.3 * len(js)))])
HELD_ODOR = np.sort(held)

net = fm.load_network("baseline", torch.float32); N = net.N
sc = pd.read_parquet("annotations.parquet", columns=["bodyId", "superclass", "rootSide", "somaSide"])
sc = sc[sc.superclass.notna()].sort_values("bodyId").reset_index(drop=True)
assert (sc.bodyId.to_numpy() == net.ann.bodyId.to_numpy()).all(), "row order"
ROOT = sc.rootSide.fillna("-").astype(str).to_numpy(); SOMA = sc.somaSide.fillna("-").astype(str).to_numpy()
typ, klass = net.ann.type.astype(str), net.ann["class"].astype(str)

r202 = np.random.default_rng(202)
POOLS = {}
for u in UNITS:                                              # step 11's split, seeds 202, reproduced
    idx = np.flatnonzero(typ.eq("ORN_" + RECEPTOR_TYPE[u]).to_numpy())
    L, Rr = idx[ROOT[idx] == "L"], idx[ROOT[idx] == "R"]
    k = min(len(L), len(Rr)); L, Rr = r202.permutation(L)[:k], r202.permutation(Rr)[:k]
    nh = int(round(0.3 * k))
    POOLS[u] = {"held": {"L": np.sort(L[:nh]), "R": np.sort(Rr[:nh])},
                "train": {"L": np.sort(L[nh:]), "R": np.sort(Rr[nh:])}}
PANEL = np.concatenate([p[s] for v in POOLS.values() for p in v.values() for s in "LR"])
OLF = np.flatnonzero(klass.eq("olfactory").to_numpy()); BGN = np.setdiff1d(OLF, PANEL)
LH = np.flatnonzero(typ.str.match(r"^LH(AV|AD|PV|PD)").to_numpy())
LH_L = np.flatnonzero(SOMA[LH] == "L"); LH_R = np.flatnonzero(SOMA[LH] == "R")
print(f"panel {len(PANEL)}; held per side {sum(len(POOLS[u]['held']['L']) for u in UNITS)}; "
      f"LH {len(LH)} = {len(LH_L)}L+{len(LH_R)}R; held odors {len(HELD_ODOR)}", flush=True)


def side_patterns(odors, which, seed):
    """Per episode: unit-concentration ORN patterns for the LEFT and RIGHT panels, plus a constant background."""
    rng = np.random.default_rng(seed)
    K = len(odors)
    OL, OR_, B = torch.zeros(N, K), torch.zeros(N, K), torch.zeros(N, K)
    for k, j in enumerate(odors):
        for ui, u in enumerate(UNITS):
            for s, T in (("L", OL), ("R", OR_)):
                p = POOLS[u][which][s]; on = p[rng.random(len(p)) < 0.5]
                T[torch.from_numpy(on.astype(np.int64)), k] = torch.from_numpy(
                    (np.full(len(on), R[j, ui] / SCALE) * rng.uniform(0.75, 1.25, len(on))).astype(np.float32))
        bg = BGN[rng.random(len(BGN)) < 0.05]
        B[torch.from_numpy(bg.astype(np.int64)), k] = torch.from_numpy(
            rng.uniform(0, BG, len(bg)).astype(np.float32))
    return OL, OR_, B


def encoder(seed):
    s_n = np.ones(N); s_n[net.pre] = np.sign(net.val)
    pre = net.pre if seed is None else np.random.default_rng(seed).permutation(net.pre)
    A = sp.csr_matrix(((0.9 * s_n[pre] * np.abs(net.val)).astype(np.float32), (net.post, pre)), shape=(N, N))
    return fm._torch_csr(A, torch.float32)


def stat(H, Hb):
    x = (H[LH] - Hb[LH])
    a, b = x[LH_L].sum(0), x[LH_R].sum(0)
    return ((a - b) / (a + b + 1e-12)).numpy()


def calibrate(W, seed=901):
    """m, the centering scalar: mean d over static presentations, half left-stronger, half right-stronger.
    Estimated OUTSIDE the loop, exactly as step 11 did, because inside it the trajectory depends on m."""
    ods = np.random.default_rng(seed).choice(HELD_ODOR, N_CAL)
    OL, OR_, B = side_patterns(ods, "held", seed + 1)
    strong = np.where(np.arange(N_CAL) % 2 == 0, 1.0, 0.5).astype(np.float32)
    weak = np.where(np.arange(N_CAL) % 2 == 0, 0.5, 1.0).astype(np.float32)
    U = OL * torch.from_numpy(strong) + OR_ * torch.from_numpy(weak) + B
    H = torch.zeros(N, N_CAL); Hb = torch.zeros(N, N_CAL)
    for _ in range(200):
        H = torch.relu(W @ H + 0.1 + U)
        Hb = torch.relu(W @ Hb + 0.1 + B)
    return float(stat(H, Hb).mean())


def conc(d, p):
    return (D0 / np.maximum(d, 1e-3)) ** p


def episode_batch(W, m, p, r0, seed, mode, keep_paths=0):
    """One batch of N_EP episodes run in parallel as columns. mode: net | random | oracle.
    keep_paths > 0 records the xy track of the first that many episodes, for the figure. It changes
    nothing about the run: the recorded episodes ARE scored episodes of the same batch."""
    track = []
    rng = np.random.default_rng(seed)
    ods = rng.choice(HELD_ODOR, N_EP)
    pos = np.zeros((N_EP, 2)); pos[:, 0] = r0
    th = np.empty(0)                                           # reject headings within THETA_MIN of the source
    while len(th) < N_EP:
        c = rng.uniform(0, 2 * np.pi, N_EP * 3)
        th = np.concatenate([th, c[np.abs(np.angle(np.exp(1j * (c - np.pi)))) >= THETA_MIN]])
    th = th[:N_EP]
    done = np.zeros(N_EP, bool); tarr = np.full(N_EP, FRAMES, float); plen = np.zeros(N_EP)
    hits = np.zeros(N_EP); shots = np.zeros(N_EP)
    if mode == "net":
        OL, OR_, B = side_patterns(ods, "held", seed + 7)
        H = torch.zeros(N, N_EP); Hb = torch.zeros(N, N_EP)
    for t in range(FRAMES):
        dist = np.linalg.norm(pos, axis=1)
        newly = (~done) & (dist < ARRIVE)
        tarr[newly] = t; done |= newly
        if done.all(): break
        # antenna positions: perpendicular to heading, half the separation each way
        nx, ny = -np.sin(th), np.cos(th)
        aL = pos + (S_ANT / 2) * np.stack([nx, ny], 1)
        aR = pos - (S_ANT / 2) * np.stack([nx, ny], 1)
        cL = conc(np.linalg.norm(aL, axis=1), p); cR = conc(np.linalg.norm(aR, axis=1), p)
        truth = cL > cR                                        # which antenna really has more
        if mode == "net":
            U = OL * torch.from_numpy(cL.astype(np.float32)) + OR_ * torch.from_numpy(cR.astype(np.float32)) + B
            H = H + 1.0 * (-H + torch.relu(W @ H + 0.1 + U))   # lam = 1, state carried across frames
            Hb = Hb + 1.0 * (-Hb + torch.relu(W @ Hb + 0.1 + B))
            go_left = stat(H, Hb) > m
        elif mode == "oracle":
            go_left = truth
        else:
            go_left = rng.random(N_EP) < 0.5
        live = ~done
        hits += live & (go_left == truth); shots += live        # per-step decision accuracy, a reading
        th = th + np.where(go_left, TURN, -TURN)
        pos[live] += STEP * np.stack([np.cos(th[live]), np.sin(th[live])], 1)
        plen[live] += STEP
        if keep_paths: track.append(np.round(pos[:keep_paths], 4).copy())
    d_final = np.linalg.norm(pos, axis=1)
    eff = np.where(done, plen / r0, np.nan)
    progress = (r0 - d_final) / np.maximum(plen, 1e-9)         # net gain per mm walked, defined for EVERY episode
    return {"arrival": float(done.mean()), "t_arrive": float(np.nanmedian(np.where(done, tarr, np.nan))),
            "d_final": float(np.median(d_final)), "progress": float(np.median(progress)),
            "path_eff": float(np.nanmedian(eff)) if done.any() else None,
            "step_acc": float(hits.sum() / max(shots.sum(), 1)), "n": int(N_EP),
            **({"paths": np.stack(track, 1).tolist(),
                "arrived": done[:keep_paths].tolist()} if keep_paths else {})}


if EXPORT:
    # Same seed, same batch size, same condition as the verdict run, so every recorded track is one of the
    # scored episodes. Only the primary condition is run, and no verdict is computed.
    out = {"condition": {"p": P_PRIMARY, "r0": R0_PRIMARY, "n_shown": EXPORT, "n_scored": N_EP},
           "geometry": {"s_ant": S_ANT, "step": STEP, "turn_deg": 30, "frames": FRAMES,
                        "arrive": ARRIVE, "theta_min_deg": 90}, "arms": {}}
    for nm, md, sd in (("ORACLE", "oracle", 950), ("RANDOM-WALK", "random", 951)):
        out["arms"][nm] = episode_batch(None, 0.0, P_PRIMARY, R0_PRIMARY, sd, md, keep_paths=EXPORT)
        print(f"  {nm} arrival {out['arms'][nm]['arrival']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    for nm, sd in (("REAL", None), ("SHUF-1000", 1000)):
        W = encoder(sd); m = calibrate(W)
        out["arms"][nm] = episode_batch(W, m, P_PRIMARY, R0_PRIMARY, 960, "net", keep_paths=EXPORT)
        out["arms"][nm]["m"] = m
        print(f"  {nm} arrival {out['arms'][nm]['arrival']:.3f} ({time.time()-t0:.0f}s)", flush=True)
        del W
    json.dump(out, open("step13_paths.json", "w"))
    print(f"wrote step13_paths.json ({time.time()-t0:.0f}s, {gb():.2f} GB)")
    sys.exit(0)

res, t_arm = {}, {}
CONDS = [(P_PRIMARY, R0_PRIMARY)] + [(p, R0_PRIMARY) for p in P_ALL if p != P_PRIMARY] \
         + [(P_PRIMARY, r) for r in R0_ALL if r != R0_PRIMARY]
print(f"conditions: {CONDS}", flush=True)

res["ORACLE"] = {f"{p}|{r}": episode_batch(None, 0.0, p, r, 950, "oracle") for p, r in CONDS}
res["RANDOM-WALK"] = {f"{p}|{r}": episode_batch(None, 0.0, p, r, 951, "random") for p, r in CONDS}
for a in ("ORACLE", "RANDOM-WALK"):
    print(f"  {a:12s} " + "  ".join(f"p{p}/r{r} arr {res[a][f'{p}|{r}']['arrival']:.3f}" for p, r in CONDS), flush=True)

for arm, seed in [("REAL", None)] + [(f"SHUF-{s}", s) for s in SHUF_SEEDS]:
    W = encoder(seed); m = calibrate(W)
    res[arm] = {f"{p}|{r}": episode_batch(W, m, p, r, 960, "net") for p, r in CONDS}
    res[arm]["m"] = m
    print(f"  {arm:12s} m {m:+.4f}  " + "  ".join(
        f"p{p}/r{r} arr {res[arm][f'{p}|{r}']['arrival']:.3f} eff "
        f"{(res[arm][f'{p}|{r}']['path_eff'] or float('nan')):.2f}" for p, r in CONDS)
        + f"   ({time.time() - t0:.0f}s, {gb():.2f} GB)", flush=True)
    del W

K = f"{P_PRIMARY}|{R0_PRIMARY}"
g = lambda a, k=K, f="arrival": res[a][k][f]
V = {}
V["N0 harness works"] = g("ORACLE") > 0.90
if not V["N0 harness works"]:
    print(f"\nN0 FAILED: ORACLE arrival {g('ORACLE'):.3f} <= 0.90. Reading nothing else.")
    json.dump({"verdicts": {"N0 harness works": False}, "arms": res}, open("step13_surge.json", "w"), indent=1)
    sys.exit(1)
V["N1 better than luck"] = g("REAL") > g("RANDOM-WALK")
V["N2 wiring earns it"] = all(g("REAL") > g(f"SHUF-{s}") for s in SHUF_SEEDS)
V["N3 not one geometry"] = all(
    all(g("REAL", f"{P_PRIMARY}|{r}") > g(f"SHUF-{s}", f"{P_PRIMARY}|{r}") for s in SHUF_SEEDS)
    for r in R0_ALL) if not SMOKE else None
V["N4 makes progress"] = all(res["REAL"][K]["progress"] > res[f"SHUF-{s}"][K]["progress"] for s in SHUF_SEEDS)
V["THE WIRING WALKS TO THE SMELL"] = V["N1 better than luck"] and V["N2 wiring earns it"] and V["N4 makes progress"]

print("\n-- verdicts --")
for k, v in V.items():
    print(f"  {k:30s} {'PASS' if v else ('n/a' if v is None else 'FAIL')}")
print(f"\n-- primary condition p={P_PRIMARY}, start {R0_PRIMARY} mm --")
for a in ["REAL"] + [f"SHUF-{s}" for s in SHUF_SEEDS] + ["RANDOM-WALK", "ORACLE"]:
    r = res[a][K]
    print(f"  {a:12s} arrival {r['arrival']:.3f}  progress {r['progress']:+.4f}  final dist {r['d_final']:.2f}  "
          f"median frames {r['t_arrive']:.0f}  in-loop acc {r['step_acc']:.3f}  "
          f"[path/straight among arrivals {(r['path_eff'] or float('nan')):.2f}, SELECTION-BIASED]")
print(f"\nreading: step 11's STATIC curve reads contrast at {R0_PRIMARY} mm as per-step accuracy ~0.551; "
      f"REAL measured {res['REAL'][K]['step_acc']:.3f} inside the loop.")
json.dump({"verdicts": {k: (None if v is None else bool(v)) for k, v in V.items()}, "arms": res,
           "geometry": {"s_ant": S_ANT, "step": STEP, "turn_deg": 30, "frames": FRAMES, "arrive": ARRIVE}},
          open("step13_surge.json", "w"), indent=1)
print(f"\nwrote step13_surge.json ({time.time() - t0:.0f}s, {gb():.2f} GB)")
