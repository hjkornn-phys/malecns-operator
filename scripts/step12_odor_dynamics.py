"""Step 12: a time axis on olfaction. Does the wiring carry the RATE of concentration change?

Step 8 put a time axis on the escape pathway and found nothing, but that pathway was already silent before
time was added: LK-8 measured DNp01_R's entire range at 0.0002 to 0.0061 and LPLC2 never cleared its own
shuffled null at any radius. Olfaction is the opposite case -- LK-11 found 1,648 of 1,929 lateral horn
output neurons past 1e-3 on the real wiring, 5.7x above shuffled, and step 11 beat ten null seeds with it.
This step asks the step 8 question where the signal is actually alive. Step 8's result is NOT assumed to
carry over and none of its numbers are reused; LK-13 measured olfaction's own.

Nothing is trained. The network is run forward only: no stored activation, no BPTT, no outside box. The
statistic below has ZERO parameters -- not even step 11's one centering scalar.

LAMBDA: WHY THE VERDICTS RUN AT lam = 1 (found at smoke, before the run). The imposed per-neuron leak is
IDENTICAL in every arm, so making it large adds lag that the wiring did not produce and swamps the lag it
did. At the slowed lam = 0.1353 the smoke measured HYST 0.855 for real against 0.723 for one shuffled seed,
a ratio of 1.18; at lam = 1 the LK-13 probe measured 0.166 against 0.018, a ratio of 9. Same wiring, same
stimulus -- the slower the imposed leak, the less of the difference is the wiring. So every primary verdict
runs at lam = 1, where the only source of lag is the network's own recurrence.
THE TENSION THIS CREATES, STATED RATHER THAN HIDDEN. A tau spread needs lam < 1 to be stable at all:
max lam = lam0 / min(m) must stay at or below 1, so sigma > 0 forces lam0 < 1. The tau question can
therefore only be asked in the regime where discrimination is already compressed. E5 is asked there anyway,
at a constant lam0 = 0.1353 across every sigma, and it is a LOW-POWER test by construction. That is a
measured property of the model, not a defect of the run, and no strong reading may be taken from it either
way.

Model, Euler on a leaky rate unit, one lambda per cell TYPE, exactly step 8's form:
    h <- h + lam * (-h + ReLU(0.9 W h + 0.1 + u(t))),    W[post, pre] = sign(pre) * w / in_tot_full[post]
    m_t = exp(sigma * z_t), z_t ~ N(0,1) at seed 800 CLIPPED to +-2, centred so mean log m = 0
    lam_i = LAM0 / m_type(i)
Which type is fast is never chosen by hand: the assignment is random at a seed, and a result needing a
particular hand-made assignment is not a wiring result (step 8's rule, kept).

THE STEP 8 D6 DEFECT, AND THE FIX. Step 8 swept tau spread and global slowing together, so its NONE IN
RANGE said nothing about tau. That happens when LAM0 = min(m), because min(m) shrinks as sigma grows and
the whole network slows with it. Here LAM0 IS A CONSTANT, 0.1353 = exp(-2 * SIGMA_MAX), sized for the
LARGEST sigma tested and reused unchanged at every sigma including 0. Since mean log m = 0, mean log lam
is then identical across sigma and spread is the only thing that varies. z is clipped to +-2 because with
11,751 cell types an unclipped draw reaches m_min ~ exp(-4), which would force LAM0 so small that the
network could not follow the fastest ramp at all.
WHAT A PER-TYPE TAU IS AND IS NOT (LK-13). 11,751 types over 166,700 neurons means a per-type draw is
nearly a per-neuron draw. This is structured noise, not a claim that tau varies by cell type in the fly.

Stimulus, a TRIANGLE in concentration, and why it is not loom/recede:
    0 -> 1 over s frames, one frame at 1, 1 -> 0 over s frames, then 0 to the end.  T = 260.
    Ramp speeds s = 10, 25, 60, every trial the same T.
THE STEP 8 D1 DEFECT, AND THE FIX. Step 8's D1 passed on an onset transient because loom and recede were
not matched at onset, and a shuffled null could not produce the artefact. A rising stimulus starts from
nothing and a falling one from full, so part of any difference is arithmetic. Here the two phases are
compared ONLY at frames carrying the SAME instantaneous concentration, within the same trial and the same
odor sample. Level is held constant by construction and what remains is the sign of the rate.
Responses are taken against a BLANK column at the MATCHING FRAME (step 8's convention), which removes the
startup transient shared by every column.

Odor input: the step 10 panel unchanged (LK-10, LK-11) -- Hallem via DoOR v2.0.1, 23 receptors, 1,282 ORNs,
SCALE 163, BG 0.5. Background is constant over time and does NOT ramp; only the odor scales with c(t).
60 odors drawn at seed 802 from the 110. NO TRAIN/TEST SPLIT IS USED OR NEEDED: nothing is fitted, so there
is nothing to leak. The subset falling in step 10's 33 held-out odors is reported separately as a reading so
the number stays comparable with steps 10 and 11.

Readout: step 10's lateral horn output, ^LH(AV|AD|PV|PD), 1,929 bodies. x = h(t) - h_blank(t).
Statistic, zero parameters. At every matched concentration c of a trial, r_up = mean over LH of |x| on the
rising frame and r_dn the same on the falling frame.
    HYST    = mean over (odor, c) of (r_dn - r_up) / (r_dn + r_up)     -- every verdict uses this
    RATEACC = the fraction of (odor, c) pairs with r_dn > r_up          -- A READING, NEVER A VERDICT
RATEACC SATURATES AND IS NOT A VERDICT (found at smoke, before the run). Both the real wiring and a
shuffled seed scored 1.000 at every ramp speed: at matched concentration the lagging limb is higher
essentially always, so the fraction carries no information once the lag exists at all. Step 8's D3 made
exactly this mistake -- it demanded a strict win on a statistic both sides had saturated at 1.000 -- and
PR-8's E2 was written on RATEACC before smoke showed the same thing here. It is kept as a reading so the
saturation stays visible, and every verdict is on HYST, which is continuous and did not saturate.
WHY PEAK IS NOT USED (LK-13, and this is the measurement that matters). Peak |x| over all frames was 0.140
on the real wiring against 0.121 and 0.109 on two shuffled seeds -- not separated. The whole separation
lives in the matched-concentration comparison, and a step 8-style peak statistic would have found nothing
here. E4 below registers that as a verdict expected to FAIL, so it is on the record rather than in a
footnote.

WHAT A PASS HERE LICENSES, AND WHAT IT DOES NOT. A normalised hysteresis above shuffled says the real
wiring integrates more slowly than its degree-matched shuffles -- stronger, more coherent recurrence, a
graph-level property. It does NOT say the wiring computes a derivative, that any downstream neuron reads
one, or that the fly uses it. The claim is that the material for a rate signal survives the wiring and does
not survive shuffling.

WHY THERE ARE NO ABSOLUTE THRESHOLDS. Step 4's E1 set a threshold without asking what the measurement could
resolve. Every verdict below is a rank test against the five shuffled seeds or an ordering, per METHOD.md.

Arms: REAL, baseline k>=3; SHUF, presynaptic partners permuted, seeds 1000..1004 (degrees and counts kept).

Verdicts, fixed before running. One-sided p ~ 1/6 for each rank test, the convention steps 10 and 11 used.
  E0 same model         at lam = 1 uniform, the final state reproduces step 11's fixed-point solve to max
                        abs difference < 1e-4. LK-13 measured 1.5e-8. If this fails the run STOPS.
  E1 signal above floor REAL's mean |x| over LH at the FIXED POINT with the odor held at full
                        concentration is above every shuffled seed. Measured at the fixed point and not on
                        a ramp frame: at a fast ramp the network has not settled, so a ramp frame measures
                        the timescale, not the response level (found at smoke).
                        This is the check step 8's pathway would have failed before it began.
  E2 rate is read       REAL's HYST at the fastest ramp (lam 1, sigma 0, s 10) is above every shuffled seed.
  E3 scales with rate   REAL's HYST is strictly decreasing over s = 10, 25, 60 at lam 1, sigma 0. A rate
                        signal must fade as the ramp slows; a fixed lag need not order at all.
  E4 peak is not it     REAL's peak |x| (lam 1, sigma 0, s 10) is above every shuffled seed.
                        REGISTERED AS EXPECTED TO FAIL. Its job is to document that the statistic step 8
                        used is blind here. A PASS would mean LK-13's peak reading was wrong and E2's
                        margin needs re-reading, not that the result is stronger.
  E5 holds over tau     E2's comparison, on HYST, holds at sigma = 0, 0.5 AND 1.0 with lam0 held constant
                        at 0.1353. Asks robustness to spread, not the value of tau -- which step 8 could
                        not ask at all. LOW POWER by construction, see the lambda note above.
  THE WIRING CARRIES RATE = E1 and E2 and E3. E4 is expected to fail and does not gate. E5 is read beside.
Readings, not verdicts: HYST spread across odors, and the correlation of the per-LH-neuron hysteresis
  pattern between odors -- a high correlation means one global lag rather than an odor-specific signal;
  the 60-odor numbers restricted to step 10's held-out 33.

Run from data/ after retention.py (and step10_odor_delay.py for door/):
  step12_odor_dynamics.py [--smoke] [--lookup]
--lookup: LK-13's own measurements, two shuffled seeds, no verdicts.
--smoke: 6 odors, one shuffled seed, sigma 0 only.
"""
import csv, json, os, resource, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import fpmodel as fm

SMOKE, LOOKUP = "--smoke" in sys.argv, "--lookup" in sys.argv
UNITS = ["Or10a", "Or19a", "Or22a", "Or23a", "Or2a", "Or35a", "Or43a", "Or43b", "Or47a", "Or47b", "Or49b",
         "Or59b", "Or65a", "Or67a", "Or67c", "Or7a", "Or82a", "Or85a", "Or85b", "Or85f", "Or88a", "Or98a", "Or9a"]
RECEPTOR_TYPE = {"Or10a": "DL1", "Or19a": "DC1", "Or22a": "DM2", "Or23a": "DA3", "Or2a": "DA4m", "Or35a": "VC3",
                 "Or43a": "DA4l", "Or43b": "VM2", "Or47a": "DM3", "Or47b": "VA1v", "Or49b": "VA5", "Or59b": "DM4",
                 "Or65a": "DL3", "Or67a": "DM6", "Or67c": "VC4", "Or7a": "DL5", "Or82a": "VA6", "Or85a": "DM5",
                 "Or85b": "VM5d", "Or85f": "DL4", "Or88a": "VA1d", "Or98a": "VM5v", "Or9a": "VM3"}
SCALE, BG, T, Z_CLIP, SIGMA_MAX = 163, 0.5, 260, 2.0, 1.0
LAM_SLOW = float(np.exp(-Z_CLIP * SIGMA_MAX))              # 0.1353, constant across every sigma, tau arm only
SPEEDS = [10, 25, 60]
SIGMAS = [0.0, 0.5, 1.0]
N_ODOR = 6 if SMOKE else 60
SHUF_SEEDS = [1000] if SMOKE else [1000, 1001, 1002, 1003, 1004]
PRIMARY = [(1.0, 0.0, sp) for sp in SPEEDS]                # lam0, sigma, ramp speed
TAU_ARM = [] if SMOKE else [(LAM_SLOW, sg, SPEEDS[0]) for sg in SIGMAS]
COMBOS = PRIMARY + TAU_ARM
torch.set_num_threads(max(1, os.cpu_count() - 1))
gb = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
t_start = time.time()

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
assert R.shape == (110, 23), R.shape

rng101 = np.random.default_rng(101)                        # step 10's split, for the reading only
by_cls = {}
for j, o in enumerate(ODORS): by_cls.setdefault(cls[o], []).append(j)
held = []
for c, js in sorted(by_cls.items()):
    js = rng101.permutation(js); held += list(js[: int(round(0.3 * len(js)))])
HELD_ODOR = set(int(x) for x in held)

net = fm.load_network("baseline", torch.float32); N = net.N
typ, klass = net.ann.type.astype(str), net.ann["class"].astype(str)
PANEL = np.concatenate([np.flatnonzero(typ.eq("ORN_" + RECEPTOR_TYPE[u]).to_numpy()) for u in UNITS])
OLF = np.flatnonzero(klass.eq("olfactory").to_numpy()); BGN = np.setdiff1d(OLF, PANEL)
LH = np.flatnonzero(typ.str.match(r"^LH(AV|AD|PV|PD)").to_numpy())
CODES = pd.Categorical(net.ann.type.astype(str)).codes.astype(np.int64)
NTYPES = int(CODES.max()) + 1
assert (len(PANEL), len(BGN), len(LH)) == (1282, 1357, 1929)
print(f"panel {len(PANEL)} bg {len(BGN)} LH {len(LH)}; {NTYPES} cell types; LAM_SLOW {LAM_SLOW:.4f}", flush=True)

rng802 = np.random.default_rng(802)
ODOR_IDX = np.sort(rng802.choice(110, N_ODOR, replace=False))
IS_HELD = np.array([int(j) in HELD_ODOR for j in ODOR_IDX])


def lam_vec(sigma, lam0, seed=800):
    if sigma == 0.0:
        lam = np.full(N, lam0)
    else:
        z = np.clip(np.random.default_rng(seed).normal(size=NTYPES), -Z_CLIP, Z_CLIP)
        m = np.exp(sigma * (z - z.mean()))
        lam = (lam0 / m)[CODES]
    assert lam.max() <= 1.0 + 1e-9, f"unstable: max lam {lam.max()}"
    return torch.from_numpy(lam.astype(np.float32)).reshape(N, 1)


def encoder(seed):
    s_neuron = np.ones(N); s_neuron[net.pre] = np.sign(net.val)
    pre = net.pre if seed is None else np.random.default_rng(seed).permutation(net.pre)
    A = sp.csr_matrix(((0.9 * s_neuron[pre] * np.abs(net.val)).astype(np.float32), (net.post, pre)), shape=(N, N))
    return fm._torch_csr(A, torch.float32)


def patterns():
    """Column 0 is the BLANK: background only, no odor. Columns 1.. are the odors at unit concentration."""
    rng = np.random.default_rng(803)
    O = torch.zeros(N, len(ODOR_IDX) + 1); B = torch.zeros(N, len(ODOR_IDX) + 1)
    for k, j in enumerate(ODOR_IDX, start=1):
        for ui, u in enumerate(UNITS):
            p = np.flatnonzero(typ.eq("ORN_" + RECEPTOR_TYPE[u]).to_numpy())
            on = p[rng.random(len(p)) < 0.5]
            O[torch.from_numpy(on.astype(np.int64)), k] = torch.from_numpy(
                (np.full(len(on), R[j, ui] / SCALE) * rng.uniform(0.75, 1.25, len(on))).astype(np.float32))
    for k in range(len(ODOR_IDX) + 1):                     # every column gets its own background draw
        bg = BGN[rng.random(len(BGN)) < 0.05]
        B[torch.from_numpy(bg.astype(np.int64)), k] = torch.from_numpy(
            rng.uniform(0, BG, len(bg)).astype(np.float32))
    return O, B


O_PAT, B_PAT = patterns()


def triangle(s):
    p = np.zeros(T)
    p[:s] = np.linspace(0, 1.0, s, endpoint=False)
    p[s] = 1.0
    p[s + 1: 2 * s + 1] = np.linspace(1.0, 0.0, s, endpoint=False)
    return p


def trajectory(W, lam, prof):
    """Returns |x| averaged over LH, T x K, plus the peak over LH and frames. Column 0 is blank."""
    H = torch.zeros(N, O_PAT.shape[1]); mean_abs, peak = [], 0.0
    for t in range(T):
        U = O_PAT * float(prof[t]) + B_PAT
        H = H + lam * (-H + torch.relu(W @ H + 0.1 + U))
        x = H[LH] - H[LH][:, :1]                           # against the blank column at the MATCHING frame
        mean_abs.append(x[:, 1:].abs().mean(0).clone())
        peak = max(peak, float(x[:, 1:].abs().max()))
    return torch.stack(mean_abs).numpy(), peak


def matched(prof, s):
    """Frame pairs carrying the same instantaneous concentration on the rising and falling limbs."""
    up = np.arange(1, s); dn = 2 * s + 1 - up
    keep = np.abs(prof[up] - prof[dn]) < 1e-9
    return up[keep], dn[keep]


def measure(W, lam0, sigma, s):
    lam = lam_vec(sigma, lam0); prof = triangle(s)
    ma, peak = trajectory(W, lam, prof)
    up, dn = matched(prof, s)
    a, b = ma[up], ma[dn]                                  # both (n_matched, n_odor)
    d = (b - a) / (a + b + 1e-12)
    return {"hyst": float(d.mean()), "hyst_sd": float(d.std()), "rateacc": float((b > a).mean()),
            "peak": peak, "at_cmax": float(ma[s].mean()),
            "hyst_held": float(d[:, IS_HELD].mean()) if IS_HELD.any() else None,
            "per_odor": d.mean(0).tolist()}


def static_level(W):
    """Mean |x| over LH at the FIXED POINT with the odor held at full concentration."""
    Hf = torch.zeros(N, O_PAT.shape[1])
    for _ in range(3000):
        Hn = torch.relu(W @ Hf + 0.1 + O_PAT + B_PAT)
        d = (Hn - Hf).abs().max().item(); Hf = Hn
        if d < 1e-6: break
    x = Hf[LH] - Hf[LH][:, :1]
    return float(x[:, 1:].abs().mean())


# ---- E0: at lam = 1 the dynamics must be step 11's fixed point
def e0_check(W):
    lam1 = lam_vec(0.0, 1.0)
    H = torch.zeros(N, O_PAT.shape[1])
    for _ in range(600):
        H = H + lam1 * (-H + torch.relu(W @ H + 0.1 + O_PAT + B_PAT))
    Hf = torch.zeros(N, O_PAT.shape[1])
    for _ in range(3000):
        Hn = torch.relu(W @ Hf + 0.1 + O_PAT + B_PAT)
        d = (Hn - Hf).abs().max().item(); Hf = Hn
        if d < 1e-6: break
    return float((H[LH] - Hf[LH]).abs().max())


if LOOKUP:
    print("\n-- LK-13, no verdicts --")
    for label, seed in (("REAL", None), ("SHUF-1000", 1000), ("SHUF-1001", 1001)):
        W = encoder(seed)
        print(f"\n{label}: lam=1 vs fixed point {e0_check(W):.3e}")
        for l0, sg, s in COMBOS:
            m = measure(W, l0, sg, s)
            print(f"  lam0 {l0:.4f} sigma {sg} s {s:3d}: at_cmax {m['at_cmax']:.3e}  peak {m['peak']:.3e}  "
                  f"HYST {m['hyst']:+.4f} (sd {m['hyst_sd']:.4f})  RATEACC {m['rateacc']:.4f}", flush=True)
        print(f"  ({time.time() - t_start:.0f}s, {gb():.2f} GB)")
    sys.exit(0)

res = {}
W_real = encoder(None)
e0 = e0_check(W_real)
print(f"\nE0 lam=1 vs fixed point: {e0:.3e}", flush=True)
if not e0 < 1e-4:
    print("E0 FAILED: the dynamics is not the same model. Stopping.")
    json.dump({"E0": e0, "verdicts": {"E0 same model": False}}, open("step12_odor_dynamics.json", "w"), indent=1)
    sys.exit(1)

for arm, seed in [("REAL", None)] + [(f"SHUF-{s}", s) for s in SHUF_SEEDS]:
    W = W_real if seed is None else encoder(seed)
    res[arm] = {f"{l0}|{sg}|{s}": measure(W, l0, sg, s) for l0, sg, s in COMBOS}
    res[arm]["static"] = static_level(W)
    r = res[arm]
    print(f"  {arm:10s} static {r['static']:.3e}  " + "  ".join(
        f"s{s} hyst {r[f'{l0}|{sg}|{s}']['hyst']:+.4f}" for l0, sg, s in PRIMARY)
          + f"   ({time.time() - t_start:.0f}s, {gb():.2f} GB)", flush=True)
    if seed is not None: del W

g = lambda arm, l0, sg, s, k: res[arm][f"{l0}|{sg}|{s}"][k]
above = lambda l0, sg, s, k: all(g("REAL", l0, sg, s, k) > g(f"SHUF-{x}", l0, sg, s, k) for x in SHUF_SEEDS)
V = {}
V["E0 same model"] = e0 < 1e-4
V["E1 signal above floor"] = all(res["REAL"]["static"] > res[f"SHUF-{x}"]["static"] for x in SHUF_SEEDS)
V["E2 rate is read"] = above(1.0, 0.0, SPEEDS[0], "hyst")
hy = [g("REAL", 1.0, 0.0, s, "hyst") for s in SPEEDS]
V["E3 scales with rate"] = all(hy[i] > hy[i + 1] for i in range(len(hy) - 1))
V["E4 peak is not it"] = above(1.0, 0.0, SPEEDS[0], "peak")
V["E5 holds over tau"] = all(above(LAM_SLOW, sg, SPEEDS[0], "hyst") for sg in SIGMAS) if not SMOKE else None
V["THE WIRING CARRIES RATE"] = V["E1 signal above floor"] and V["E2 rate is read"] and V["E3 scales with rate"]

print("\n-- verdicts --")
for k, v in V.items():
    tag = "PASS" if v else ("n/a" if v is None else "FAIL")
    note = "  (expected to FAIL; a PASS re-opens E2)" if k.startswith("E4") else ""
    print(f"  {k:26s} {tag}{note}")
print("\n-- REAL over ramp speed, sigma 0 --")
for s in SPEEDS:
    m = res["REAL"][f"1.0|0.0|{s}"]
    print(f"  s {s:3d}: HYST {m['hyst']:+.4f}  RATEACC {m['rateacc']:.4f}  at_cmax {m['at_cmax']:.3e}  "
          f"peak {m['peak']:.3e}  held-33 HYST {m['hyst_held']:+.4f}")
po = np.array(res["REAL"][f"1.0|0.0|{SPEEDS[0]}"]["per_odor"])
print(f"\nreading: per-odor HYST spread at s {SPEEDS[0]} -- mean {po.mean():+.4f} sd {po.std():.4f} "
      f"min {po.min():+.4f} max {po.max():+.4f}  (a small sd means one global lag, not an odor-specific signal)")
json.dump({"E0": e0, "LAM_SLOW": LAM_SLOW, "verdicts": {k: (None if v is None else bool(v)) for k, v in V.items()},
           "arms": res, "odors": ODOR_IDX.tolist(), "held_mask": IS_HELD.tolist()},
          open("step12_odor_dynamics.json", "w"), indent=1)
print(f"\nwrote step12_odor_dynamics.json ({time.time() - t_start:.0f}s, {gb():.2f} GB)")
