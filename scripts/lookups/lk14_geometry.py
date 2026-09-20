"""LK-14 and LK-16: what bilateral contrast a navigating fly gets, and whether a shorter start helps.

Pure geometry, no network. Step 11's measured accuracy-vs-contrast curve on test B drives a walker, and
the question is whether per-step accuracy that is barely above chance still accumulates over 400 frames.
Run: python3 lk14_geometry.py
"""
import numpy as np

C_MEAS = np.array([0.00, 0.25, 0.50, 0.80])          # step 11 REAL test B, contrast -> accuracy
A_MEAS = np.array([0.862, 0.805, 0.715, 0.591])
S_ANT = 0.30                                          # mm, antenna separation


def acc_of(c):
    c = np.clip(c, 0, 1)
    a = np.interp(c, C_MEAS, A_MEAS)
    t = c > 0.8
    a[t] = np.interp(c[t], [0.8, 1.0], [0.591, 0.5])
    return a


def init_headings(n, theta_min, rng):
    """Reject starts within theta_min of facing the source, so no episode begins pointed at it."""
    th = np.empty(0)
    while len(th) < n:
        c = rng.uniform(0, 2 * np.pi, n * 3)
        off = np.abs(np.angle(np.exp(1j * (c - np.pi))))
        th = np.concatenate([th, c[off >= theta_min]])
    return th[:n]


def sim(p, r0, theta_min=0.0, n=8000, max_steps=400, step=0.25, turn=np.deg2rad(30), seed=0, force=None):
    rng = np.random.default_rng(seed)
    pos = np.zeros((n, 2)); pos[:, 0] = r0
    th = init_headings(n, theta_min, rng)
    done = np.zeros(n, bool)
    for _ in range(max_steps):
        d = np.linalg.norm(pos, axis=1); done |= d < 0.5
        if done.all(): break
        bear = np.arctan2(-pos[:, 1], -pos[:, 0]) - th
        left = np.sin(bear); off = np.abs(left) * S_ANT / 2
        contrast = (np.maximum(d - off, 1e-3) / (d + off)) ** p
        a = acc_of(contrast) if force is None else np.full(n, force)
        ok = rng.random(n) < a
        sgn = np.sign(left); sgn[sgn == 0] = 1
        th = th + np.where(ok, sgn, -sgn) * turn
        mv = ~done
        pos[mv] += step * np.stack([np.cos(th[mv]), np.sin(th[mv])], 1)
    return done.mean()


if __name__ == "__main__":
    print("contrast the geometry delivers, and what step 11's curve reads it as (source at 90 deg)")
    print(f"{'p':>3} {'r (mm)':>7} {'contrast':>10} {'acc':>7}")
    for p in (1.0, 2.0, 3.0):
        for r in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0):
            c = ((r - S_ANT / 2) ** -p) and min((r - S_ANT / 2) ** -p, (r + S_ANT / 2) ** -p) / \
                max((r - S_ANT / 2) ** -p, (r + S_ANT / 2) ** -p)
            print(f"{p:>3.0f} {r:>7.1f} {c:>10.4f} {float(acc_of(np.array([c]))[0]):>7.3f}")
        print()
    print("does it accumulate? arrival over 400 frames\n")
    print(f"{'THETA':>6} {'r0':>5} {'REAL':>7} {'floor':>7} {'ratio':>7}")
    for th_deg in (0, 30, 45, 60, 90):
        tm = np.deg2rad(th_deg)
        for r0 in (2.0, 3.0, 5.0, 10.0):
            real = sim(2.0, r0, tm); floor = sim(2.0, r0, tm, force=0.5, seed=1)
            print(f"{th_deg:>5}d {r0:>5.1f} {real:>7.3f} {floor:>7.3f} {real / max(floor, 1e-9):>7.2f}")
        print()
