"""Schematic animation for 'AI-Driven Scenario Discovery' (Energy & Buildings 2025).

Left: disturbance-scenario latent space. Real data (grey) is augmented by
diffusion samples (blue) that denoise from noise; CUCB rounds then try
super arms of K scenarios (orange rings) and keep the most diverse (colors).
Right: closed-loop room temperature for the kept scenarios.
Usage: python animate.py OUT_DIR
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(7)
FPS = 20
K = 5

# ---------- data -----------------------------------------------------------
def manifold(n, spread):
    """Points along a curved band; spread widens the arc that is covered."""
    s = rng.uniform(-spread, spread, n)
    x = s
    y = 0.55 * s**2 - 0.6 + rng.normal(0, 0.12 + 0.05 * abs(spread), n)
    return np.c_[x, y]

real = manifold(55, 0.55)
real += rng.normal(0, 0.05, real.shape)
syn_target = manifold(260, 1.55)
syn_start = rng.normal(0, 1.1, syn_target.shape) + [0, 0.2]
pool = np.vstack([real, syn_target])

def traj(z, t):
    """Closed-loop room temperature for a scenario at latent point z."""
    x, y = z
    peak = 14 + 2.6 * x                       # timing of the heat-load peak
    amp = 1.2 + 1.5 * (y + 0.6)               # size of the excursion
    width = 2.6 + 0.8 * np.tanh(x)
    bump = amp * np.exp(-0.5 * ((t - peak) / width) ** 2)
    dip = -0.6 * (x < -0.6) * np.exp(-0.5 * ((t - 7) / 1.5) ** 2) * (1 - x)
    return 22 + bump + dip + 0.12 * np.sin(t / 1.7 + 3 * x)

T = np.linspace(0, 24, 200)

def diversity(idx):
    P = pool[idx]
    d = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1))
    return d[np.triu_indices(len(idx), 1)].mean()

# CUCB-like sequence of candidate super arms: early rounds sample near the
# real data, later rounds explore the augmented space; keep the best so far.
rounds = []
best = rng.choice(len(real), K, replace=False)
best_val = diversity(best)
for r in range(11):
    frac = r / 10
    if r == 10:                               # final: farthest-point spread
        cand = [int(np.argmin(pool[:, 0]))]
        while len(cand) < K:
            d = np.min(np.sqrt(((pool[:, None] - pool[cand][None]) ** 2).sum(-1)), 1)
            cand.append(int(np.argmax(d)))
        cand = np.array(cand)
    else:
        w = np.exp(frac * 2.2 * np.abs(pool[:, 0]))
        while True:  # keep scenarios in a set visibly distinct
            cand = rng.choice(len(pool), K, replace=False, p=w / w.sum())
            P = pool[cand]
            dmin = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1))[np.triu_indices(K, 1)].min()
            if dmin > 0.3:
                break
    val = diversity(cand)
    improved = val > best_val
    if improved:
        best, best_val = cand, val
    rounds.append((cand, best.copy(), improved))

# ---------- timeline (seconds) ----------------------------------------------
T_REAL, T_DIFF0, T_DIFF1, T_BANDIT = 0.0, 1.2, 4.2, 4.6
ROUND = 0.75
T_END = T_BANDIT + ROUND * len(rounds) + 2.2
nframes = int(T_END * FPS)

COLORS = ["#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#1f77b4"]
GREY, BLUE, RING = "#4a4a4a", "#9ecae1", "#e6550d"

def ease(u):
    u = np.clip(u, 0, 1)
    return u * u * (3 - 2 * u)

fig = plt.figure(figsize=(10, 4.6), dpi=120)
axL = fig.add_axes([0.02, 0.06, 0.42, 0.88])
axR = fig.add_axes([0.52, 0.12, 0.46, 0.78])

prev_best = rounds[0][1]
for f in range(nframes):
    t = f / FPS
    axL.cla(); axR.cla()
    axL.set_xlim(-2.0, 2.0); axL.set_ylim(-1.5, 1.6)
    axL.set_xticks([]); axL.set_yticks([])
    for s in axL.spines.values(): s.set_color("#cccccc")

    # real data fades in
    a_real = ease((t - T_REAL) / 0.8)
    # diffusion: noise -> samples
    u = ease((t - T_DIFF0) / (T_DIFF1 - T_DIFF0))
    if t >= T_DIFF0:
        noise_scale = (1 - u) * 0.35
        pos = (1 - u) * syn_start + u * syn_target + rng.normal(0, noise_scale, syn_target.shape) * (u < 1)
        axL.scatter(pos[:, 0], pos[:, 1], s=14, color=BLUE, alpha=0.35 + 0.5 * u, lw=0, zorder=1)
    axL.scatter(real[:, 0], real[:, 1], s=20, color=GREY, alpha=0.9 * a_real, lw=0, zorder=2)

    # bandit rounds
    k = int((t - T_BANDIT) // ROUND) if t >= T_BANDIT else -1
    k = min(k, len(rounds) - 1)
    if k >= 0:
        cand, best_k, improved = rounds[k]
        prev = rounds[k - 1][1] if k > 0 else rounds[0][1] if not rounds[0][2] else best_k
        phase = ((t - T_BANDIT) % ROUND) / ROUND if (t - T_BANDIT) < ROUND * len(rounds) else 1.0
        # candidate rings during the first part of the round
        if phase < 0.6:
            axL.scatter(pool[cand, 0], pool[cand, 1], s=150, facecolors="none",
                        edgecolors=RING, lw=2.0, alpha=1 - phase / 0.6 * 0.4, zorder=4)
        show = best_k if phase >= 0.45 else prev
        m = ease((phase - 0.45) / 0.35) if improved else 1.0
        cur_best = best_k
    else:
        show, m, cur_best = None, 1.0, None

    # selected set + trajectories
    axR.set_xlim(0, 24); axR.set_ylim(20.6, 26.2)
    axR.set_xticks([]); axR.set_yticks([])
    for side in ("top", "right"): axR.spines[side].set_visible(False)
    for side in ("left", "bottom"): axR.spines[side].set_color("#888888")
    axR.axhspan(21.0, 24.5, color="#e5f5e0", zorder=0)
    if k >= 0:
        prev_set = rounds[k - 1][1] if k > 0 else rounds[0][1]
        for j in range(K):
            if rounds[k][2]:
                p0, p1 = pool[prev_set[j]], pool[rounds[k][1][j]]
                z = (1 - m) * p0 + m * p1
            else:
                z = pool[rounds[k][1][j]]
            axL.scatter([z[0]], [z[1]], s=70, color=COLORS[j], edgecolors="white", lw=1.2, zorder=5)
            axR.plot(T, traj(z, T), color=COLORS[j], lw=2.4, zorder=3)
    elif t >= 0.3:
        # before the bandit starts: what handpicked real scenarios look like
        a = ease((t - 0.3) / 0.8)
        for j, i in enumerate(range(K)):
            axR.plot(T, traj(real[i], T), color=GREY, lw=1.6, alpha=0.6 * a, zorder=2)

    fig.savefig(OUT / f"f{f:04d}.png", facecolor="white")
print("frames", nframes, "seconds", T_END)
