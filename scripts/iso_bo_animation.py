"""Schematic animation for 'Isotropic Gaussian Processes Improve Vanilla
Bayesian Optimization in High Dimensions' (Iso-BO).

Same objective and initial design in two panels. Left: vanilla BO with ARD
lengthscales, whose poorly identified coordinates are filled in by the prior,
so the correlated region around the incumbent is an elongated, axis-aligned
ellipse that changes shape between refits and queries spread out. Middle:
Iso-BO with one shared lengthscale, a Euclidean ball that tightens around
the incumbent. Right: best-so-far regret for both. Illustrative only, not
data from the paper.
Usage: python iso_bo_animation.py OUT_DIR
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
FPS = 20
N_ITER = 20
ARD_C, ISO_C, OPT_C = "#c44e52", "#1f77b4", "#2ca02c"

OPT = np.array([0.68, 0.32])


def f(x, y):
    """Smooth multimodal objective with its global minimum at OPT."""
    r2 = (x - OPT[0]) ** 2 + (y - OPT[1]) ** 2
    return (1.0 - np.exp(-r2 / 0.12)
            - 0.25 * np.exp(-((x - 0.22) ** 2 + (y - 0.75) ** 2) / 0.02)
            - 0.2 * np.exp(-((x - 0.2) ** 2 + (y - 0.2) ** 2) / 0.015)
            + 0.15 * np.sin(6 * x) * np.cos(5 * y) + 0.2)


g = np.linspace(0, 1, 160)
GX, GY = np.meshgrid(g, g)
GZ = f(GX, GY)
F_MIN = min(GZ.min(), f(*OPT)) - 1e-3  # small offset keeps log-regret finite

rng0 = np.random.default_rng(3)
X0 = rng0.uniform(0.05, 0.95, (40, 2))
X0 = X0[np.linalg.norm(X0 - OPT, axis=1) > 0.38][:10]  # start away from the optimum
X0[0] = OPT + [-0.24, 0.2]  # best initial point sits on the slope of the global basin


def run(kind, seed):
    rng = np.random.default_rng(seed)
    X = [p.copy() for p in X0]
    regions = []
    for t in range(N_ITER):
        Y = np.array([f(*p) for p in X])
        best = X[int(np.argmin(Y))]
        if kind == "iso":
            r = 0.32 * 0.86 ** t + 0.03
            ls = (r, r)
            q = best + rng.normal(0, r / 2.2, 2)
        else:
            # one coordinate weakly identified -> prior fills in a long scale,
            # and which one flips between refits
            long_axis = rng.integers(2)
            ls = [0.06 + 0.04 * 0.9 ** t] * 2
            ls[long_axis] = 0.55
            q = best + rng.normal(0, 1, 2) * np.array(ls) / 1.6
        q = np.clip(q, 0.01, 0.99)
        regions.append((best.copy(), tuple(ls)))
        X.append(q)
    Y = np.array([f(*p) for p in X])
    regret = np.minimum.accumulate(Y) - F_MIN
    return np.array(X), regions, regret


def median_seed(kind, n=60):
    """Seed whose final regret is the median over n runs (a typical run)."""
    final = np.array([run(kind, s)[2][-1] for s in range(n)])
    return int(np.argsort(final)[n // 2])


ard_X, ard_R, ard_reg = run("ard", median_seed("ard"))
iso_X, iso_R, iso_reg = run("iso", median_seed("iso"))

T_INIT, STEP, HOLD = 1.2, 0.45, 2.2
T_END = T_INIT + STEP * N_ITER + HOLD
nframes = int(T_END * FPS)


def ease(u):
    u = np.clip(u, 0, 1)
    return u * u * (3 - 2 * u)


fig = plt.figure(figsize=(12, 4.2), dpi=110)
axA = fig.add_axes([0.01, 0.04, 0.31, 0.9])
axI = fig.add_axes([0.34, 0.04, 0.31, 0.9])
axC = fig.add_axes([0.70, 0.12, 0.28, 0.76])
reg_lo = max(min(ard_reg.min(), iso_reg.min()) * 0.5, 1e-4)
reg_hi = max(ard_reg.max(), iso_reg.max()) * 1.6

for fr in range(nframes):
    t = fr / FPS
    k = int(np.clip((t - T_INIT) // STEP, -1, N_ITER - 1)) if t >= T_INIT else -1
    phase = ((t - T_INIT) % STEP) / STEP if 0 <= t - T_INIT < STEP * N_ITER else 1.0
    a0 = ease(t / 0.8)

    for ax, X, R, col in ((axA, ard_X, ard_R, ARD_C), (axI, iso_X, iso_R, ISO_C)):
        ax.cla()
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_color("#cccccc")
        ax.contour(GX, GY, GZ, levels=18, cmap="Greens_r", linewidths=0.8, alpha=0.55, zorder=0)
        ax.scatter(*OPT, marker="*", s=260, color=OPT_C, edgecolors="white", lw=1, zorder=2)
        ax.scatter(X0[:, 0], X0[:, 1], s=26, color="black", alpha=a0, zorder=3)
        if k >= 0:
            prev = X[10:10 + k]
            if len(prev):
                ax.scatter(prev[:, 0], prev[:, 1], s=30, color=col, alpha=0.55, lw=0, zorder=3)
            center, (lx, ly) = R[k]
            grow = ease(phase / 0.4)
            ax.add_patch(Ellipse(center, 2 * lx * grow, 2 * ly * grow, facecolor=col,
                                 alpha=0.13, edgecolor=col, lw=1.6, zorder=1))
            if phase >= 0.45:
                q = X[10 + k]
                ax.scatter(*q, s=110, facecolors="none", edgecolors=col, lw=2.2, zorder=5)
                ax.scatter(*q, s=34, color=col, zorder=5)
            Ys = np.array([f(*p) for p in X[:10 + k + (phase >= 0.45)]])
            inc = X[int(np.argmin(Ys))]
        else:
            inc = X0[np.argmin(f(X0[:, 0], X0[:, 1]))]
        ax.scatter(*inc, marker="s", s=60, color=col if k >= 0 else "black",
                   edgecolors="white", lw=1, alpha=a0, zorder=6)

    axC.cla()
    axC.set_yscale("log"); axC.set_xlim(0, N_ITER); axC.set_ylim(reg_lo, reg_hi)
    axC.set_xticks([]); axC.set_yticks([]); axC.minorticks_off()
    for side in ("top", "right"): axC.spines[side].set_visible(False)
    for side in ("left", "bottom"): axC.spines[side].set_color("#888888")
    if k >= 0:
        n = k + 1 + (phase >= 0.45)
        xs = np.arange(n)
        axC.step(xs, ard_reg[9:9 + n], where="post", color=ARD_C, lw=2.6)
        axC.step(xs, iso_reg[9:9 + n], where="post", color=ISO_C, lw=2.6)
    fig.savefig(OUT / f"f{fr:04d}.png", facecolor="white")

print("frames", nframes, "seconds", round(T_END, 2),
      "final regret ard", ard_reg[-1], "iso", iso_reg[-1])
