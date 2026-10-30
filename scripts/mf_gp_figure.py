"""1D multi-fidelity GP figure for the CAGES paper.

Many cheap low-fidelity (LF, red) samples and a few expensive high-fidelity
(HF, blue) samples of the same function. A GP fit to HF data alone (blue) is
poor; a two-level autoregressive multi-fidelity GP, f_H = rho * f_L + delta
(Kennedy & O'Hagan), tracks the exact function (black dotted) closely (green).
Usage: python mf_gp_figure.py figures/CAGES.png
"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RED, BLUE, GREEN = "#e41a1c", "#1f3fd6", "#2e8b2e"


def f_high(x):
    return (np.sin(14 * x) * 1.1 + 0.8 * np.sin(31 * x + 0.5) * np.exp(-1.5 * x)
            + 1.4 * np.exp(-((x - 0.92) / 0.05) ** 2) - 0.9 * np.exp(-((x - 0.1) / 0.06) ** 2))


def f_low(x):
    return 0.6 * f_high(x) + 1.2 + 0.8 * np.sin(4 * x) + 0.3 * x


def rbf(a, b, ls, var):
    return var * np.exp(-0.5 * (a[:, None] - b[None]) ** 2 / ls ** 2)


def fit_gp(X, y, noise=1e-4):
    """Constant-mean GP; lengthscale and variance chosen by marginal likelihood."""
    m = y.mean(); yc = y - m
    best = None
    for ls in np.geomspace(0.01, 0.5, 60):
        for var in np.geomspace(0.05, 10, 30):
            K = rbf(X, X, ls, var) + noise * np.eye(len(X))
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                continue
            a = np.linalg.solve(L.T, np.linalg.solve(L, yc))
            nll = 0.5 * yc @ a + np.log(np.diag(L)).sum()
            if best is None or nll < best[0]:
                best = (nll, ls, var)
    _, ls, var = best
    K = rbf(X, X, ls, var) + noise * np.eye(len(X))
    Kinv_y = np.linalg.solve(K, yc)

    def predict(xs):
        Ks = rbf(xs, X, ls, var)
        mu = m + Ks @ Kinv_y
        v = var - np.einsum("ij,ji->i", Ks, np.linalg.solve(K, Ks.T))
        return mu, np.maximum(v, 1e-12)
    return predict


xs = np.linspace(0, 1, 600)
X_L = np.linspace(0, 1, 26)
X_H = np.array([0.0, 0.17, 0.36, 0.5, 0.63, 0.8, 1.0])
y_L, y_H = f_low(X_L), f_high(X_H)

gp_L = fit_gp(X_L, y_L)
gp_H = fit_gp(X_H, y_H)

mu_L_at_H, _ = gp_L(X_H)
rho = float(np.dot(mu_L_at_H - mu_L_at_H.mean(), y_H - y_H.mean())
            / np.dot(mu_L_at_H - mu_L_at_H.mean(), mu_L_at_H - mu_L_at_H.mean()))
gp_D = fit_gp(X_H, y_H - rho * mu_L_at_H)

mL, vL = gp_L(xs)
mH, vH = gp_H(xs)
mD, vD = gp_D(xs)
mMF, vMF = rho * mL + mD, rho ** 2 * vL + vD

fig, ax = plt.subplots(figsize=(8, 4), dpi=200)
ax.fill_between(xs, mH - 2 * np.sqrt(vH), mH + 2 * np.sqrt(vH), color=BLUE, alpha=0.14, lw=0)
ax.fill_between(xs, mMF - 2 * np.sqrt(vMF), mMF + 2 * np.sqrt(vMF), color=GREEN, alpha=0.22, lw=0)
ax.plot(xs, mL, color=RED, lw=2)
ax.plot(xs, mH, color=BLUE, lw=2)
ax.plot(xs, mMF, color=GREEN, lw=2.2)
ax.plot(xs, f_high(xs), color="black", lw=1.8, ls=(0, (1, 1.5)))
ax.scatter(X_L, y_L, marker="*", s=70, color=RED, zorder=5)
ax.scatter(X_H, y_H, marker="*", s=170, color=BLUE, edgecolors="white", lw=0.6, zorder=6)
ax.set_xlim(-0.02, 1.02)
ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout(pad=0.3)
fig.savefig(sys.argv[1], facecolor="white")
print("saved", sys.argv[1], "rho", round(rho, 3))
