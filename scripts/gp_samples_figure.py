"""Side-by-side 3D surfaces of 2D GP prior samples.

Left: anisotropic squared-exponential (infinitely smooth), lengthscales
(0.55, 0.1). Right: isotropic Matern-5/2 (twice differentiable), shared
lengthscale 0.2. Both are drawn with random Fourier features so the grid can
be fine without an O(n^3) Cholesky.
Usage: python gp_samples_figure.py figures/Iso-BO.png
"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, LinearSegmentedColormap

M = 4000          # random features
NU = 2.5          # Matern smoothness
rng = np.random.default_rng(12)

g = np.linspace(0, 1, 220)
GX, GY = np.meshgrid(g, g)
X = np.c_[GX.ravel(), GY.ravel()]


def rff_sample(omega):
    """f(x) = sqrt(2/M) sum_m w_m cos(omega_m . x + b_m), w ~ N(0, 1)."""
    b = rng.uniform(0, 2 * np.pi, M)
    w = rng.normal(0, 1, M)
    return (np.sqrt(2 / M) * np.cos(X @ omega.T + b) @ w).reshape(GX.shape)


# SE spectral density: Gaussian with covariance diag(1/ls^2)
ls_aniso = np.array([0.55, 0.1])
omega_se = rng.normal(0, 1, (M, 2)) / ls_aniso

# Matern-nu spectral density: multivariate Student-t with 2*nu dof, scale 1/ls
ls_iso = 0.2
chi = rng.chisquare(2 * NU, M)
omega_mat = rng.normal(0, 1, (M, 2)) / ls_iso * np.sqrt(2 * NU / chi)[:, None]

F_left = rff_sample(omega_se)
F_right = rff_sample(omega_mat)

# single light-blue hue, shaded by a light source (no axes, no edges)
BLUE = LinearSegmentedColormap.from_list("sky", ["#3f8fc4", "#a9dcef"])
light = LightSource(azdeg=300, altdeg=35)

fig = plt.figure(figsize=(12, 5.4), dpi=150)
for k, F in enumerate((F_left, F_right)):
    ax = fig.add_axes([0.0 + 0.5 * k, 0.0, 0.5, 1.0], projection="3d")
    F = (F - F.mean()) / F.std()
    rgb = light.shade(F, cmap=BLUE, vert_exag=0.08, blend_mode="soft")
    ax.plot_surface(GX, GY, F, facecolors=rgb, rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=False)
    ax.set_box_aspect((1, 1, 0.8), zoom=1.12)
    ax.view_init(elev=26, azim=(-28 if k == 0 else -58))
    ax.set_axis_off()
fig.savefig(sys.argv[1], facecolor="white", bbox_inches="tight", pad_inches=0.05)
print("saved", sys.argv[1])
