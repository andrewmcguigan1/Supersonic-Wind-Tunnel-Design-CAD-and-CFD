"""Presentation-quality flow visuals from the 2D Euler solution (dark theme).
All fields come from the in-house 2D planar inviscid (Euler) solver, empty tunnel."""
import os, sys, glob, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.interpolate import griddata
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from tunnel_geometry import (CASE, build_contour, moc_min_length_nozzle, quasi1d_shock_location,
                             P_ATM_PSIA, M_from_A, shock_p02_p01, p0_p)
OUT = os.path.join(ROOT, "output", "flow"); os.makedirs(OUT, exist_ok=True)
SOL = os.path.join(ROOT, "output", "cfd", "solution_shock5h.npz")
xs, ys, st = build_contour(); h = CASE["h_test"]; At = st["h_throat"]

BG = "#0b0f17"; FG = "#e6e9ef"; MUTED = "#8a93a6"; ACC = "#ff6b3d"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
                     "text.color": FG, "axes.labelcolor": FG, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.edgecolor": "#2a3142", "font.size": 12, "font.family": "DejaVu Sans"})
TAG = "2D inviscid (Euler) CFD · in-house finite-volume solver · 54k cells"

def mirror(d):
    X, Y = d["X"], d["Y"]
    Xn = np.concatenate([X[:, ::-1], X[:, 1:]], axis=1); Yn = np.concatenate([-Y[:, ::-1], Y[:, 1:]], axis=1)
    f = lambda a, s=1: np.concatenate([s * a[:, ::-1], a], axis=1)
    return Xn, Yn, f

def walls(ax, lw=1.6, x0=None):
    m = xs >= (x0 if x0 is not None else -1e9)
    ax.plot(xs[m], ys[m], color="#cfd6e4", lw=lw); ax.plot(xs[m], -ys[m], color="#cfd6e4", lw=lw)

def stamp(fig, text=TAG):
    fig.text(0.995, 0.01, text, ha="right", va="bottom", fontsize=9, color=MUTED)

d = dict(np.load(SOL)); P0 = float(d["P0_psia"])
Xn, Yn, F = mirror(d)

# ---------------------------------------------------------------- 1. schlieren
r = d["rho"]; xc, yc = d["xc"], d["yc"]
gx = np.gradient(r, axis=0) / np.gradient(xc, axis=0); gy = np.gradient(r, axis=1) / np.gradient(yc, axis=1)
gr = np.sqrt(gx**2 + gy**2)
sch = np.exp(-12 * gr / np.percentile(gr, 99.7))
fig, ax = plt.subplots(figsize=(18, 4.6))
ax.pcolormesh(Xn, Yn, F(sch), cmap="bone", shading="flat", vmin=0.15, vmax=1.0, rasterized=True)
walls(ax, x0=-2)
for xx, lab in [(0, "throat"), (st["x_nozzle_exit"], "test section"), (st["x_test_end"], "diffuser"),
                (st["x_shock_design"], "normal shock (5h)")]:
    ax.annotate(lab, (xx, float(np.interp(xx, xs, ys)) + 0.3), xytext=(xx, 7.8),
                ha="center", color=FG, fontsize=11, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.set_aspect("equal"); ax.set_xlim(-2, st["x_diff_exit"] + 0.5); ax.set_ylim(-7, 9.2); ax.axis("off")
ax.set_title("Numerical schlieren  |∇ρ|  —  Mach 2 blowdown tunnel, P0 = %.1f psia" % P0, color=FG, fontsize=15, pad=4)
stamp(fig); fig.tight_layout(); fig.savefig(os.path.join(OUT, "01_numerical_schlieren.png"), dpi=200); plt.close(fig)

# ---------------------------------------------------------------- 2. Mach + streamlines
fig, ax = plt.subplots(figsize=(18, 4.8))
pc = ax.pcolormesh(Xn, Yn, F(d["M"]), cmap="turbo", shading="flat", vmin=0, vmax=2.6, rasterized=True)
ax.contour(np.concatenate([xc[:, ::-1], xc], 1), np.concatenate([-yc[:, ::-1], yc], 1), F(d["M"]),
           levels=np.arange(0.2, 2.6, 0.1), colors="k", linewidths=0.25, alpha=0.35)
ax.contour(np.concatenate([xc[:, ::-1], xc], 1), np.concatenate([-yc[:, ::-1], yc], 1), F(d["M"]),
           levels=[1.0], colors="white", linewidths=1.4)
# streamlines on a regular grid
gxv = np.linspace(-11.5, st["x_diff_exit"] - 0.3, 900); gyv = np.linspace(-7.0, 7.0, 180)
GX, GY = np.meshgrid(gxv, gyv)
pts = np.c_[np.concatenate([xc, xc], 1).ravel(), np.concatenate([yc, -yc[:, ::-1]], 1).ravel()]
U = griddata(pts, np.concatenate([d["u"], d["u"][:, ::-1]], 1).ravel(), (GX, GY), method="linear")
V = griddata(pts, np.concatenate([d["v"], -d["v"][:, ::-1]], 1).ravel(), (GX, GY), method="linear")
inside = np.abs(GY) < np.interp(GX, xs, ys) - 0.02
U = np.where(inside, U, np.nan); V = np.where(inside, V, np.nan)
seeds = np.c_[np.full(15, -11.0), np.linspace(-6.5, 6.5, 15)]
ax.streamplot(gxv, gyv, U, V, start_points=seeds, color=(1, 1, 1, 0.75), linewidth=0.8, arrowsize=0.8,
              density=10, integration_direction="forward", broken_streamlines=False)
walls(ax)
cb = plt.colorbar(pc, ax=ax, fraction=0.02, pad=0.01); cb.set_label("Mach number"); cb.outline.set_edgecolor("#2a3142")
ax.set_aspect("equal"); ax.set_xlim(-12, st["x_diff_exit"] + 0.5); ax.axis("off")
ax.set_title("Mach number with streamlines and sonic line (white)", color=FG, fontsize=15, pad=4)
stamp(fig); fig.tight_layout(); fig.savefig(os.path.join(OUT, "02_mach_streamlines.png"), dpi=200); plt.close(fig)

# ---------------------------------------------------------------- 3. MOC net over CFD (nozzle)
N = 14
wall, P = moc_min_length_nozzle(CASE["M"], At / 2, N)
fig, ax = plt.subplots(figsize=(14, 6.2))
m = xc[:, 0] < st["x_nozzle_exit"] + 3
pc = ax.pcolormesh(Xn[:m.sum() + 1], Yn[:m.sum() + 1], F(d["M"])[:m.sum()], cmap="turbo", shading="flat",
                   vmin=0.5, vmax=2.3, rasterized=True)
for sgn in (1, -1):
    for i in range(N):
        pts_ = [(0, At / 2)] + [P[(i, j)][:2] for j in range(i + 1)] + [P[(k, i)][:2] for k in range(i + 1, N)] + [tuple(wall[i + 1, :2])]
        pts_ = np.array(pts_); ax.plot(pts_[:, 0], sgn * pts_[:, 1], color="white", lw=0.7, alpha=0.85)
walls(ax)
ax.set_xlim(-3, st["x_nozzle_exit"] + 3); ax.set_ylim(-3.4, 3.4); ax.set_aspect("equal")
ax.set_xlabel("x from throat (in)"); ax.set_ylabel("y (in)")
cb = plt.colorbar(pc, ax=ax, fraction=0.03, pad=0.01); cb.set_label("Mach (CFD)"); cb.outline.set_edgecolor("#2a3142")
ax.set_title("Method-of-characteristics design (white net) over the CFD Mach field", color=FG, fontsize=14)
ax.text(0.2, -3.15, "Minimum-length nozzle, θmax = ν(2)/2 = 13.19°.  CFD test-section Mach: 1.998 mean (1.94–2.02)",
        color=FG, fontsize=10)
stamp(fig); fig.tight_layout(); fig.savefig(os.path.join(OUT, "03_moc_net_over_cfd.png"), dpi=200); plt.close(fig)

# ---------------------------------------------------------------- 4. pressure field + wall distribution
fig, axs = plt.subplots(2, 1, figsize=(16, 8.5), gridspec_kw=dict(height_ratios=[1, 1.15]), sharex=True)
pc = axs[0].pcolormesh(Xn, Yn, F(d["p"] * P0), cmap="magma", shading="flat", rasterized=True)
walls(axs[0]); axs[0].set_aspect("equal"); axs[0].axis("off")
cb = plt.colorbar(pc, ax=axs[0], fraction=0.02, pad=0.01); cb.set_label("static p (psia)"); cb.outline.set_edgecolor("#2a3142")
axs[0].set_title("Static pressure field and wall pressure distribution", color=FG, fontsize=15)
xq = np.linspace(xc[0, 0], xc[-1, 0], 700)
xsh, M1 = quasi1d_shock_location(P0=P0); r02 = shock_p02_p01(M1)
pq = []
for xx in xq:
    a = 2 * np.interp(xx, xs, ys) / At
    if xx < 0: Mq = M_from_A(a, False); pq.append(1 / p0_p(Mq))
    elif xx < xsh: Mq = M_from_A(a, True); pq.append(1 / p0_p(Mq))
    else: Mq = M_from_A(a * r02, False); pq.append(r02 / p0_p(Mq))
ax = axs[1]
ax.plot(xc[:, -1], d["p"][:, -1] * P0, color=ACC, lw=2, label="CFD, wall")
ax.plot(xc[:, 0], d["p"][:, 0] * P0, color="#4cc9f0", lw=1.4, label="CFD, centreline")
ax.plot(xq, np.array(pq) * P0, "--", color=FG, lw=1.3, label="quasi-1D theory")
ax.axhline(P_ATM_PSIA, color=MUTED, lw=0.8, ls=":"); ax.text(xq[3], P_ATM_PSIA + 0.5, "atmosphere", color=MUTED, fontsize=10)
ax.set_ylabel("static pressure (psia)"); ax.set_xlabel("x from throat (in)"); ax.grid(alpha=0.12)
ax.legend(frameon=False, loc="upper right")
stamp(fig); fig.tight_layout(); fig.savefig(os.path.join(OUT, "04_pressure_field_and_wall.png"), dpi=200); plt.close(fig)
print("static figures done")
