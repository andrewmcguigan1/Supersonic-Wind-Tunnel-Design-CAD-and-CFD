"""Animation of the normal shock walking down the diffuser as stagnation pressure P0 rises.
Each frame is a converged steady 2D Euler solution (quasi-steady sweep, not a time-accurate run)."""
import os, sys, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from tunnel_geometry import CASE, build_contour, quasi1d_shock_location
OUT = os.path.join(ROOT, "output", "flow"); os.makedirs(OUT, exist_ok=True)
xs, ys, st = build_contour(); h = CASE["h_test"]
BG = "#0b0f17"; FG = "#e6e9ef"; MUTED = "#8a93a6"; ACC = "#ff6b3d"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": FG,
                     "axes.labelcolor": FG, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": "#2a3142",
                     "font.size": 12})
files = sorted(glob.glob(os.path.join(ROOT, "output", "sweep", "P0_*.npz")))
sols = [dict(np.load(f)) for f in files]
# below ~28 psia the shock sits within ~3h of the diffuser entry and keeps oscillating (+-0.4h) in this
# inviscid solver, so those states are not steady; only steady frames are animated
sols = [d for d in sols if float(d["P0_psia"]) >= 28.0]
def shock_x(d):
    M = d["M"][:, 0]; x = d["xc"][:, 0]
    k = np.where((M[:-1] > 1) & (M[1:] <= 1) & (x[:-1] > st["x_test_end"]))[0]
    if len(k) == 0: return np.nan
    k = k[0]; return x[k] + (1 - M[k]) * (x[k + 1] - x[k]) / (M[k + 1] - M[k])
P0s = np.array([float(d["P0_psia"]) for d in sols])
xsh = np.array([(shock_x(d) - st["x_test_end"]) / h for d in sols])
Pq = np.linspace(P0s.min(), P0s.max(), 60)
xq = []
for p in Pq:
    r = quasi1d_shock_location(P0=p); xq.append((r[0] - st["x_test_end"]) / h if r else np.nan)

fig = plt.figure(figsize=(16, 7.2))
ax = fig.add_axes([0.02, 0.36, 0.90, 0.58]); cax = fig.add_axes([0.925, 0.42, 0.012, 0.46])
ax2 = fig.add_axes([0.09, 0.08, 0.84, 0.24])
X, Y = sols[0]["X"], sols[0]["Y"]
Xn = np.concatenate([X[:, ::-1], X[:, 1:]], 1); Yn = np.concatenate([-Y[:, ::-1], Y[:, 1:]], 1)
mir = lambda a: np.concatenate([a[:, ::-1], a], 1)
pc = ax.pcolormesh(Xn, Yn, mir(sols[0]["M"]), cmap="turbo", vmin=0, vmax=2.6, shading="flat", rasterized=True)
ax.plot(xs, ys, color="#cfd6e4", lw=1.4); ax.plot(xs, -ys, color="#cfd6e4", lw=1.4)
ax.set_aspect("equal"); ax.set_xlim(-12, st["x_diff_exit"] + 0.5); ax.axis("off")
cb = fig.colorbar(pc, cax=cax); cb.set_label("Mach"); cb.outline.set_edgecolor("#2a3142")
title = ax.set_title("", color=FG, fontsize=16, pad=14)
vl = ax.axvline(st["x_shock_design"], color="w", ls=":", lw=1.2)
ax.text(st["x_shock_design"], -7.6, "design station 5h", ha="center", va="top", color=FG, fontsize=10)
ax2.plot(Pq, xq, "--", color=FG, lw=1.2, label="quasi-1D theory")
ax2.plot(P0s, xsh, "o", color=ACC, ms=5, alpha=0.35)
dot, = ax2.plot([], [], "o", color=ACC, ms=11, label="2D CFD")
ax2.axhline(5, color=MUTED, lw=0.8, ls=":")
ax2.set_xlabel("stagnation pressure P0 (psia)"); ax2.set_ylabel("shock distance\ninto diffuser (h)")
ax2.grid(alpha=0.12); ax2.legend(frameon=False, loc="upper left")
fig.text(0.5, 0.965, "Each frame: converged 2D inviscid (Euler) solution, in-house solver. Quasi-steady P0 sweep, not a time-accurate run.",
         ha="center", va="top", fontsize=10, color=MUTED)
def frame(i):
    pc.set_array(mir(sols[i]["M"]).ravel())
    title.set_text(f"Normal shock in the diffuser vs stagnation pressure:  P0 = {P0s[i]:.1f} psia,  shock at {xsh[i]:.2f} h")
    dot.set_data([P0s[i]], [xsh[i]])
    return pc, title, dot
seq = list(range(len(sols))) + list(range(len(sols) - 2, 0, -1))
anim = FuncAnimation(fig, frame, frames=seq, blit=False)
anim.save(os.path.join(OUT, "05_shock_sweep.mp4"), writer=FFMpegWriter(fps=4, bitrate=6000), dpi=150)
anim.save(os.path.join(OUT, "05_shock_sweep.gif"), writer=PillowWriter(fps=4), dpi=80)
frame(int(np.argmin(abs(P0s - 31.43)))); fig.savefig(os.path.join(OUT, "05_shock_sweep_still.png"), dpi=170)
print("P0", P0s, "\nshock/h", np.round(xsh, 2))
