"""Plots and comparison with the hand / quasi-1D analysis.  usage: python post_cfd.py TAG [TAG2 ...]"""
import sys, os, json, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from tunnel_geometry import (CASE, build_contour, A_Astar, M_from_A, shock_p02_p01, p0_p,
                             quasi1d_shock_location, moc_min_length_nozzle, P_ATM_PSIA, nu)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFD = os.path.join(ROOT, "output", "cfd"); FIG = os.path.join(ROOT, "output", "cfd_figures"); os.makedirs(FIG, exist_ok=True)
G = 1.4
xs, ys, st = build_contour()
h = CASE["h_test"]; At = st["h_throat"]
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})

def quasi1d(P0, x):
    """quasi-1D M and p/P0 along x for the shock location set by p_exit = Patm."""
    xsh, M1 = quasi1d_shock_location(P0=P0)
    A = 2 * np.interp(x, xs, ys); M = np.empty_like(x); pr = np.empty_like(x)
    r02 = shock_p02_p01(M1)
    for k, xx in enumerate(x):
        a = A[k] / At
        if xx < 0: M[k] = M_from_A(a, False); pr[k] = 1 / p0_p(M[k])
        elif xx < xsh: M[k] = M_from_A(a, True); pr[k] = 1 / p0_p(M[k])
        else: M[k] = M_from_A(a * r02, False); pr[k] = r02 / p0_p(M[k])
    return M, pr, xsh

def shock_x(xc, M, row):
    Mc = M[:, row]; x = xc[:, row]
    k = np.where((Mc[:-1] > 1) & (Mc[1:] <= 1) & (x[:-1] > st["x_test_end"]))[0]
    if len(k) == 0: return float("nan")
    k = k[0]
    return float(x[k] + (1 - Mc[k]) * (x[k + 1] - x[k]) / (Mc[k + 1] - Mc[k]))

def full(a, sign=1):  # mirror upper half to full height
    return np.concatenate([sign * a[:, ::-1], a], axis=1)

summary = {}
for TAG in sys.argv[1:]:
    d = dict(np.load(os.path.join(CFD, f"solution_{TAG}.npz")))
    P0 = float(d["P0_psia"]); xc, yc, M, p = d["xc"], d["yc"], d["M"], d["p"]
    X, Y = d["X"], d["Y"]
    # ---- Mach contour
    fig, ax = plt.subplots(figsize=(15, 4.2))
    Xn = np.concatenate([X[:, ::-1], X[:, 1:]], axis=1); Yn = np.concatenate([-Y[:, ::-1], Y[:, 1:]], axis=1)
    cf = ax.pcolormesh(Xn, Yn, full(M), cmap="turbo", shading="flat", vmin=0, vmax=2.6)
    ax.plot(xs, ys, "k", lw=1.2); ax.plot(xs, -ys, "k", lw=1.2)
    for xx, lab in [(st["x_nozzle_exit"], "test section"), (st["x_test_end"], "diffuser")]:
        ax.axvline(xx, color="w", lw=0.8, ls="--")
    xsd = st["x_shock_design"]
    ax.axvline(xsd, color="m", lw=1.5, ls=":", label=f"design shock station (5h = {5*h:.0f} in into diffuser)")
    ax.set_aspect("equal"); ax.set_xlim(st["x_conv"] - 2, st["x_diff_exit"] + 0.5)
    ax.set_xlabel("x from throat (in)"); ax.set_ylabel("y (in)"); ax.grid(False)
    ax.set_title(f"2D Euler CFD, Mach number   (6x6 in, M = 2, 3° diffuser, P0 = {P0:.2f} psia, exit p = 14.7 psia)")
    plt.colorbar(cf, ax=ax, label="Mach", fraction=0.025, pad=0.01); ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"mach_contour_{TAG}.png"), dpi=170); plt.close(fig)
    # ---- numerical schlieren
    r = d["rho"]
    gx = np.gradient(r, axis=0) / np.gradient(xc, axis=0)
    gy = np.gradient(r, axis=1) / np.gradient(yc, axis=1)
    gr = np.sqrt(gx**2 + gy**2); sch = np.exp(-15 * gr / gr.max())
    fig, ax = plt.subplots(figsize=(15, 4.0))
    ax.pcolormesh(Xn, Yn, full(sch), cmap="gray", shading="flat", vmin=0.2, vmax=1)
    ax.plot(xs, ys, "k", lw=1); ax.plot(xs, -ys, "k", lw=1)
    ax.set_aspect("equal"); ax.set_xlim(-2, st["x_diff_exit"] + 0.5); ax.grid(False)
    ax.set_title(f"Numerical schlieren (|grad rho|), P0 = {P0:.2f} psia"); ax.set_xlabel("x from throat (in)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"schlieren_{TAG}.png"), dpi=170); plt.close(fig)
    # ---- centerline / wall vs quasi-1D
    xq = np.linspace(xc[0, 0], xc[-1, 0], 800)
    Mq, pq, xsh_q = quasi1d(P0, xq)
    # area-averaged (mass-weighted would be similar) Mach per column
    fig, axs = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    axs[0].plot(xc[:, 0], M[:, 0], label="CFD centerline")
    axs[0].plot(xc[:, -1], M[:, -1], label="CFD wall")
    axs[0].plot(xq, Mq, "k--", label="quasi-1D (same geometry, exit p = Patm)")
    axs[0].set_ylabel("Mach")
    axs[1].plot(xc[:, 0], p[:, 0] * P0, label="CFD centerline")
    axs[1].plot(xc[:, -1], p[:, -1] * P0, label="CFD wall")
    axs[1].plot(xq, pq * P0, "k--", label="quasi-1D")
    axs[1].axhline(P_ATM_PSIA, color="gray", lw=0.8); axs[1].text(xq[5], P_ATM_PSIA + 0.4, "Patm", color="gray")
    axs[1].set_ylabel("static pressure (psia)"); axs[1].set_xlabel("x from throat (in)")
    for a in axs:
        for xx in (0, st["x_nozzle_exit"], st["x_test_end"]): a.axvline(xx, color="gray", lw=0.6, ls="--")
        a.axvline(st["x_shock_design"], color="m", ls=":", lw=1.3)
    axs[0].legend(fontsize=9); axs[0].set_title(f"Centerline and wall distributions, P0 = {P0:.2f} psia  (magenta = 5h design station)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"centerline_{TAG}.png"), dpi=160); plt.close(fig)
    # ---- test-section uniformity
    ts = (xc[:, 0] >= st["x_nozzle_exit"]) & (xc[:, 0] <= st["x_test_end"])
    Mts = M[ts]
    core = Mts[:, :]  # whole test section, wall to centerline
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        i = np.argmin(abs(xc[:, 0] - (st["x_nozzle_exit"] + frac * (st["x_test_end"] - st["x_nozzle_exit"]))))
        ax.plot(M[i], yc[i], label=f"x = {xc[i,0]-st['x_nozzle_exit']:.1f} in into test section")
    ax.axvline(2.0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("Mach"); ax.set_ylabel("y (in, centerline = 0, wall = 3)"); ax.set_xlim(1.85, 2.15)
    ax.set_title("Test-section Mach profiles"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"test_section_profiles_{TAG}.png"), dpi=160); plt.close(fig)
    # ---- convergence
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.semilogy(d["hist"]); ax.set_xlabel("iteration"); ax.set_ylabel("density residual (RMS)")
    ax.set_title("Convergence history (plateau = low-speed sloshing in plenum/subsonic diffuser;\nshock position fixed to within one cell)", fontsize=9); fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"convergence_{TAG}.png"), dpi=140); plt.close(fig)
    # ---- mass flow check
    # mass flow through every column of the steady supersonic region (throat to test-section end)
    cols = np.where((xc[:, 0] > 1.0) & (xc[:, 0] < st["x_test_end"]))[0]
    mdot_nd = np.mean([2 * np.sum(d["rho"][i] * d["u"][i] * np.diff(Y[i])) for i in cols])
    mdot_th = math.sqrt(G) * (2 / (G + 1)) ** ((G + 1) / (2 * (G - 1))) * At
    xsh_c = shock_x(xc, M, 0); xsh_w = shock_x(xc, M, -1)
    # mass-averaged Mach in test section
    summary[TAG] = dict(
        P0_psia=P0, iterations=len(d["hist"]), final_residual=float(d["hist"][-1]),
        test_section_M_mean=float(Mts.mean()), test_section_M_min=float(Mts.min()),
        test_section_M_max=float(Mts.max()), test_section_M_std=float(Mts.std()),
        shock_x_centerline_in=xsh_c, shock_x_wall_in=xsh_w,
        shock_into_diffuser_centerline_in=xsh_c - st["x_test_end"],
        shock_into_diffuser_over_h=(xsh_c - st["x_test_end"]) / h,
        quasi1d_shock_into_diffuser_in=xsh_q - st["x_test_end"],
        design_shock_into_diffuser_in=5 * h,
        exit_static_p_psia=float((p[-1] * np.diff(Y[-1])).sum() / Y[-1, -1] * P0),
        mdot_cfd_over_theory=float(mdot_nd / mdot_th),
        mdot_cfd_kg_s=float(mdot_nd / mdot_th * 6.3831 * P0 / 28.8274))
    print(json.dumps(summary[TAG], indent=1))

with open(os.path.join(CFD, "summary.json"), "a") as f:
    f.write(json.dumps(summary, indent=1) + "\n")

# ---- MOC nozzle characteristic net (independent of CFD)
wall, P = moc_min_length_nozzle(CASE["M"], At / 2, 12)
fig, ax = plt.subplots(figsize=(10, 4.5))
N = 12
for i in range(N):
    pts = [(0, At / 2)] + [P[(i, j)][:2] for j in range(i + 1)] + [P[(k, i)][:2] for k in range(i + 1, N)] + [tuple(wall[i + 1, :2])]
    pts = np.array(pts); ax.plot(pts[:, 0], pts[:, 1], color="tab:blue", lw=0.7)
ax.plot(wall[:, 0], wall[:, 1], "k", lw=2, label="nozzle wall (MOC, 12 characteristics shown)")
w40, _ = moc_min_length_nozzle(CASE["M"], At / 2, CASE["n_char"])
ax.plot(w40[:, 0], w40[:, 1], "r--", lw=1, label=f"design contour ({CASE['n_char']} characteristics)")
ax.set_aspect("equal"); ax.set_xlabel("x (in)"); ax.set_ylabel("y (in)")
ax.set_title(f"Minimum-length nozzle by method of characteristics, M = 2  (θ_max = ν/2 = {math.degrees(nu(2)/2):.2f}°)")
ax.legend(loc="lower right"); fig.tight_layout(); fig.savefig(os.path.join(FIG, "moc_nozzle.png"), dpi=160); plt.close(fig)
