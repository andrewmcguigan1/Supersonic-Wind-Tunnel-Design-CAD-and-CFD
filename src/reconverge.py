"""Extra iterations for sweep frames whose shock was still drifting."""
import os, sys, glob, numpy as np
from euler2d import Solver, prim_to_cons
from tunnel_geometry import P_ATM_PSIA, build_contour
xs, ys, st = build_contour()
HERE = os.path.dirname(os.path.abspath(__file__))
def shx(f, xc):
    M = f["M"][:, 0]; x = xc[:, 0]
    k = np.where((M[:-1] > 1) & (M[1:] <= 1) & (x[:-1] > st["x_test_end"]))[0][0]; return (x[k] - st["x_test_end"]) / 6
for fn in sorted(glob.glob(os.path.join(os.path.dirname(HERE), "output", "sweep", "P0_*.npz"))):
    d = dict(np.load(fn)); P0 = float(d["P0_psia"])
    if P0 > 30.6: continue
    s = Solver(d["X"], d["Y"], 1.0, P_ATM_PSIA / P0)
    for q, k in enumerate(["rho", "u", "v", "p"]): s.W[q, 2:-2, 2:-2] = d[k]
    s.U = prim_to_cons(s.W)
    prev = shx(s.fields(), s.xc)
    for blk in range(8):
        s.run(4000, cfl=0.8, order=2, report=0); now = shx(s.fields(), s.xc)
        print(P0, blk, round(prev, 3), "->", round(now, 3), flush=True)
        if abs(now - prev) < 0.01: break
        prev = now
    f = s.fields()
    np.savez_compressed(fn, X=s.X, Y=s.Y, xc=s.xc, yc=s.yc, P0_psia=P0, **f)
