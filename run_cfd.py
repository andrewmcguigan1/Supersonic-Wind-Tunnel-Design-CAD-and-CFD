"""Run the 2D Euler simulation of the tunnel and post-process.
usage: python run_cfd.py [P0_psia] [nx] [ny] [iters] [tag]"""
import sys, os, math, json, time
import numpy as np
from tunnel_geometry import (CASE, build_contour, A_Astar, M_from_A, shock_p02_p01,
                             p0_p, shock_M2, quasi1d_shock_location, P_ATM_PSIA)
from euler2d import make_grid, Solver

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output", "cfd"); os.makedirs(OUT, exist_ok=True)
P0 = float(sys.argv[1]) if len(sys.argv) > 1 else CASE["P0_psia"]
NX = int(sys.argv[2]) if len(sys.argv) > 2 else 900
NY = int(sys.argv[3]) if len(sys.argv) > 3 else 60
NIT = int(sys.argv[4]) if len(sys.argv) > 4 else 30000
TAG = sys.argv[5] if len(sys.argv) > 5 else f"P0_{P0:.2f}"

xs, ys, st = build_contour()
x_in = st["x_conv"] - 2.0           # start 2 in upstream of the contraction (plenum)
x_out = st["x_diff_exit"]
X, Y = make_grid(xs, ys, x_in, x_out, NX, NY,
                 x_cluster=[(st["x_throat"] - 1.5, 0.6), (st["x_nozzle_exit"] + 1, 1.6),
                            (st["x_test_end"], 1.0), (x_out, 0.9)])
pb = P_ATM_PSIA / P0
At = st["h_throat"]

# quasi-1D initial guess with a shock placed at the design station (5h into diffuser)
x_sh0 = st["x_shock_design"]
def A(x): return 2 * np.interp(x, xs, ys)
def M1d(x):
    out = np.empty_like(x)
    for k, xx in np.ndenumerate(x):
        r = A(xx) / At
        if xx < 0: out[k] = M_from_A(r, False)
        elif xx < x_sh0: out[k] = M_from_A(r, True)
        else:
            M1 = M_from_A(A(x_sh0) / At, True)
            out[k] = M_from_A(r * shock_p02_p01(M1), False)
    return out
def p1d(x):
    M = M1d(x); M1 = M_from_A(A(x_sh0) / At, True)
    p0r = np.where(x < x_sh0, 1.0, shock_p02_p01(M1))
    return p0r / (1 + 0.2 * M**2) ** 3.5

s = Solver(X, Y, P0=1.0, pb=pb)
s.init_from_1d(M1d, p1d)
t0 = time.time()
s.run(300, cfl=0.5, order=1, report=100)
hist = s.run(NIT, cfl=0.8, order=2, report=2000, tol=1e-8)
print(f"solver time {time.time()-t0:.0f} s, iterations {len(hist)}")
f = s.fields()
np.savez_compressed(os.path.join(OUT, f"solution_{TAG}.npz"), X=X, Y=Y, xc=s.xc, yc=s.yc,
                    P0_psia=P0, pb=pb, hist=np.array(hist), **f)
print("saved", TAG)
