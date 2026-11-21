"""Quasi-steady P0 sweep: restart from the previous converged field, raise P0, re-converge,
save the field. Each frame is a converged steady state (not a time-accurate transient)."""
import os, sys, numpy as np
from euler2d import Solver, prim_to_cons
from tunnel_geometry import P_ATM_PSIA
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(ROOT, "output", "sweep"); os.makedirs(OUT, exist_ok=True)
d = np.load(os.path.join(ROOT, "output", "cfd", "solution_design.npz"))   # from: python src/run_cfd.py 28.83 900 60 40000 design
P0s = np.round(np.linspace(26.5, 34.0, 16), 2)
s = Solver(d["X"], d["Y"], 1.0, P_ATM_PSIA / P0s[0])
for q, k in enumerate(["rho", "u", "v", "p"]): s.W[q, 2:-2, 2:-2] = d[k]
s.U = prim_to_cons(s.W)
for P0 in P0s:
    s.pb = P_ATM_PSIA / P0
    s.hist = []
    s.run(6000, cfl=0.8, order=2, report=0)
    f = s.fields()
    np.savez_compressed(os.path.join(OUT, f"P0_{P0:05.2f}.npz"), X=s.X, Y=s.Y, xc=s.xc, yc=s.yc, P0_psia=P0, **f)
    print("done", P0, flush=True)
