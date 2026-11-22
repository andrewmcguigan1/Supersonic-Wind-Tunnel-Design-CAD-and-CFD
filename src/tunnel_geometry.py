"""
Parametric geometry for the supersonic blowdown wind tunnel (planar, 2D nozzle).

All lengths in inches. x = 0 at the nozzle throat, flow in +x.
Layout:  plenum | converging section | MOC diverging nozzle | test section | diffuser | exhaust

Change CASE below to build another configuration (section, Mach, diffuser angle).
"""
import math
import numpy as np

g = 1.4
P_ATM_PSIA = 14.696

# ---------------------------------------------------------------- design case
CASE = dict(
    h_test=6.0,          # test-section height, in
    w_test=6.0,          # test-section width (span), in
    M=2.0,               # design test-section Mach number
    diff_angle=3.0,      # diffuser half-angle, deg, per wall (top and bottom)
    P0_psia=28.83,       # stagnation pressure from Part A (p02 = Patm criterion)
    L_test_over_h=2.0,   # test-section length / h  (Part B used L = 2h)
    sidewall_angle=0.4816,  # Part B side-wall divergence, deg each side (6x6, M2)
    L_diff_over_h=10.0,  # diffuser length / h (shock sits at 5h)
    plenum_h=14.0,       # plenum/settling-chamber height, in (A/A* ~ 3.9, M ~ 0.15)
    L_plenum=18.0,       # plenum length, in
    L_conv=10.0,         # converging section length, in
    n_char=40,           # number of characteristics in MOC fan
    wall_t=0.75,         # structural wall thickness for the CAD shell, in
)

# ---------------------------------------------------------------- gas dynamics
def A_Astar(M):
    return (1 / M) * ((2 / (g + 1)) * (1 + (g - 1) / 2 * M**2)) ** ((g + 1) / (2 * (g - 1)))

def M_from_A(r, supersonic=True):
    lo, hi = (1.0, 50.0) if supersonic else (1e-6, 1.0)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f = A_Astar(mid) - r
        if supersonic:
            lo, hi = (mid, hi) if f < 0 else (lo, mid)
        else:
            lo, hi = (lo, mid) if f < 0 else (mid, hi)
    return 0.5 * (lo + hi)

def nu(M):
    a = math.sqrt((g + 1) / (g - 1))
    return a * math.atan(math.sqrt((M**2 - 1) / a**2)) - math.atan(math.sqrt(M**2 - 1))

def M_from_nu(v):
    lo, hi = 1.0, 50.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if nu(mid) < v else (lo, mid)
    return 0.5 * (lo + hi)

def mu(M): return math.asin(1 / M)
def p0_p(M): return (1 + (g - 1) / 2 * M**2) ** (g / (g - 1))
def shock_M2(M1): return math.sqrt((1 + (g - 1) / 2 * M1**2) / (g * M1**2 - (g - 1) / 2))
def shock_p02_p01(M1):
    a = ((g + 1) * M1**2 / ((g - 1) * M1**2 + 2)) ** (g / (g - 1))
    b = ((g + 1) / (2 * g * M1**2 - (g - 1))) ** (1 / (g - 1))
    return a * b

# ---------------------------------------------------------------- MOC nozzle
def moc_min_length_nozzle(M_exit, y_throat, N=40):
    """Anderson-style minimum-length planar nozzle (sharp-corner throat), upper half.
    Returns wall points (x, y) starting at the corner, and the interior grid points."""
    th_max = nu(M_exit) / 2
    ths = [th_max * (i + 1) / N for i in range(N)]       # flow angle on each C- from corner
    P = {}   # (i, j) -> (x, y, theta, nu)   C-_i  x  C+_j  (j <= i)
    corner = (0.0, y_throat)
    for i in range(N):
        for j in range(i + 1):
            th = ths[i] - ths[j]
            v = ths[i] + ths[j]
            Mloc = M_from_nu(v); m = mu(Mloc)
            # previous point on C-_i
            if j == 0:
                xa, ya, tha, ma = corner[0], corner[1], ths[i], mu(M_from_nu(ths[i]))
            else:
                xa, ya, tha, va = P[(i, j - 1)]; ma = mu(M_from_nu(va))
            sm = math.tan(0.5 * ((tha - ma) + (th - m)))
            if j == i:   # centerline point
                x = xa - ya / sm; y = 0.0
            else:        # previous point on C+_j is (i-1, j)
                xb, yb, thb, vb = P[(i - 1, j)]; mb = mu(M_from_nu(vb))
                sp = math.tan(0.5 * ((thb + mb) + (th + m)))
                x = (yb - ya + sm * xa - sp * xb) / (sm - sp)
                y = ya + sm * (x - xa)
            P[(i, j)] = (x, y, th, v)
    # wall points: C+_j leaves (N-1, j) and meets the wall
    wall = [(corner[0], corner[1], th_max)]
    for j in range(N):
        xb, yb, thb, vb = P[(N - 1, j)]; mb = mu(M_from_nu(vb))
        xw0, yw0, thw0 = wall[-1]
        sw = math.tan(0.5 * (thw0 + thb))
        sp = math.tan(thb + mb)
        x = (yb - yw0 + sw * xw0 - sp * xb) / (sw - sp)
        y = yw0 + sw * (x - xw0)
        wall.append((x, y, thb))
    wall = np.array(wall)
    return wall, P

# ---------------------------------------------------------------- full contour
def build_contour(c=CASE):
    """Return dict with upper-wall contour y(x) (half height) for the whole tunnel
    and key stations."""
    h = c["h_test"]; M = c["M"]
    ht = h / A_Astar(M)                     # throat height
    wall, _ = moc_min_length_nozzle(M, ht / 2, c["n_char"])
    # rescale tiny MOC discretisation error so the exit is exactly h/2
    xw, yw = wall[:, 0], wall[:, 1]
    yw = ht / 2 + (yw - ht / 2) * (h / 2 - ht / 2) / (yw[-1] - ht / 2)
    x_ne = xw[-1]
    # converging section: smooth (C1 at both ends) contraction from plenum to throat
    Lc = c["L_conv"]; Hp = c["plenum_h"] / 2
    s = np.linspace(0, 1, 60)
    blend = 6 * s**5 - 15 * s**4 + 10 * s**3         # zero slope & curvature at both ends
    xc = -Lc + Lc * s
    yc = Hp + (ht / 2 - Hp) * blend
    x_plen0 = -Lc - c["L_plenum"]
    L_t = c["L_test_over_h"] * h
    L_d = c["L_diff_over_h"] * h
    x_ts1 = x_ne + L_t
    x_de = x_ts1 + L_d
    th_d = math.radians(c["diff_angle"])
    xs = np.concatenate([[x_plen0], xc, xw[1:], [x_ts1, x_de]])
    ys = np.concatenate([[Hp], yc, yw[1:], [h / 2, h / 2 + L_d * math.tan(th_d)]])
    st = dict(x_plenum=x_plen0, x_conv=-Lc, x_throat=0.0, x_nozzle_exit=x_ne,
              x_test_end=x_ts1, x_shock_design=x_ts1 + 5 * h, x_diff_exit=x_de,
              h_throat=ht, h_exit=2 * ys[-1])
    return xs, ys, st

def y_wall(x, xs, ys):
    return np.interp(x, xs, ys)

# ---------------------------------------------------------------- quasi-1D check
def quasi1d_shock_location(c=CASE, P0=None, pb=P_ATM_PSIA):
    """Normal-shock position in the diffuser for a given P0 and *static* exit
    pressure pb (finite-length diffuser).  Returns x of the shock (in)."""
    xs, ys, st = build_contour(c)
    P0 = P0 or c["P0_psia"]
    At = st["h_throat"]; Ae = st["h_exit"]
    def pe_for(xsh):
        A1 = 2 * y_wall(xsh, xs, ys)
        M1 = M_from_A(A1 / At, True)
        p02 = P0 * shock_p02_p01(M1)
        A2s = At / shock_p02_p01(M1)
        Me = M_from_A(Ae / A2s, False)
        return p02 / p0_p(Me), M1
    a, b = st["x_test_end"] + 1e-6, st["x_diff_exit"] - 1e-6
    fa = pe_for(a)[0] - pb; fb = pe_for(b)[0] - pb
    if fa * fb > 0:
        return None
    for _ in range(100):
        m = 0.5 * (a + b); fm = pe_for(m)[0] - pb
        if fa * fm <= 0: b = m
        else: a, fa = m, fm
    return 0.5 * (a + b), pe_for(0.5 * (a + b))[1]

if __name__ == "__main__":
    xs, ys, st = build_contour()
    for k, v in st.items():
        print(f"{k:16s} {v:9.3f}")
    print("nozzle length / h_throat =", st["x_nozzle_exit"] / st["h_throat"])
    print("theta_max (deg) =", math.degrees(nu(CASE['M']) / 2))
    print("quasi-1D shock (P0=%.2f, pb=Patm static):" % CASE["P0_psia"], quasi1d_shock_location())
    xs_d = st["x_test_end"] + 5 * CASE["h_test"]
    print("design shock station x =", xs_d, " -> distance into diffuser =", 5 * CASE["h_test"])
