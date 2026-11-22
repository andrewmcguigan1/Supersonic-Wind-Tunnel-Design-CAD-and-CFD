"""
Small 2D planar compressible-flow (Euler) solver for the wind tunnel passage.

Finite volume on a structured body-fitted grid, HLLC flux, MUSCL (van Leer limiter)
reconstruction of primitive variables, 3-stage SSP Runge-Kutta with local time
stepping to steady state. Upper half of the tunnel only (y = 0 is a symmetry plane).

Boundary conditions
  inlet  (plenum):  total pressure P0 and total temperature T0, flow along +x
  outlet (exhaust): static pressure p_b = P_atm when subsonic, extrapolated when supersonic
  top wall:         inviscid slip wall;  bottom: symmetry plane

Non-dimensional: P0 = 1, rho0 = 1 (so a0 = sqrt(gamma)); lengths in inches.
"""
import numpy as np
from numba import njit, prange

G = 1.4

# ------------------------------------------------------------------ grid
def make_grid(xs_wall, ys_wall, x_in, x_out, nx, ny, x_cluster=None, beta_y=0.0):
    """Node coordinates X, Y of shape (nx+1, ny+1)."""
    s = np.linspace(0, 1, nx + 1)
    if x_cluster is None:
        xn = x_in + (x_out - x_in) * s
    else:  # piecewise-uniform spacing: x_cluster = list of (x_break, relative_density)
        xb = np.array([x_in] + [b for b, _ in x_cluster] + [x_out])
        dens = np.array([d for _, d in x_cluster] + [x_cluster[-1][1]])
        w = np.diff(xb) * dens
        cum = np.concatenate([[0], np.cumsum(w)]) / w.sum()
        xn = np.interp(s, cum, xb)
    eta = np.linspace(0, 1, ny + 1)
    if beta_y > 0:  # mild clustering toward the wall
        eta = np.tanh(beta_y * eta) / np.tanh(beta_y)
        eta = 0.5 * (eta + np.linspace(0, 1, ny + 1))
    yw = np.interp(xn, xs_wall, ys_wall)
    X = np.repeat(xn[:, None], ny + 1, axis=1)
    Y = yw[:, None] * eta[None, :]
    return X, Y

def metrics(X, Y):
    nx, ny = X.shape[0] - 1, X.shape[1] - 1
    # i-faces (nx+1, ny): node (i,j)->(i,j+1); normal (dy, -dx)
    dxi = X[:, 1:] - X[:, :-1]; dyi = Y[:, 1:] - Y[:, :-1]
    Si = np.sqrt(dxi**2 + dyi**2); nxi = dyi / Si; nyi = -dxi / Si
    # j-faces (nx, ny+1): node (i,j)->(i+1,j); normal (-dy, dx)
    dxj = X[1:, :] - X[:-1, :]; dyj = Y[1:, :] - Y[:-1, :]
    Sj = np.sqrt(dxj**2 + dyj**2); nxj = -dyj / Sj; nyj = dxj / Sj
    # cell areas (shoelace)
    x1, y1 = X[:-1, :-1], Y[:-1, :-1]; x2, y2 = X[1:, :-1], Y[1:, :-1]
    x3, y3 = X[1:, 1:], Y[1:, 1:];     x4, y4 = X[:-1, 1:], Y[:-1, 1:]
    vol = 0.5 * ((x1 * y2 - x2 * y1) + (x2 * y3 - x3 * y2) + (x3 * y4 - x4 * y3) + (x4 * y1 - x1 * y4))
    xc = 0.25 * (x1 + x2 + x3 + x4); yc = 0.25 * (y1 + y2 + y3 + y4)
    return Si, nxi, nyi, Sj, nxj, nyj, vol, xc, yc

# ------------------------------------------------------------------ numerics
@njit(inline="always")
def vanleer(a, b):
    if a * b <= 0.0:
        return 0.0
    return 2.0 * a * b / (a + b)

@njit(inline="always")
def hllc(rL, uL, vL, pL, rR, uR, vR, pR, nx, ny):
    unL = uL * nx + vL * ny; unR = uR * nx + vR * ny
    EL = pL / (G - 1) + 0.5 * rL * (uL * uL + vL * vL)
    ER = pR / (G - 1) + 0.5 * rR * (uR * uR + vR * vR)
    cL = np.sqrt(G * pL / rL); cR = np.sqrt(G * pR / rR)
    SL = min(unL - cL, unR - cR); SR = max(unL + cL, unR + cR)
    SM = (pR - pL + rL * unL * (SL - unL) - rR * unR * (SR - unR)) / (rL * (SL - unL) - rR * (SR - unR))
    if SL >= 0.0:
        return (rL * unL, rL * uL * unL + pL * nx, rL * vL * unL + pL * ny, (EL + pL) * unL)
    if SR <= 0.0:
        return (rR * unR, rR * uR * unR + pR * nx, rR * vR * unR + pR * ny, (ER + pR) * unR)
    if SM >= 0.0:
        r, u, v, p, E, un, S = rL, uL, vL, pL, EL, unL, SL
    else:
        r, u, v, p, E, un, S = rR, uR, vR, pR, ER, unR, SR
    fac = r * (S - un) / (S - SM)
    Us0 = fac
    Us1 = fac * (u + (SM - un) * nx)
    Us2 = fac * (v + (SM - un) * ny)
    Us3 = fac * (E / r + (SM - un) * (SM + p / (r * (S - un))))
    F0 = r * un; F1 = r * u * un + p * nx; F2 = r * v * un + p * ny; F3 = (E + p) * un
    U0 = r; U1 = r * u; U2 = r * v; U3 = E
    return (F0 + S * (Us0 - U0), F1 + S * (Us1 - U1), F2 + S * (Us2 - U2), F3 + S * (Us3 - U3))

@njit(parallel=True)
def to_prim(U, W):
    for i in prange(U.shape[1]):
        for j in range(U.shape[2]):
            r = U[0, i, j]; u = U[1, i, j] / r; v = U[2, i, j] / r
            W[0, i, j] = r; W[1, i, j] = u; W[2, i, j] = v
            W[3, i, j] = (G - 1) * (U[3, i, j] - 0.5 * r * (u * u + v * v))

@njit
def apply_bc(W, nxi, nyi, nxj, nyj, P0, T0r, pb):
    nx = W.shape[1] - 4; ny = W.shape[2] - 4
    # top wall (j = ny) and bottom symmetry (j = 0): mirror about the face normal
    for i in range(nx):
        ii = i + 2
        for k in range(2):
            # top
            n1 = nxj[i, ny]; n2 = nyj[i, ny]
            src = ny + 1 - k; dst = ny + 2 + k
            u = W[1, ii, src]; v = W[2, ii, src]; un = u * n1 + v * n2
            W[0, ii, dst] = W[0, ii, src]; W[3, ii, dst] = W[3, ii, src]
            W[1, ii, dst] = u - 2 * un * n1; W[2, ii, dst] = v - 2 * un * n2
            # bottom
            n1 = nxj[i, 0]; n2 = nyj[i, 0]
            src = 2 + k; dst = 1 - k
            u = W[1, ii, src]; v = W[2, ii, src]; un = u * n1 + v * n2
            W[0, ii, dst] = W[0, ii, src]; W[3, ii, dst] = W[3, ii, src]
            W[1, ii, dst] = u - 2 * un * n1; W[2, ii, dst] = v - 2 * un * n2
    for j in range(W.shape[2]):
        # inlet: total conditions, static p extrapolated from first interior cell
        p = min(W[3, 2, j], 0.9999 * P0)
        M2 = 2 / (G - 1) * ((P0 / p) ** ((G - 1) / G) - 1)
        T = T0r / (1 + 0.5 * (G - 1) * M2)
        r = p / T                       # nondim gas constant R = 1 (p0 = rho0 = 1 -> T0 = 1)
        u = np.sqrt(M2 * G * T)
        for k in range(2):
            W[0, k, j] = r; W[1, k, j] = u; W[2, k, j] = 0.0; W[3, k, j] = p
        # outlet
        jj = j
        r = W[0, nx + 1, jj]; u = W[1, nx + 1, jj]; v = W[2, nx + 1, jj]; p = W[3, nx + 1, jj]
        c = np.sqrt(G * p / r)
        pg = p if u >= c else pb
        for k in range(2):
            W[0, nx + 2 + k, jj] = r; W[1, nx + 2 + k, jj] = u
            W[2, nx + 2 + k, jj] = v; W[3, nx + 2 + k, jj] = pg

@njit(parallel=True)
def residual(W, R, Si, nxi, nyi, Sj, nxj, nyj, vol, dt, cfl, order):
    nx = W.shape[1] - 4; ny = W.shape[2] - 4
    for i in prange(nx):
        for j in range(ny):
            for q in range(4):
                R[q, i, j] = 0.0
    # i-faces: compute per face row, owner-only accumulation to stay race-free
    for i in prange(nx):
        ii = i + 2
        for j in range(ny):
            jj = j + 2
            acc0 = 0.0; acc1 = 0.0; acc2 = 0.0; acc3 = 0.0; spec = 0.0
            for side in range(4):
                if side == 0:   # west face of cell (between ii-1 and ii)
                    a = ii - 1; b = ii; ja = jj; jb = jj; n1 = nxi[i, j]; n2 = nyi[i, j]; S = Si[i, j]; sg = -1.0
                    am = ii - 2; bp = ii + 1; jam = jj; jbp = jj
                elif side == 1: # east face
                    a = ii; b = ii + 1; ja = jj; jb = jj; n1 = nxi[i + 1, j]; n2 = nyi[i + 1, j]; S = Si[i + 1, j]; sg = 1.0
                    am = ii - 1; bp = ii + 2; jam = jj; jbp = jj
                elif side == 2: # south face
                    a = ii; b = ii; ja = jj - 1; jb = jj; n1 = nxj[i, j]; n2 = nyj[i, j]; S = Sj[i, j]; sg = -1.0
                    am = ii; bp = ii; jam = jj - 2; jbp = jj + 1
                else:           # north face
                    a = ii; b = ii; ja = jj; jb = jj + 1; n1 = nxj[i, j + 1]; n2 = nyj[i, j + 1]; S = Sj[i, j + 1]; sg = 1.0
                    am = ii; bp = ii; jam = jj - 1; jbp = jj + 2
                qL = np.empty(4); qR = np.empty(4)
                for q in range(4):
                    wa = W[q, a, ja]; wb = W[q, b, jb]
                    if order == 2:
                        qL[q] = wa + 0.5 * vanleer(wa - W[q, am, jam], wb - wa)
                        qR[q] = wb - 0.5 * vanleer(wb - wa, W[q, bp, jbp] - wb)
                    else:
                        qL[q] = wa; qR[q] = wb
                if qL[0] <= 0 or qL[3] <= 0 or qR[0] <= 0 or qR[3] <= 0:
                    for q in range(4):
                        qL[q] = W[q, a, ja]; qR[q] = W[q, b, jb]
                f0, f1, f2, f3 = hllc(qL[0], qL[1], qL[2], qL[3], qR[0], qR[1], qR[2], qR[3], n1, n2)
                acc0 += sg * f0 * S; acc1 += sg * f1 * S; acc2 += sg * f2 * S; acc3 += sg * f3 * S
                r = W[0, ii, jj]; u = W[1, ii, jj]; v = W[2, ii, jj]; c = np.sqrt(G * W[3, ii, jj] / r)
                spec += (abs(u * n1 + v * n2) + c) * S
            R[0, i, j] = acc0; R[1, i, j] = acc1; R[2, i, j] = acc2; R[3, i, j] = acc3
            dt[i, j] = cfl * 2.0 * vol[i, j] / spec

@njit(parallel=True)
def update(U, U0, R, vol, dt, a0, a1):
    nx = R.shape[1]; ny = R.shape[2]
    for i in prange(nx):
        for j in range(ny):
            f = dt[i, j] / vol[i, j]
            for q in range(4):
                U[q, i + 2, j + 2] = a0 * U0[q, i + 2, j + 2] + a1 * (U[q, i + 2, j + 2] - f * R[q, i, j])

def prim_to_cons(W):
    U = np.empty_like(W)
    U[0] = W[0]; U[1] = W[0] * W[1]; U[2] = W[0] * W[2]
    U[3] = W[3] / (G - 1) + 0.5 * W[0] * (W[1]**2 + W[2]**2)
    return U

class Solver:
    def __init__(self, X, Y, P0=1.0, pb=0.5, T0=1.0):
        self.X, self.Y = X, Y
        (self.Si, self.nxi, self.nyi, self.Sj, self.nxj, self.nyj,
         self.vol, self.xc, self.yc) = metrics(X, Y)
        nx, ny = self.vol.shape
        self.nx, self.ny = nx, ny
        self.P0, self.pb, self.T0 = P0, pb, T0
        self.W = np.zeros((4, nx + 4, ny + 4))
        self.R = np.zeros((4, nx, ny)); self.dt = np.zeros((nx, ny))
        self.hist = []

    def init_from_1d(self, M_of_x, p_of_x):
        """Initialise every column from a quasi-1D solution (functions of x)."""
        M = M_of_x(self.xc); p = p_of_x(self.xc)
        T = self.T0 / (1 + 0.5 * (G - 1) * M**2)
        r = p / T; u = M * np.sqrt(G * T)
        s = slice(2, -2)
        self.W[0, s, s] = r; self.W[1, s, s] = u; self.W[2, s, s] = 0; self.W[3, s, s] = p
        self.W[:, :2, :] = self.W[:, 2:3, :]; self.W[:, -2:, :] = self.W[:, -3:-2, :]
        self.W[:, :, :2] = self.W[:, :, 2:3]; self.W[:, :, -2:] = self.W[:, :, -3:-2]
        self.U = prim_to_cons(self.W)

    def run(self, n_iter, cfl=0.8, order=2, report=500, tol=1e-9):
        U, W = self.U, self.W
        args = (self.Si, self.nxi, self.nyi, self.Sj, self.nxj, self.nyj, self.vol)
        rk = [(0.0, 1.0), (0.75, 0.25), (1 / 3, 2 / 3)]
        for it in range(n_iter):
            U0 = U.copy()
            for k, (a0, a1) in enumerate(rk):
                to_prim(U, W)
                apply_bc(W, self.nxi, self.nyi, self.nxj, self.nyj, self.P0, self.T0, self.pb)
                residual(W, self.R, *args, self.dt, cfl, order)
                if k == 0:
                    res = np.sqrt(np.mean((self.R[0] / self.vol) ** 2))
                update(U, U0, self.R, self.vol, self.dt, a0, a1)
            self.hist.append(res)
            if report and it % report == 0:
                print(f"  iter {it:6d}  rho-residual {res:.3e}", flush=True)
            if res < tol:
                break
        to_prim(U, W)
        return self.hist

    def fields(self):
        s = slice(2, -2)
        r, u, v, p = (self.W[q, s, s] for q in range(4))
        T = p / r; M = np.sqrt(u**2 + v**2) / np.sqrt(G * T)
        p0loc = p * (1 + 0.5 * (G - 1) * M**2) ** (G / (G - 1))
        return dict(rho=r, u=u, v=v, p=p, T=T, M=M, p0=p0loc)
