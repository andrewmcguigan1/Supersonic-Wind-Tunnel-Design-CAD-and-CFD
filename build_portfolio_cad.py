"""
Detailed, portfolio-grade CAD assembly of the supersonic blowdown wind tunnel.

Built in inches from tunnel_geometry.CASE, exported at true size in millimetres.
Each physical component is its own part in the STEP assembly.
Coordinates: x along the flow (0 at the throat), y up, z across the span.
"""
import os, math
import numpy as np
import cadquery as cq
from tunnel_geometry import CASE, build_contour

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output", "generated_cad")
os.makedirs(os.path.join(OUT, "parts"), exist_ok=True)
IN_MM = 25.4

c = CASE
xs, ys, st = build_contour(c)
h, w = c["h_test"], c["w_test"]
X_PL0, X_CONV, X_NE, X_TE, X_DE = st["x_plenum"], st["x_conv"], st["x_nozzle_exit"], st["x_test_end"], st["x_diff_exit"]
TH_D = math.radians(c["diff_angle"])
SIDE_A = math.radians(c["sidewall_angle"])
DW = (X_TE - X_NE) * math.tan(SIDE_A)          # Part B side-wall growth per side
FLOOR = -42.0                                   # floor level (tunnel centreline 42 in up)
BIG = 400.0

def yw(x): return float(np.interp(x, xs, ys))                     # passage half height
def zw(x): return float(np.interp(x, [X_PL0, X_NE, X_TE, X_DE], [w/2, w/2, w/2 + DW, w/2 + DW]))

# ------------------------------------------------------------------ primitives
def box(x0, x1, y0, y1, z0, z1):
    return cq.Workplane("XY").box(x1 - x0, y1 - y0, z1 - z0).translate(((x0+x1)/2, (y0+y1)/2, (z0+z1)/2))

def cyl_x(r, x0, x1, y=0.0, z=0.0):
    return cq.Workplane("YZ").circle(r).extrude(x1 - x0).translate((x0, y, z))

def cyl_y(r, y0, y1, x=0.0, z=0.0):
    return cq.Workplane("XZ").circle(r).extrude(-(y1 - y0)).translate((x, y0, z))

def cyl_z(r, z0, z1, x=0.0, y=0.0):
    return cq.Workplane("XY").circle(r).extrude(z1 - z0).translate((x, y, z0))

def tube_x(ro, ri, x0, x1, y=0.0, z=0.0):
    return cyl_x(ro, x0, x1, y, z).cut(cyl_x(ri, x0 - 0.01, x1 + 0.01, y, z))

def prism_xy(pts, z0, z1):
    return cq.Workplane("XY").polyline(pts).close().extrude(z1 - z0).translate((0, 0, z0))

def fluid_solid():
    """Air passage (in), used to cut matching openings in flanges."""
    pts = [(float(a), float(b)) for a, b in zip(xs, ys)]
    pts += [(float(a), -float(b)) for a, b in zip(xs[::-1], ys[::-1])]
    side = cq.Workplane("XY").polyline(pts).close().extrude(BIG/2, both=True)
    X = [X_PL0 - 5, X_NE, X_TE, X_DE + 5]
    Z = [w/2, w/2, w/2 + DW, w/2 + DW]
    pp = [(a, b) for a, b in zip(X, Z)] + [(a, -b) for a, b in zip(X[::-1], Z[::-1])]
    plan = cq.Workplane("XZ").polyline(pp).close().extrude(BIG/2, both=True)
    return side.intersect(plan)

FLUID = fluid_solid()
# extend the passage past the open exhaust so flanges there are open
FLUID_EXT = FLUID.union(box(X_DE - 0.5, X_DE + 3, -yw(X_DE) + 0.001, yw(X_DE) - 0.001, -zw(X_DE) + 0.001, zw(X_DE) - 0.001))

# hex bolt along x with head at x_head pointing in direction s (+1 / -1)
def hex_bolt_x(y, z, x_head, grip, s=+1, d=0.5):
    """Hex bolt along x: head bearing face at x_head, clamping `grip` of material, nut on the far side."""
    hd = 1.5 * d / math.cos(math.pi / 6)     # across corners
    hh, nh = 0.65 * d, 0.85 * d
    L = grip + nh + 0.3
    head = cq.Workplane("YZ").polygon(6, hd).extrude(hh)
    shank = cq.Workplane("YZ").circle(d / 2 * 0.98).extrude(L)
    nut = cq.Workplane("YZ").polygon(6, hd).extrude(nh).cut(cq.Workplane("YZ").circle(d / 2).extrude(nh))
    if s > 0:
        b = head.translate((x_head - hh, 0, 0)).union(shank.translate((x_head, 0, 0))) \
                .union(nut.translate((x_head + grip, 0, 0)))
    else:
        b = head.translate((x_head, 0, 0)).union(shank.translate((x_head - L, 0, 0))) \
                .union(nut.translate((x_head - grip - nh, 0, 0)))
    return b.translate((0, y, z))

def bolt_pattern_rect(hy, hz, pitch=2.0):
    """Points on a rectangle |y| = hy, |z| = hz spaced about `pitch` apart."""
    pts = []
    ny = max(2, int(round(2 * hy / pitch)) + 1); nz = max(2, int(round(2 * hz / pitch)) + 1)
    for yv in np.linspace(-hy, hy, ny):
        pts += [(yv, -hz), (yv, hz)]
    for zv in np.linspace(-hz, hz, nz)[1:-1]:
        pts += [(-hy, zv), (hy, zv)]
    return pts

def rect_flange(x0, x1, hy, hz, holes, d_hole=0.56, fluid=FLUID):
    f = box(x0, x1, -hy, hy, -hz, hz).cut(fluid)
    for (yy, zz) in holes:
        f = f.cut(cyl_x(d_hole / 2, x0 - 0.1, x1 + 0.1, yy, zz))
    return f

def joint(xj, hy, hz, pad=1.6, label=""):
    """Mating flange pair at xj (each 1 in thick) and its bolt set."""
    by, bz = hy - 0.8, hz - 0.8
    holes = bolt_pattern_rect(by, bz)
    up = rect_flange(xj - 1, xj, hy, hz, holes)
    dn = rect_flange(xj, xj + 1, hy, hz, holes, fluid=FLUID_EXT)
    bolts = None
    for (yy, zz) in holes:
        b = hex_bolt_x(yy, zz, xj - 1, 2.0, +1)
        bolts = b if bolts is None else bolts.union(b)
    return up, dn, bolts

PARTS = []      # (name, workplane, rgb, group)
def add(name, wp, rgb, group):
    PARTS.append((name, wp, rgb, group))

STEEL = (0.50, 0.55, 0.60)      # painted structural steel
ALU = (0.78, 0.80, 0.83)        # machined aluminium / stainless
BLUE = (0.18, 0.40, 0.75)       # test-section paint
BOLT = (0.22, 0.22, 0.24)
GLASS = (0.62, 0.85, 0.95)
BRASS = (0.80, 0.62, 0.25)
RED = (0.75, 0.12, 0.10)
YELLOW = (0.95, 0.72, 0.10)
STAND = (0.95, 0.55, 0.10)
MODEL = (0.85, 0.20, 0.15)

# ================================================================== 1. PLENUM
Hp = c["plenum_h"] / 2; t_pl = 1.0
x_pl_a, x_pl_b = X_PL0, X_CONV - 1          # body ends where the plenum flange starts
pl_out = box(x_pl_a - t_pl, x_pl_b, -Hp - t_pl, Hp + t_pl, -w/2 - t_pl, w/2 + t_pl)
pl_in = box(x_pl_a, x_pl_b + 0.1, -Hp, Hp, -w/2, w/2)
plenum = pl_out.cut(pl_in)
for xr in np.linspace(x_pl_a + 2, x_pl_b - 2, 4):            # external stiffener ribs
    rib = box(xr - 0.375, xr + 0.375, -Hp - 2.0, Hp + 2.0, -w/2 - 2.0, w/2 + 2.0).cut(
          box(xr - 1, xr + 1, -Hp - t_pl + 0.01, Hp + t_pl - 0.01, -w/2 - t_pl + 0.01, w/2 + t_pl - 0.01))
    plenum = plenum.union(rib)
# supply inlet: welded nozzle + pipe flange on the upstream end cap
PIPE_RO, PIPE_RI = 2.25, 2.013                 # 4 in sch 40
plenum = plenum.union(tube_x(PIPE_RO, PIPE_RI, x_pl_a - t_pl - 4, x_pl_a - t_pl + 0.01))
plenum = plenum.cut(cyl_x(PIPE_RI, x_pl_a - t_pl - 5, x_pl_a + 0.5))
# instrument bosses (top: pressure transducer, side: thermocouple)
plenum = plenum.union(cyl_y(0.75, Hp + t_pl - 0.1, Hp + t_pl + 0.75, x=x_pl_a + 4.5, z=0))
plenum = plenum.union(cyl_z(0.6, w/2 + t_pl - 0.1, w/2 + t_pl + 0.6, x=x_pl_a + 4.5, y=2.5))
plenum = plenum.cut(cyl_y(0.25, Hp - 1, Hp + 3, x=x_pl_a + 4.5)).cut(cyl_z(0.2, w/2 - 1, w/2 + 3, x=x_pl_a + 4.5, y=2.5))
pf_up, pf_dn, pf_bolts = joint(X_CONV, Hp + t_pl + 1.6, w/2 + t_pl + 1.6)
plenum = plenum.union(pf_up)
add("Plenum chamber", plenum, STEEL, "plenum")

# flow conditioning: honeycomb + two screens
hc = box(x_pl_a + 7, x_pl_a + 9, -Hp, Hp, -w/2, w/2)
cell = 0.7
pts = []
for i, yy in enumerate(np.arange(-Hp + 0.5, Hp - 0.3, cell * 0.866)):
    off = 0 if i % 2 == 0 else cell / 2
    for zz in np.arange(-w/2 + 0.45 + off, w/2 - 0.3, cell):
        pts.append((yy, zz))
hc = hc.cut(cq.Workplane("YZ").pushPoints(pts).polygon(6, cell * 0.92).extrude(3).translate((x_pl_a + 6.5, 0, 0)))
add("Honeycomb flow straightener", hc, ALU, "plenum")
for k, xsr in enumerate([x_pl_a + 11.5, x_pl_a + 14]):
    sc = box(xsr, xsr + 0.12, -Hp, Hp, -w/2, w/2)
    p2 = [(yy, zz) for yy in np.arange(-Hp + 0.35, Hp - 0.2, 0.45) for zz in np.arange(-w/2 + 0.35, w/2 - 0.2, 0.45)]
    sc = sc.cut(cq.Workplane("YZ").pushPoints(p2).rect(0.33, 0.33).extrude(1).translate((xsr - 0.4, 0, 0)))
    add(f"Turbulence screen {k+1}", sc, ALU, "plenum")
# instruments
xi = x_pl_a + 4.5
pt = cyl_y(0.5, Hp + t_pl + 0.75, Hp + t_pl + 3.2, x=xi).union(
     cq.Workplane("XZ").polygon(6, 1.3).extrude(-0.6).translate((xi, Hp + t_pl + 0.75, 0))).union(
     cyl_y(0.18, Hp + t_pl + 3.2, Hp + t_pl + 4.2, x=xi))
add("Plenum pressure transducer", pt, ALU, "instrumentation")
tc = cyl_z(0.15, w/2 - 1.5, w/2 + t_pl + 2.5, x=xi, y=2.5).union(
     cq.Workplane("XY").polygon(6, 1.0).extrude(0.5).translate((xi, 2.5, w/2 + t_pl + 0.6))).union(
     cyl_z(0.4, w/2 + t_pl + 1.1, w/2 + t_pl + 2.3, x=xi, y=2.5))
add("Stagnation thermocouple probe", tc, BRASS, "instrumentation")

# ================================================================== 2. NOZZLE
Y_BLK = Hp + 1.0                     # outer face of the contoured blocks
x_n0, x_n1 = X_CONV + 1, X_NE - 1    # blocks sit between the two nozzle flanges
m = (xs >= x_n0) & (xs <= x_n1)
cx = np.concatenate([[x_n0], xs[m], [x_n1]]); cy = np.array([yw(v) for v in cx])
top_pts = [(float(a), float(b)) for a, b in zip(cx, cy)] + [(x_n1, Y_BLK), (x_n0, Y_BLK)]
blk_top = prism_xy(top_pts, -w/2, w/2)
blk_bot = prism_xy([(a, -b) for a, b in top_pts], -w/2, w/2)
add("Nozzle block, upper (MOC contour)", blk_top, ALU, "nozzle")
add("Nozzle block, lower (MOC contour)", blk_bot, ALU, "nozzle")
for s, nm in [(+1, "right"), (-1, "left")]:
    sp = box(x_n0, x_n1, -Y_BLK, Y_BLK, 0, 1.0).translate((0, 0, s * w/2 if s > 0 else -w/2 - 1.0))
    # dowel/cap-screw rows that clamp the blocks
    for xb in np.linspace(x_n0 + 1, x_n1 - 1, 6):
        for yb in (Y_BLK - 0.5, -Y_BLK + 0.5):
            zc = w/2 + 1.0 if s > 0 else -w/2 - 1.0
            sp = sp.union(cyl_z(0.3, min(zc, zc + 0.3 * s), max(zc, zc + 0.3 * s), x=xb, y=yb))
    add(f"Nozzle side plate, {nm}", sp, (0.62, 0.66, 0.72), "nozzle")
NZ_HY, NZ_HZ = Y_BLK + 1.6, w/2 + 1.0 + 1.6
add("Nozzle inlet flange", pf_dn.intersect(box(X_CONV, X_CONV + 1, -BIG, BIG, -BIG, BIG)).cut(FLUID), STEEL, "nozzle")
add("Bolts, plenum-nozzle joint", pf_bolts, BOLT, "fasteners")
TS_T = 1.5                                          # test-section wall thickness
TS_HY, TS_HZ = h/2 + TS_T, w/2 + DW + TS_T
j2_hy, j2_hz = max(NZ_HY, TS_HY + 1.6), max(NZ_HZ, TS_HZ + 1.6)
j2_up, j2_dn, j2_bolts = joint(X_NE, j2_hy, j2_hz)
add("Nozzle exit flange", j2_up, STEEL, "nozzle")
add("Bolts, nozzle-test section joint", j2_bolts, BOLT, "fasteners")

# ================================================================== 3. TEST SECTION
xa, xb = X_NE + 1, X_TE - 1
ts_top = box(xa, xb, h/2, h/2 + TS_T, -TS_HZ, TS_HZ)
ts_bot = box(xa, xb, -h/2 - TS_T, -h/2, -TS_HZ, TS_HZ)
XW = 0.5 * (X_NE + X_TE); R_WIN, R_SEAT = 2.5, 3.1
def side_wall(s):
    zi0, zi1 = w/2 + (xa - X_NE) * math.tan(SIDE_A), w/2 + (xb - X_NE) * math.tan(SIDE_A)
    pts = [(xa, s * zi0), (xb, s * zi1), (xb, s * TS_HZ), (xa, s * TS_HZ)]
    sw = cq.Workplane("XZ").polyline(pts).close().extrude(h/2, both=True)
    sw = sw.cut(cyl_z(R_WIN, -BIG, BIG, x=XW))
    zo = TS_HZ * s
    sw = sw.cut(cyl_z(R_SEAT, zo - s * 0.8 if s > 0 else zo, zo if s > 0 else zo + 0.8, x=XW)) if s > 0 else \
         sw.cut(cyl_z(R_SEAT, zo, zo + 0.8, x=XW))
    return sw
ts_body = ts_top.union(ts_bot).union(side_wall(+1)).union(side_wall(-1)).union(j2_dn)
j3_hy, j3_hz = TS_HY + 2.2, TS_HZ + 1.6
j3_up, j3_dn, j3_bolts = joint(X_TE, j3_hy, j3_hz)
ts_body = ts_body.union(j3_up)
# static pressure taps along the top wall
taps_x = np.linspace(xa + 1.0, xb - 1.0, 5)
for xt in taps_x:
    ts_body = ts_body.union(cyl_y(0.3, h/2 + TS_T - 0.05, h/2 + TS_T + 0.35, x=xt, z=-2.6))
    ts_body = ts_body.cut(cyl_y(0.03, h/2 - 0.1, h/2 + TS_T + 0.5, x=xt, z=-2.6))
add("Test section body", ts_body, BLUE, "test section")
for s, nm in [(+1, "right"), (-1, "left")]:
    zo = TS_HZ * s
    g0, g1 = (zo - 0.8, zo - 0.05) if s > 0 else (zo + 0.05, zo + 0.8)
    add(f"Schlieren window glass, {nm}", cyl_z(R_SEAT - 0.05, g0, g1, x=XW), GLASS, "test section")
    r0, r1 = (zo, zo + 0.5) if s > 0 else (zo - 0.5, zo)
    ring = cyl_z(3.75, r0, r1, x=XW).cut(cyl_z(R_WIN - 0.15, r0 - 1, r1 + 1, x=XW))
    bolts = None
    for k in range(8):
        a = 2 * math.pi * k / 8 + math.pi / 8
        bx, by = XW + 3.4 * math.cos(a), 3.4 * math.sin(a)
        ring = ring.cut(cyl_z(0.15, r0 - 1, r1 + 1, x=bx, y=by))
        hz0, hz1 = (r1, r1 + 0.22) if s > 0 else (r0 - 0.22, r0)
        b = cyl_z(0.21, hz0, hz1, x=bx, y=by).cut(
            cq.Workplane("XY").polygon(6, 0.2).extrude(0.2).translate((bx, by, hz1 - 0.12 if s > 0 else hz0 - 0.08)))
        bolts = b if bolts is None else bolts.union(b)
    add(f"Window retainer ring, {nm}", ring, ALU, "test section")
    add(f"Window retainer screws, {nm}", bolts, BOLT, "fasteners")
taps = None
for xt in taps_x:
    f = cq.Workplane("XZ").polygon(6, 0.5).extrude(-0.3).translate((xt, h/2 + TS_T + 0.35, -2.6)).union(
        cyl_y(0.09, h/2 + TS_T + 0.65, h/2 + TS_T + 1.6, x=xt, z=-2.6))
    taps = f if taps is None else taps.union(f)
add("Pressure tap fittings, test section", taps, BRASS, "instrumentation")
add("Bolts, test section-diffuser joint", j3_bolts, BOLT, "fasteners")

# test model: 10 deg half-angle cone-cylinder on a sting
x_tip = XW - 3.0; r_b = 0.75; L_cone = r_b / math.tan(math.radians(10))
cone = cq.Workplane("YZ").circle(0.001).workplane(offset=L_cone).circle(r_b).loft().translate((x_tip, 0, 0))
model = cone.union(cyl_x(r_b, x_tip + L_cone, x_tip + L_cone + 2.5))
add("Test model, 10 deg cone-cylinder", model, MODEL, "model")
x_sting1 = X_TE + 9.0
sting = cyl_x(0.3, x_tip + L_cone + 2.5, x_sting1 + 0.5).union(
        cq.Workplane("YZ").circle(0.3).workplane(offset=2.0).circle(0.55).loft().translate((x_tip + L_cone + 3.5, 0, 0)))
add("Model sting", sting, ALU, "model")

# ================================================================== 4. DIFFUSER
t_d = 0.75
xd0, xd1 = X_TE + 1, X_DE - 1
def ydi(x): return yw(x)
zd = w/2 + DW
top = prism_xy([(xd0, ydi(xd0)), (xd1, ydi(xd1)), (xd1, ydi(xd1) + t_d), (xd0, ydi(xd0) + t_d)], -zd - t_d, zd + t_d)
bot = prism_xy([(xd0, -ydi(xd0)), (xd0, -ydi(xd0) - t_d), (xd1, -ydi(xd1) - t_d), (xd1, -ydi(xd1))], -zd - t_d, zd + t_d)
sp_pts = [(xd0, -ydi(xd0)), (xd1, -ydi(xd1)), (xd1, ydi(xd1)), (xd0, ydi(xd0))]
diff = top.union(bot).union(prism_xy(sp_pts, zd, zd + t_d)).union(prism_xy(sp_pts, -zd - t_d, -zd))
diff = diff.union(j3_dn)
for xr in np.arange(X_TE + 10, X_DE - 4, 10.0):             # stiffening frames
    yo = ydi(xr) + t_d
    fr = box(xr - 0.375, xr + 0.375, -yo - 1.5, yo + 1.5, -zd - t_d - 1.5, zd + t_d + 1.5).cut(
         box(xr - 1, xr + 1, -yo + 0.01, yo - 0.01, -zd - t_d + 0.01, zd + t_d - 0.01))
    diff = diff.union(fr)
ex_hy, ex_hz = ydi(X_DE) + t_d + 1.6, zd + t_d + 1.6
ex_holes = bolt_pattern_rect(ex_hy - 0.8, ex_hz - 0.8)
diff = diff.union(rect_flange(X_DE - 1, X_DE, ex_hy, ex_hz, ex_holes, fluid=FLUID_EXT))
# wall pressure taps along the diffuser to locate the shock
dtaps_x = np.arange(X_TE + 4, X_DE - 4, 5.0)
for xt in dtaps_x:
    yt = ydi(xt) + t_d
    diff = diff.union(cyl_y(0.3, yt - 0.2, yt + 0.35, x=xt, z=-2.4)).cut(cyl_y(0.03, yt - 1.5, yt + 0.5, x=xt, z=-2.4))
# sting support strut: diamond aerofoil from the floor of the diffuser
xs0 = x_sting1 - 1.5
y_bot_out = -ydi(x_sting1) - t_d
strut_pts = [(xs0, 0), (xs0 + 1.5, 0.25), (xs0 + 3.0, 0), (xs0 + 1.5, -0.25)]
strut = cq.Workplane("XZ").polyline(strut_pts).close().extrude(-(abs(y_bot_out) + 0.8)).translate((0, y_bot_out - 0.8, 0))
strut = strut.union(box(xs0 - 0.75, xs0 + 3.75, y_bot_out - 1.6, y_bot_out - 0.8, -1.5, 1.5))
strut = strut.union(cyl_y(0.55, -0.6, 0.6, x=x_sting1, z=0))
diff = diff.cut(box(xs0 - 0.05, xs0 + 3.05, y_bot_out - 0.5, -ydi(x_sting1) + 0.3, -0.3, 0.3))
add("Diffuser (3 deg per wall)", diff, STEEL, "diffuser")
add("Sting support strut", strut, ALU, "model")
dt = None
for xt in dtaps_x:
    yt = ydi(xt) + t_d
    f = cq.Workplane("XZ").polygon(6, 0.5).extrude(-0.3).translate((xt, yt + 0.35, -2.4)).union(
        cyl_y(0.09, yt + 0.65, yt + 1.6, x=xt, z=-2.4))
    dt = f if dt is None else dt.union(f)
add("Pressure tap fittings, diffuser", dt, BRASS, "instrumentation")
# exhaust protection screen bolted to the exit flange
scr = rect_flange(X_DE, X_DE + 0.25, ex_hy, ex_hz, ex_holes, fluid=FLUID_EXT)
grid = box(X_DE + 0.05, X_DE + 0.2, -ydi(X_DE), ydi(X_DE), -zd, zd)
gp = [(yy, zz) for yy in np.arange(-ydi(X_DE) + 0.6, ydi(X_DE) - 0.3, 1.0) for zz in np.arange(-zd + 0.6, zd - 0.3, 1.0)]
grid = grid.cut(cq.Workplane("YZ").pushPoints(gp).rect(0.85, 0.85).extrude(1).translate((X_DE - 0.3, 0, 0)))
add("Exhaust debris screen", scr.union(grid), STEEL, "diffuser")
eb = None
for (yy, zz) in ex_holes:
    b = hex_bolt_x(yy, zz, X_DE - 1, 1.25, +1)
    eb = b if eb is None else eb.union(b)
add("Bolts, exhaust screen", eb, BOLT, "fasteners")

# ================================================================== 5. SUPPLY LINE
xp = X_PL0 - t_pl - 4            # end of plenum inlet nozzle
def pipe_flange(x0, thick=0.9, ro=4.5):
    f = cyl_x(ro, x0, x0 + thick).cut(cyl_x(PIPE_RI, x0 - 1, x0 + thick + 1))
    for k in range(8):
        a = 2 * math.pi * k / 8 + math.pi / 8
        f = f.cut(cyl_x(0.4, x0 - 1, x0 + thick + 1, 3.75 * math.sin(a), 3.75 * math.cos(a)))
    return f
def flange_bolts(xface):
    bs = None
    for k in range(8):
        a = 2 * math.pi * k / 8 + math.pi / 8
        b = hex_bolt_x(3.75 * math.sin(a), 3.75 * math.cos(a), xface + 0.9, 1.8, -1, d=0.625)
        bs = b if bs is None else bs.union(b)
    return bs
supply = []
# flange pairs and pipe spools going upstream (-x)
x = xp
spool = pipe_flange(x - 0.9).union(tube_x(PIPE_RO, PIPE_RI, x - 10, x - 0.9)).union(pipe_flange(x - 10.9 + 0.0))
add("Plenum inlet flange", pipe_flange(x), STEEL, "supply")
add("Bolts, plenum inlet", flange_bolts(x), BOLT, "fasteners")
add("Pipe spool 1", spool, STEEL, "supply")
x = x - 10.9
# gate valve (isolation)
gv0, gv1 = x - 10, x
valve = pipe_flange(gv1 - 0.9).union(pipe_flange(gv0)).union(cyl_x(3.2, gv0 + 0.9, gv1 - 0.9))
valve = valve.union(cq.Workplane("YZ").sphere(3.6).translate(((gv0 + gv1) / 2, 0, 0)).intersect(box(gv0 + 1, gv1 - 1, -5, 5, -5, 5)))
valve = valve.union(cyl_y(2.4, 2.5, 8.5, x=(gv0 + gv1) / 2)).union(cyl_y(3.2, 8.5, 9.4, x=(gv0 + gv1) / 2))
valve = valve.cut(cyl_x(PIPE_RI, gv0 - 1, gv1 + 1))
add("Isolation gate valve body", valve, (0.30, 0.42, 0.30), "supply")
xm = (gv0 + gv1) / 2
stem = cyl_y(0.4, 9.4, 15.5, x=xm)
wheel = cyl_y(4.5, 14.5, 15.0, x=xm).cut(cyl_y(3.9, 14, 16, x=xm)).union(cyl_y(0.8, 14.3, 15.2, x=xm))
for k in range(4):
    a = math.pi * k / 4
    wheel = wheel.union(cq.Workplane("XY").box(8.0, 0.35, 0.5).rotate((0, 0, 0), (0, 1, 0), math.degrees(a)).translate((xm, 14.75, 0)))
add("Gate valve stem", stem, ALU, "supply")
add("Gate valve handwheel", wheel, RED, "supply")
add("Bolts, valve-spool", flange_bolts(gv1), BOLT, "fasteners")
x = gv0
# dome-loaded pressure regulator (holds P0 during the run)
rg0, rg1 = x - 12, x
xr = (rg0 + rg1) / 2
reg = cyl_x(4.5, rg1 - 0.9, rg1).union(cyl_x(4.5, rg0, rg0 + 0.9)).union(cyl_x(PIPE_RO, rg0 + 0.5, rg1 - 0.5))
reg = reg.union(cq.Workplane("XY").sphere(4.2).translate((xr, 0.01, 0.01)))
reg = reg.union(cyl_y(2.0, 3.5, 6.0, x=xr)).union(cyl_y(4.6, 6.0, 7.2, x=xr))
reg = reg.cut(cyl_x(PIPE_RI, rg0 - 1, rg1 + 1))
for k in range(8):
    a = 2 * math.pi * k / 8 + math.pi / 8
    for x0f in (rg0, rg1 - 0.9):
        reg = reg.cut(cyl_x(0.4, x0f - 0.5, x0f + 1.4, 3.75 * math.sin(a), 3.75 * math.cos(a)))
dome = cyl_y(4.2, 7.2, 12.0, x=xr).union(cq.Workplane("XZ").sphere(4.2).translate((xr, 12.0, 0)).intersect(box(xr - 5, xr + 5, 12, 17, -5, 5)))
for k in range(12):
    a = 2 * math.pi * k / 12
    dome = dome.union(cyl_y(0.25, 7.2, 7.6, x=xr + 4.35 * math.cos(a), z=4.35 * math.sin(a)))
add("Pressure regulator body", reg, (0.55, 0.57, 0.60), "supply")
add("Regulator dome loader", dome, YELLOW, "supply")
gauge = cyl_z(1.6, -0.6, 0.6).rotate((0, 0, 0), (0, 0, 1), 0).translate((xr + 4.0, 9.5, 4.8)).union(
        cyl_z(1.4, 0.6, 0.75).translate((xr + 4.0, 9.5, 4.8))).union(
        box(xr + 3.8, xr + 4.2, 5.0, 8.0, 4.6, 5.0))
add("Dome pressure gauge", gauge, (0.9, 0.9, 0.9), "instrumentation")
add("Bolts, regulator-valve", flange_bolts(rg1), BOLT, "fasteners")
x = rg0
# spool and long-radius elbow down to the tank header below the floor
sp2 = pipe_flange(x - 0.9).union(tube_x(PIPE_RO, PIPE_RI, x - 8, x - 0.9))
add("Bolts, spool-regulator", flange_bolts(x), BOLT, "fasteners")
R_EL = 6.0
xe = x - 8
elbow = (cq.Workplane("YZ").workplane(offset=xe).circle(PIPE_RO).circle(PIPE_RI)
         .revolve(90, (-R_EL, 0, 0), (-R_EL, 1, 0)))
sp2 = sp2.union(elbow)
add("Pipe spool 2 with elbow", sp2, STEEL, "supply")
# vertical run to the floor header
y_top = -R_EL; x_v = xe - R_EL
vert = cq.Workplane("XZ").circle(PIPE_RO).circle(PIPE_RI).extrude(-(y_top - (FLOOR + 1))).translate((x_v, FLOOR + 1, 0))
floor_fl = cyl_y(6.0, FLOOR, FLOOR + 1.0, x=x_v).cut(cyl_y(PIPE_RI, FLOOR - 1, FLOOR + 2, x=x_v))
add("Supply riser from storage tanks", vert.union(floor_fl), STEEL, "supply")

# ================================================================== 6. SUPPORT STANDS
def stand(xc, y_top, half_w, name, depth=6.0):
    """Two square-tube legs, cross brace, saddle and base plates."""
    leg = 1.5
    parts = None
    for zz in (-half_w, half_w):
        lg = box(xc - leg, xc + leg, FLOOR + 0.75, y_top - 0.75, zz - leg, zz + leg).cut(
             box(xc - leg + 0.25, xc + leg - 0.25, FLOOR, y_top, zz - leg + 0.25, zz + leg - 0.25))
        parts = lg if parts is None else parts.union(lg)
        foot = box(xc - 4, xc + 4, FLOOR, FLOOR + 0.75, zz - 3.5, zz + 3.5)
        for fx in (-3, 3):
            for fz in (-2.5, 2.5):
                foot = foot.cut(cyl_y(0.35, FLOOR - 1, FLOOR + 2, x=xc + fx, z=zz + fz))
        parts = parts.union(foot)
    for yb in (FLOOR + 8, (FLOOR + y_top) / 2 + 2):
        parts = parts.union(box(xc - 1.0, xc + 1.0, yb - 1.0, yb + 1.0, -half_w, half_w))
    parts = parts.union(box(xc - depth / 2, xc + depth / 2, y_top - 0.75, y_top, -half_w - 2, half_w + 2))
    add(name, parts, STAND, "stands")

stand(-19.0, -Hp - t_pl - 2.0, 7.0, "Stand 1, plenum")
stand(X_NE + 0.0 - 4.0, -Y_BLK, 7.0, "Stand 2, nozzle")
stand(X_TE + 25.0, -ydi(X_TE + 25.0) - t_d, 7.0, "Stand 3, diffuser")
stand(X_TE + 55.0, -ydi(X_TE + 55.0) - t_d, 7.5, "Stand 4, diffuser exit")
stand(xr, -4.2, 5.0, "Stand 5, regulator")
stand(xm, -3.6, 5.0, "Stand 6, gate valve")

# ================================================================== export
def to_mm(wp):
    return cq.Compound.makeCompound([o for o in wp.vals()]).scale(IN_MM)

asm = cq.Assembly(name="Supersonic_Blowdown_Wind_Tunnel_M2")
bad = []
for i, (name, wp, rgb, group) in enumerate(PARTS):
    shp = to_mm(wp)
    if not shp.isValid() or shp.Volume() <= 0:
        bad.append(name)
    fname = f"{i+1:02d}_" + "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")
    cq.exporters.export(shp, os.path.join(OUT, "parts", fname + ".step"))
    cq.exporters.export(shp, os.path.join(OUT, "parts", fname + ".stl"), tolerance=0.1, angularTolerance=0.15)
    asm.add(shp, name=fname, color=cq.Color(*rgb, 1.0))
asm.export(os.path.join(OUT, "Wind_Tunnel_Assembly.step"))
# half-sections (z <= 0) of the flow-path parts, for cutaway renders only
CUTDIR = os.path.join(OUT, "_render_cut"); os.makedirs(CUTDIR, exist_ok=True)
for i, (name, wp, rgb, group) in enumerate(PARTS):
    if group in ("plenum", "nozzle", "test section", "diffuser", "fasteners") and "Bolts, plenum inlet" not in name:
        fname = f"{i+1:02d}_" + "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")
        half = wp.intersect(box(-BIG, BIG, -BIG, BIG, -BIG, 0.0))
        if half.vals():
            cq.exporters.export(to_mm(half), os.path.join(CUTDIR, fname + ".stl"), tolerance=0.1, angularTolerance=0.15)
print("parts:", len(PARTS), "invalid:", bad)
import json
json.dump([dict(file=f"{i+1:02d}_" + "".join(ch if ch.isalnum() else "_" for ch in n).strip("_"), name=n, rgb=r, group=g)
           for i, (n, _, r, g) in enumerate(PARTS)], open(os.path.join(OUT, "parts", "parts.json"), "w"), indent=1)
