"""Offscreen portfolio renders of the detailed assembly (VTK)."""
import os, json, vtk
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PD = os.path.join(ROOT, "output", "mesh")
OUT = os.path.join(ROOT, "output", "renders"); os.makedirs(OUT, exist_ok=True)
PARTS = json.load(open(os.path.join(PD, "parts.json")))
IN = 25.4

CUT = os.path.join(ROOT, "output", "mesh_cut")
def actor(p, clip=None):
    r = vtk.vtkSTLReader()
    cutf = os.path.join(CUT, p["file"] + ".stl")
    r.SetFileName(cutf if (clip is not None and os.path.exists(cutf)) else os.path.join(PD, p["file"] + ".stl"))
    n = vtk.vtkPolyDataNormals(); n.SetInputConnection(r.GetOutputPort()); n.SetFeatureAngle(40); n.SplittingOn()
    m = vtk.vtkPolyDataMapper(); m.SetInputConnection(n.GetOutputPort())
    a = vtk.vtkActor(); a.SetMapper(m); pr = a.GetProperty()
    pr.SetColor(*p["rgb"]); pr.SetInterpolationToPhong()
    metal = p["group"] in ("fasteners",) or p["rgb"][0] > 0.7 and p["rgb"][1] > 0.7
    pr.SetSpecular(0.5 if metal else 0.25); pr.SetSpecularPower(40); pr.SetDiffuse(0.8); pr.SetAmbient(0.28)
    if "glass" in p["name"].lower():
        pr.SetOpacity(0.35); pr.SetSpecular(0.9)
    return a

import numpy as np
from matplotlib import colormaps
def cfd_plane_actor(npz, field="M", vmin=0.0, vmax=2.6, z=-0.3):
    d = np.load(npz)
    X, Y, F = d["X"], d["Y"], d[field]
    Xf = np.concatenate([X[:, ::-1], X[:, 1:]], axis=1); Yf = np.concatenate([-Y[:, ::-1], Y[:, 1:]], axis=1)
    Ff = np.concatenate([F[:, ::-1], F], axis=1)
    ni, nj = Xf.shape
    pts = vtk.vtkPoints()
    for j in range(nj):
        for i in range(ni):
            pts.InsertNextPoint(Xf[i, j] * IN, Yf[i, j] * IN, z)
    sg = vtk.vtkStructuredGrid(); sg.SetDimensions(ni, nj, 1); sg.SetPoints(pts)
    arr = vtk.vtkDoubleArray(); arr.SetName(field)
    for j in range(nj - 1):
        for i in range(ni - 1):
            arr.InsertNextValue(Ff[i, j])
    sg.GetCellData().SetScalars(arr)
    lut = vtk.vtkLookupTable(); lut.SetNumberOfTableValues(256); lut.SetRange(vmin, vmax)
    cm = colormaps["turbo"]
    for k in range(256):
        r, g, b, _ = cm(k / 255); lut.SetTableValue(k, r, g, b, 1)
    lut.Build()
    gf = vtk.vtkStructuredGridGeometryFilter(); gf.SetInputData(sg)
    m = vtk.vtkPolyDataMapper(); m.SetInputConnection(gf.GetOutputPort()); m.SetLookupTable(lut)
    m.SetScalarRange(vmin, vmax); m.SetScalarModeToUseCellData()
    a = vtk.vtkActor(); a.SetMapper(m); a.GetProperty().SetAmbient(0.9); a.GetProperty().SetDiffuse(0.2)
    a.GetProperty().LightingOff()
    return a, lut

def floor_actor(x0, x1, z0, z1, y):
    src = vtk.vtkPlaneSource(); src.SetOrigin(x0, y, z0); src.SetPoint1(x1, y, z0); src.SetPoint2(x0, y, z1)
    m = vtk.vtkPolyDataMapper(); m.SetInputConnection(src.GetOutputPort())
    a = vtk.vtkActor(); a.SetMapper(m); a.GetProperty().SetColor(0.90, 0.91, 0.92); a.GetProperty().SetAmbient(0.6); a.GetProperty().SetDiffuse(0.4)
    return a

def render(name, pos, focal, up=(0, 1, 0), angle=30, size=(3600, 2000), clip=None, skip=(), only=None, floor=True, cfd=None, dark=False, bar=False):
    ren = vtk.vtkRenderer()
    ren.SetBackground(0.97, 0.97, 0.98); ren.SetBackground2(0.80, 0.84, 0.90); ren.GradientBackgroundOn()
    if dark:
        ren.SetBackground(0.03, 0.04, 0.07); ren.SetBackground2(0.16, 0.20, 0.28)
    if cfd:
        ca, lut = cfd_plane_actor(cfd); ren.AddActor(ca)
        if bar:
            sb = vtk.vtkScalarBarActor(); sb.SetLookupTable(lut); sb.SetTitle("Mach"); sb.SetNumberOfLabels(6); sb.SetLabelFormat("%.1f")
            sb.SetOrientationToHorizontal(); sb.SetPosition(0.30, 0.04); sb.SetWidth(0.40); sb.SetHeight(0.07)
            for tp in (sb.GetTitleTextProperty(), sb.GetLabelTextProperty()):
                tp.SetColor((0.95, 0.95, 0.95) if dark else (0.1, 0.1, 0.1)); tp.SetFontSize(40); tp.ItalicOff(); tp.ShadowOff()
            sb.SetUnconstrainedFontSize(True)
            ren.AddViewProp(sb)
    for p in PARTS:
        if any(s in p["name"] for s in skip): continue
        if cfd and p["group"] == "model": continue   # CFD is the empty tunnel, so hide the model
        if only and not any(s in p["name"] for s in only): continue
        if clip is not None and "glass" in p["name"].lower() and "right" in p["name"]: continue
        if clip is not None and p["group"] in ("plenum", "nozzle", "test section", "diffuser", "fasteners") \
                and "Bolts, plenum inlet" not in p["name"] and True and not os.path.exists(os.path.join(CUT, p["file"] + ".stl")):
            continue   # part lies entirely in the cut-away half
        cl = clip
        ren.AddActor(actor(p, cl))
    if floor:
        fa = floor_actor(-120 * IN, 100 * IN, -40 * IN, 40 * IN, -42 * IN - 0.5)
        if dark: fa.GetProperty().SetColor(0.12, 0.13, 0.16)
        ren.AddActor(fa)
    ren.SetUseShadows(False)
    for lp, inten in [((-60, 150, 120), 0.75), ((80, 80, -60), 0.35), ((0, -50, 100), 0.2)]:
        L = vtk.vtkLight(); L.SetPosition(*[v * IN for v in lp]); L.SetFocalPoint(0, 0, 0); L.SetIntensity(inten)
        L.SetLightTypeToSceneLight(); ren.AddLight(L)
    ren.AutomaticLightCreationOff()
    win = vtk.vtkRenderWindow(); win.SetOffScreenRendering(1); win.AddRenderer(ren); win.SetSize(*size)
    win.SetMultiSamples(8)
    cam = ren.GetActiveCamera()
    cam.SetPosition(*[v * IN for v in pos]); cam.SetFocalPoint(*[v * IN for v in focal]); cam.SetViewUp(*up)
    cam.SetViewAngle(angle); ren.ResetCameraClippingRange()
    win.Render()
    w2i = vtk.vtkWindowToImageFilter(); w2i.SetInput(win); w2i.Update()
    wr = vtk.vtkPNGWriter(); wr.SetFileName(os.path.join(OUT, name)); wr.SetInputConnection(w2i.GetOutputPort()); wr.Write()
    print("rendered", name)

if __name__ == "__main__":
    SOL = os.path.join(ROOT, "output", "cfd", "solution_shock5h.npz")
    import sys
    if "cfd_only" not in sys.argv:
        render("01_hero_iso.png", pos=(70, 70, 170), focal=(-8, -14, 0), angle=36)
        render("02_hero_dark.png", pos=(70, 60, 165), focal=(-8, -14, 0), angle=36, dark=True)
    render("03_cutaway_with_CFD_Mach_field.png", pos=(15, 30, 140), focal=(15, -4, 0), angle=40, clip=0.0, cfd=SOL, dark=True, bar=True)
    render("04_nozzle_test_section_CFD_closeup.png", pos=(10, 12, 42), focal=(8, 0, 0), angle=44, clip=0.0, cfd=SOL, dark=True, bar=True)
    if "cfd_only" in sys.argv: raise SystemExit
    render("05_test_section_closeup.png", pos=(30, 20, 40), focal=(16, 0, 0), angle=40)
    render("06_flow_conditioning_cutaway.png", pos=(2, 14, 38), focal=(0, 0, 0), angle=42, clip=0.0)
    render("07_supply_line.png", pos=(-35, 35, 75), focal=(-62, -5, 0), angle=40)
    render("08_side_elevation.png", pos=(-10, -10, 330), focal=(-10, -10, 0), angle=34)
