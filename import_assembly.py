"""Read the SolidWorks assembly in cad/ (external-reference STEP) and write located STLs
(+ half-section STLs for cutaways) for rendering. usage: python src/import_assembly.py"""
import sys, os, json, re
import cadquery as cq
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TDocStd import TDocStd_Document
from OCP.TCollection import TCollection_ExtendedString
from OCP.XCAFDoc import XCAFDoc_DocumentTool
from OCP.TDF import TDF_LabelSequence, TDF_Label
from OCP.TDataStd import TDataStd_Name
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "cad")
OUT = os.path.join(ROOT, "output", "mesh"); CUT = os.path.join(ROOT, "output", "mesh_cut")
os.makedirs(OUT, exist_ok=True); os.makedirs(CUT, exist_ok=True)
meta = {p["file"][:2]: p for p in json.load(open(os.path.join(SRC, "parts_metadata.json")))}
os.chdir(SRC)
doc = TDocStd_Document(TCollection_ExtendedString("d"))
r = STEPCAFControl_Reader(); r.SetNameMode(True); r.ReadFile("wind_tunnel_assembly.STEP"); r.Transfer(doc)
st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
free = TDF_LabelSequence(); st.GetFreeShapes(free)
comps = TDF_LabelSequence(); st.GetComponents_s(free.Value(1), comps, False)
half_box = cq.Solid.makeBox(20000, 20000, 10000, cq.Vector(-10000, -10000, -10000))
parts = []
for i in range(1, comps.Length() + 1):
    lab = comps.Value(i); ref = TDF_Label(); st.GetReferredShape_s(lab, ref)
    nm = TDataStd_Name(); ref.FindAttribute(TDataStd_Name.GetID_s(), nm); name = nm.Get().ToExtString()
    shp = cq.Shape.cast(st.GetShape_s(lab))
    m = dict(meta[name[:2]]); m["file"] = re.sub(r"[^0-9A-Za-z_]", "_", name)
    shp.exportStl(os.path.join(OUT, m["file"] + ".stl"), 0.1 * 25.4, 0.15)
    if m["group"] in ("plenum", "nozzle", "test section", "diffuser", "fasteners") and "Bolts__plenum_inlet" not in name:
        h = shp.intersect(half_box)
        if h.Solids():
            h.exportStl(os.path.join(CUT, m["file"] + ".stl"), 0.1 * 25.4, 0.15)
    parts.append(m)
json.dump(parts, open(os.path.join(OUT, "parts.json"), "w"), indent=1)
print("parts", len(parts))
