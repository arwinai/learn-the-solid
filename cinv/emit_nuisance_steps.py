"""Emit rotated / NURBS-converted STEP dirs (+ .seg labels) for BRepNet.

Each part of the chosen split is rotated (per-part draw from nuisance_seed.py)
or NURBS-converted and written to <DS_ROOT>/steps_<mode>_bn[_train]/<name>.step
with a matching <name>.seg label file.  Labels transfer geometrically in the
ORIGINAL frame (inverse-rotate the re-read file first) because face order
survives neither Transform nor write.  Runs in the geometry environment
(pythonocc).

Env: DS_ROOT, EMIT_SPLIT (test|train, default test)
Usage: python emit_nuisance_steps.py <rot|nc>
"""
import os, sys, math, warnings
warnings.filterwarnings("ignore")
import numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths
AAG = paths.AAGNET
sys.path.insert(0, AAG); sys.path.insert(0, os.path.join(AAG, "dataset"))

MODE = sys.argv[1]
ROOT = os.environ.get("DS_ROOT", paths.dataset("mfinstseg"))
OUT = f"{ROOT}/steps_{MODE}_bn" + ("_train" if os.environ.get("EMIT_SPLIT") == "train" else "")
os.makedirs(OUT, exist_ok=True)

def one(arg):
    nm, sp, y = arg
    dst = os.path.join(OUT, nm + ".step")
    if os.path.exists(dst + ".seg" if False else os.path.join(OUT, nm + ".seg")):
        return "cached"
    try:
        from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs
        from OCC.Core.gp import gp_Trsf, gp_Ax1, gp_Pnt, gp_Dir
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform, BRepBuilderAPI_NurbsConvert
        from repart_extract import transfer_labels
        rd = STEPControl_Reader(); rd.ReadFile(sp); rd.TransferRoots()
        O = rd.OneShape()
        yy = np.asarray(y, np.int64)
        if MODE == "rot":
            from nuisance_seed import rot_params
            v, ang = rot_params(nm)
            tr = gp_Trsf(); tr.SetRotation(gp_Ax1(gp_Pnt(0,0,0), gp_Dir(*v)), ang)
            shape = BRepBuilderAPI_Transform(O, tr, True).Shape()
        else:
            shape = BRepBuilderAPI_NurbsConvert(O, True).Shape()
        w = STEPControl_Writer(); w.Transfer(shape, STEPControl_AsIs); w.Write(dst)
        rd2 = STEPControl_Reader(); rd2.ReadFile(dst); rd2.TransferRoots()
        B = rd2.OneShape()
        if MODE == "rot":
            tri = gp_Trsf(); tri.SetRotation(gp_Ax1(gp_Pnt(0,0,0), gp_Dir(*v)), -ang)
            B_match = BRepBuilderAPI_Transform(B, tri, True).Shape()
        else:
            B_match = B
        yb, _ = transfer_labels(O, B_match, yy)
        if yb is None:
            os.remove(dst); return "transfer_failed"
        np.savetxt(os.path.join(OUT, nm + ".seg"), np.asarray(yb, np.int64), fmt="%d")
        return "ok"
    except Exception as e:
        return "err:" + type(e).__name__

if __name__ == "__main__":
    import pc_extract as pe
    pe.ROOT = ROOT
    jobs = pe.jobs_split(os.environ.get("EMIT_SPLIT", "test"))
    stats = {}
    with Pool(32) as p:
        for r in p.imap_unordered(one, jobs, chunksize=8):
            stats[r] = stats.get(r, 0) + 1
    print(f"[{MODE}] {stats}", flush=True)
