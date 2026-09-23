"""Emit the ALL-nuisance composition (re-partition + rigid motion + NURBS
re-expression) as a STEP dir (+ .seg labels) for BRepNet — the same
composition combo_extract.py evaluates for the pkl-based models.
Labels transfer geometrically in the FINAL frame (rotate the analytic
original by the same motion, match the converted children against it);
face order in the written file is what BRepNet reads, so .seg is computed
from the re-read file. Seed is crc32(name): deterministic across runs.
Output: <DS_ROOT>/steps_combo_bn[_train]/<name>.step + <name>.seg.
Runs in the geometry environment (pythonocc).

Env: DS_ROOT, EMIT_SPLIT (test|train, default test)
Usage: python emit_combo_bn.py
"""
import os, sys, math, zlib, warnings
warnings.filterwarnings("ignore")
import numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths
AAG = paths.AAGNET
sys.path.insert(0, AAG); sys.path.insert(0, os.path.join(AAG, "dataset"))

ROOT = os.environ.get("DS_ROOT", paths.dataset("mfinstseg"))
OUT = f"{ROOT}/steps_combo_bn" + ("_train" if os.environ.get("EMIT_SPLIT") == "train" else "")
os.makedirs(OUT, exist_ok=True)


def one(arg):
    nm, sp, y = arg
    dst = os.path.join(OUT, nm + ".step")
    if os.path.exists(os.path.join(OUT, nm + ".seg.ok")):
        return "cached"
    try:
        from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs
        from OCC.Core.gp import gp_Trsf, gp_Ax1, gp_Pnt, gp_Dir
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform, BRepBuilderAPI_NurbsConvert
        from repartition_testset import split_faces
        from repart_extract import transfer_labels
        rd = STEPControl_Reader(); rd.ReadFile(sp); rd.TransferRoots()
        O = rd.OneShape()
        yy = np.asarray(y, np.int64)
        seed = zlib.crc32(nm.encode()) % (2**31)
        rng = np.random.default_rng(seed)
        v = rng.normal(size=3); v /= np.linalg.norm(v)
        ang = float(rng.uniform(0.2, math.pi))
        tr = gp_Trsf(); tr.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(*v)), ang)
        shape = split_faces(O, 1.0, seed)
        shape = BRepBuilderAPI_Transform(shape, tr, True).Shape()
        shape = BRepBuilderAPI_NurbsConvert(shape, True).Shape()
        if not os.path.exists(dst):
            w = STEPControl_Writer(); w.Transfer(shape, STEPControl_AsIs); w.Write(dst)
        rd2 = STEPControl_Reader(); rd2.ReadFile(dst); rd2.TransferRoots()
        B = rd2.OneShape()
        # split_faces can leave orphan shell fragments beside the solid in
        # the written file; BRepNet's loader (occwl load_step) reads solids
        # only, so labels must be computed over the solid's faces alone.
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_SOLID
        ex = TopExp_Explorer(B, TopAbs_SOLID)
        if not ex.More():
            os.remove(dst); return "no_solid"
        sol = ex.Current(); ex.Next()
        if ex.More():
            os.remove(dst); return "multi_solid"
        ref = BRepBuilderAPI_Transform(O, tr, True).Shape()
        yb, _ = transfer_labels(ref, sol, yy)
        if yb is None:
            os.remove(dst); return "transfer_failed"
        # match occwl's ignore_orientation enumeration exactly
        from region_graph import face_list
        fl = face_list(sol)
        keep, seen = [], {}
        for i, f in enumerate(fl):
            js = seen.setdefault(hash(f), [])
            if not any(f.IsSame(fl[j]) for j in js):
                keep.append(i); js.append(i)
        yb = np.asarray(yb, np.int64)[keep]
        np.savetxt(os.path.join(OUT, nm + ".seg"), yb, fmt="%d")
        open(os.path.join(OUT, nm + ".seg.ok"), "w").write("1")
        return "ok" if len(keep) == len(fl) else "ok_dedup"
    except Exception as e:
        return "err:" + type(e).__name__


if __name__ == "__main__":
    import pc_extract as pe
    pe.ROOT = ROOT
    jobs = pe.jobs_split(os.environ.get("EMIT_SPLIT", "test"))
    print(f"[combo_bn] {len(jobs)} jobs -> {OUT}", flush=True)
    stats = {}
    with Pool(32) as p:
        for r in p.imap_unordered(one, jobs, chunksize=8):
            stats[r] = stats.get(r, 0) + 1
    print(f"[combo_bn] {stats}", flush=True)
