"""NURBS-conversion extraction: the same test solids with every surface and
curve re-expressed as NURBS (BRepBuilderAPI_NurbsConvert, what a kernel
translation does), geometry unchanged.  Both representations are extracted
(our region graph and AAGNet's graph); labels are carried over unchanged,
since the conversion does not touch topology and face order is identical.

Writes one pickle per part to <DS_ROOT>/<OUT_DIR>.  Runs in the geometry
environment (pythonocc).

Env: DS_ROOT, OUT_DIR (default eval_nc_pfr), NPROC, feature flags as usual.
Usage: python nc_extract.py test
"""
from __future__ import annotations
import os, sys, math, pickle, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths
AAG = paths.AAGNET
sys.path.insert(0, AAG)
sys.path.insert(0, os.path.join(AAG, "dataset"))
import json as _json

ROOT = os.environ.get("DS_ROOT", paths.dataset("mfinstseg"))
OUT = os.path.join(ROOT, os.environ.get("OUT_DIR", "eval_nc_pfr"))
os.makedirs(OUT, exist_ok=True)
SCHEMA = _json.load(open(os.path.join(AAG, "feature_lists/all.json")))


def _code_sha():
    """Provenance fingerprint stored in every pickle: git HEAD of this
    checkout plus a dirty flag ("unknown" outside a git checkout)."""
    import subprocess
    try:
        cw = os.path.dirname(os.path.abspath(__file__))
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=cw,
                           capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "-uno"], cwd=cw,
                               capture_output=True, text=True).stdout.strip()
        return h + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


CODE_SHA = _code_sha()

def one(arg):
    nm, sp, y = arg
    dst = os.path.join(OUT, nm + ".pkl")
    if os.path.exists(dst):
        return "cached"
    try:
        from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_Writer, \
            STEPControl_AsIs
        from OCC.Core.gp import gp_Trsf, gp_Ax1, gp_Pnt, gp_Dir
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
        from dataset.AAGExtractor import AAGExtractor, scale_solid_to_unit_box
        from region_graph import build

        rd = STEPControl_Reader(); rd.ReadFile(sp); rd.TransferRoots()
        # the nuisance: re-express every surface as NURBS (what kernel
        # translation does), geometry unchanged
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_NurbsConvert
        shape = BRepBuilderAPI_NurbsConvert(rd.OneShape(), True).Shape()

        sb = scale_solid_to_unit_box(shape)
        g = build(sb)
        if g is None:
            return "nograph"
        f2r, feats, (src, dsti), ef = g
        yy = np.asarray(y, np.int64)
        if len(yy) != len(f2r):
            return "labelmismatch"
        # AAGNet input needs a STEP file on disk for its extractor
        tmp = os.path.join(OUT, nm + ".tmp.step")
        w = STEPControl_Writer(); w.Transfer(shape, STEPControl_AsIs); w.Write(tmp)
        try:
            aag = AAGExtractor(tmp, SCHEMA).process()
        except Exception:
            aag = None
        finally:
            os.remove(tmp)
        pickle.dump({"code_sha": CODE_SHA, "f2r": f2r, "feats": feats, "src": src, "dst": dsti,
                     "ef": ef, "y": yy, "parent": np.arange(len(yy)),
                     "aag": aag}, open(dst, "wb"))
        return "ok" if aag is not None else "ok_noaag"
    except Exception as e:
        return "err:" + type(e).__name__


if __name__ == "__main__":
    import pc_extract as pe
    pe.ROOT = ROOT
    jobs = pe.jobs_split(sys.argv[1])
    print(f"[rot] {len(jobs)} jobs -> {OUT}", flush=True)
    t0 = time.time(); stats = {}
    with Pool(int(os.environ.get("NPROC", "32"))) as p:
        for r in p.imap_unordered(one, jobs, chunksize=8):
            stats[r] = stats.get(r, 0) + 1
    print(f"[rot] done in {time.time()-t0:.0f}s  {stats}", flush=True)
