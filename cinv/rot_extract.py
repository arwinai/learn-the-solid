"""Rotation-sensitivity extraction: the same test solids under a fixed random
SO(3) rotation per part (drawn from the part name, see nuisance_seed.py), both
representations (our region graph and AAGNet's graph), labels carried over
unchanged (rotation does not touch topology, so face order is identical).

Writes one pickle per part to <DS_ROOT>/<OUT_DIR>.  Runs in the geometry
environment (pythonocc).

Env: DS_ROOT, OUT_DIR (default eval_rot_chs), NPROC, EPS_ANG (optional fixed
     rotation angle in radians, axis still drawn per part), feature flags.
Usage: python rot_extract.py test [max_parts]
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
OUT = os.path.join(ROOT, os.environ.get("OUT_DIR", "eval_rot_chs"))
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
        shape = rd.OneShape()
        # fixed by the part name, so every model's cell sees the same rotation
        # and the cell can be regenerated exactly (see nuisance_seed.py)
        from nuisance_seed import rot_params
        v, ang = rot_params(nm)
        if "EPS_ANG" in os.environ:
            ang = float(os.environ["EPS_ANG"])  # micro-dose: fixed magnitude, seeded axis
        tr = gp_Trsf()
        tr.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(*v)), ang)
        shape = BRepBuilderAPI_Transform(shape, tr, True).Shape()

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
    if len(sys.argv) > 2:
        jobs = jobs[:int(sys.argv[2])]
    print(f"[rot] {len(jobs)} jobs -> {OUT}", flush=True)
    t0 = time.time(); stats = {}
    with Pool(int(os.environ.get("NPROC", "32"))) as p:
        for r in p.imap_unordered(one, jobs, chunksize=8):
            stats[r] = stats.get(r, 0) + 1
    print(f"[rot] done in {time.time()-t0:.0f}s  {stats}", flush=True)
