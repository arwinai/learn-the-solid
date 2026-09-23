"""ALL-nuisance combined-cell extraction (re-partition + rigid motion + NURBS
re-expression), for the pickle-based models.

Every model's features AND labels come from the same on-disk file.  Split
shapes can carry structurally defective faces that the STEP writer heals by
moving them to an orphan shell, which permutes face order; so the emitted
combo STEP (steps_combo_bn, written by emit_combo_bn.py with crc32 seeds) is
read, its solid taken, re-written alone if the file carries orphan fragments,
and our graph, AAGNet's graph and the transferred labels are all computed
from that one file's own read.  Labels transfer geometrically in the FINAL
frame: the analytic original is rotated by the same per-part motion and the
converted children are matched against it (repart_extract.transfer_labels).

Writes one pickle per part to <DS_ROOT>/<OUT_DIR>.  Runs in the geometry
environment (pythonocc).

Env: DS_ROOT, STEPS_SUB (default steps_combo_bn), OUT_DIR (default
     eval_combo2_pfr), NPROC + feature flags.
Usage: python combo_extract.py test
"""
from __future__ import annotations
import os, sys, math, zlib, pickle, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths
AAG = paths.AAGNET
sys.path.insert(0, AAG)
sys.path.insert(0, os.path.join(AAG, "dataset"))

ROOT = os.environ.get("DS_ROOT", paths.dataset("mfinstseg"))
STEPS = os.path.join(ROOT, os.environ.get("STEPS_SUB", "steps_combo_bn"))
OUT = os.path.join(ROOT, os.environ.get("OUT_DIR", "eval_combo2_pfr"))
os.makedirs(OUT, exist_ok=True)


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
    combo_fp = os.path.join(STEPS, nm + ".step")
    if not os.path.exists(combo_fp):
        return "no_combo_step"
    tmp = None
    try:
        from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_Writer, \
            STEPControl_AsIs
        from OCC.Core.gp import gp_Trsf, gp_Ax1, gp_Pnt, gp_Dir
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_SOLID
        from dataset.AAGExtractor import AAGExtractor, scale_solid_to_unit_box
        from region_graph import build, face_list
        from repart_extract import transfer_labels
        import json as _json

        rd = STEPControl_Reader(); rd.ReadFile(combo_fp); rd.TransferRoots()
        W = rd.OneShape()
        ex = TopExp_Explorer(W, TopAbs_SOLID)
        if not ex.More():
            return "no_solid"
        sol = ex.Current(); ex.Next()
        if ex.More():
            return "multi_solid"

        if len(face_list(W)) == len(face_list(sol)):
            src_fp, S = combo_fp, W          # file is already solid-only
        else:
            # orphan shell fragments beside the solid: re-write the solid
            # alone and use that file for every model
            tmp = os.path.join(OUT, nm + ".tmp.step")
            w = STEPControl_Writer(); w.Transfer(sol, STEPControl_AsIs); w.Write(tmp)
            rd2 = STEPControl_Reader(); rd2.ReadFile(tmp); rd2.TransferRoots()
            S = rd2.OneShape()
            e2 = TopExp_Explorer(S, TopAbs_SOLID)
            if not e2.More():
                return "heal_failed"
            e2.Next()
            if e2.More():
                return "heal_multi"
            src_fp = tmp

        # same deterministic motion the emitter applied (crc32 seed)
        rdo = STEPControl_Reader(); rdo.ReadFile(sp); rdo.TransferRoots()
        O = rdo.OneShape()
        yy = np.asarray(y, np.int64)
        rng = np.random.default_rng(zlib.crc32(nm.encode()) % (2**31))
        v = rng.normal(size=3); v /= np.linalg.norm(v)
        ang = float(rng.uniform(0.2, math.pi))
        tr = gp_Trsf(); tr.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(*v)), ang)
        ref = BRepBuilderAPI_Transform(O, tr, True).Shape()

        # AAGExtractor enumerates unique faces (ignore_orientation); our raw
        # explorer must agree or the alignment argument breaks
        fl = face_list(S)
        seen = {}
        for i, f in enumerate(fl):
            js = seen.setdefault(hash(f), [])
            if any(f.IsSame(fl[j]) for j in js):
                return "dup_faces"
            js.append(i)
        y_child, parent = transfer_labels(ref, S, yy)
        if y_child is None:
            return "transfer_failed"

        g = build(scale_solid_to_unit_box(S))
        if g is None:
            return "nograph"
        f2r, feats, (src, dsti), ef = g
        if len(y_child) != len(f2r):
            return "labelmismatch"

        schema = _json.load(open(os.path.join(AAG, "feature_lists/all.json")))
        try:
            aag = AAGExtractor(src_fp, schema).process()
        except Exception:
            aag = None
        pickle.dump({"code_sha": CODE_SHA, "f2r": f2r, "feats": feats, "src": src, "dst": dsti,
                     "ef": ef, "y": np.asarray(y_child, np.int64),
                     "parent": np.asarray(parent, np.int64), "aag": aag},
                    open(dst, "wb"))
        return "ok" if aag is not None else "ok_noaag"
    except Exception as e:
        return "err:" + type(e).__name__
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)


if __name__ == "__main__":
    import pc_extract as pe
    pe.ROOT = ROOT
    jobs = pe.jobs_split(sys.argv[1])
    print(f"[combo2] {len(jobs)} jobs, steps={STEPS} -> {OUT}", flush=True)
    t0 = time.time(); stats = {}
    with Pool(int(os.environ.get("NPROC", "32"))) as p:
        for r in p.imap_unordered(one, jobs, chunksize=8):
            stats[r] = stats.get(r, 0) + 1
    print(f"[combo2] done in {time.time()-t0:.0f}s  {stats}", flush=True)
