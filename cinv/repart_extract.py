"""Extract BOTH representations from the re-partitioned test parts, with labels
transferred from the original faces.

A split face's children inherit its label, so per-face mIoU stays exactly
comparable to the clean test set.  Each child face is matched to its parent by
minimum point-to-face distance from the child's centroid (0 when it lies on the
parent), restricted to parents on the same analytic surface.

Produces one pickle per part in RP_OUT holding
  - our canonical region graph (f2r, feats, src, dst, ef) + transferred
    per-face labels `y` and the child->parent map `parent` (-1 = rejected),
  - AAGNet's attributed adjacency graph (`aag`), via its own extractor.

Reads the original STEP/labels from DS_ROOT and the re-partitioned STEP from
RP_IN.  Runs in the geometry environment (pythonocc).

Env: DS_ROOT  RP_IN (default <DS_ROOT>/steps_repartitioned)
     RP_OUT (default <DS_ROOT>/repart_eval)  NPROC  + feature flags
Usage: python repart_extract.py
"""
from __future__ import annotations
import os, sys, json, pickle, warnings, time
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
RP_STEP = os.environ.get("RP_IN", os.path.join(ROOT, "steps_repartitioned"))
OUT = os.environ.get("RP_OUT", os.path.join(ROOT, "repart_eval"))
os.makedirs(OUT, exist_ok=True)
SCHEMA = json.load(open(os.path.join(AAG, "feature_lists/all.json")))


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

def read(path):
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone
    rd = STEPControl_Reader()
    if rd.ReadFile(path) != IFSelect_RetDone:
        return None
    rd.TransferRoots()
    return rd.OneShape()


def _face_probe_points(f, n=3):
    """Up to n*n ON-FACE sample points (trimmed-interior UV grid).

    Interior surface points are used rather than the area centroid: the
    centroid of an annulus lies in its hole, and a curved face's centroid
    need not lie on the surface at all."""
    from OCC.Core.BRepTools import breptools
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.BRepTopAdaptor import BRepTopAdaptor_FClass2d
    from OCC.Core.TopAbs import TopAbs_IN
    from OCC.Core.gp import gp_Pnt2d
    u0, u1, v0, v1 = breptools.UVBounds(f)
    ad = BRepAdaptor_Surface(f)
    cls = BRepTopAdaptor_FClass2d(f, 1e-9)
    pts = []
    for i in range(n):
        for j in range(n):
            u = u0 + (u1 - u0) * (i + 0.5) / n
            v = v0 + (v1 - v0) * (j + 0.5) / n
            if cls.Perform(gp_Pnt2d(u, v)) == TopAbs_IN:
                pts.append(ad.Value(u, v).Coord())
    return pts


def transfer_labels(orig, rep, y_orig):
    """Child face -> parent face label by MULTI-POINT surface overlap with
    explicit rejection: every interior probe point of the child must lie on
    the chosen parent within TOL, the runner-up must be clearly worse, and
    unmatched children get parent -1 (dropped from evaluation and counted)
    rather than a guess.  Returns (labels, parents), or (None, None) if the
    original's face count disagrees with the label vector."""
    from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
    from OCC.Core.gp import gp_Pnt
    from region_graph import face_list
    from surface_graph import surface_key

    TOL = 1e-6
    fo, fr = face_list(orig), face_list(rep)
    if len(fo) != len(y_orig):
        return None, None
    ko = [surface_key(f) for f in fo]
    out = np.full(len(fr), -1, dtype=np.int64)
    par = np.full(len(fr), -1, dtype=np.int64)
    for i, f in enumerate(fr):
        k = surface_key(f)
        cand = [j for j in range(len(fo)) if ko[j] == k] or list(range(len(fo)))
        pts = _face_probe_points(f)
        if not pts:
            continue
        verts = [BRepBuilderAPI_MakeVertex(gp_Pnt(*c)).Vertex() for c in pts]
        scores = []
        for j in cand:
            try:
                dmax = max(BRepExtrema_DistShapeShape(v, fo[j]).Value()
                           for v in verts)
            except Exception:
                dmax = 1e30
            scores.append((dmax, j))
        scores.sort()
        best_d, best = scores[0]
        second_d = scores[1][0] if len(scores) > 1 else 1e30
        # accept only an unambiguous on-surface match; each of the three
        # rules applies unconditionally (in particular two parents inside
        # tolerance are rejected even when the best distance is ~0)
        if best_d > TOL:
            continue                          # rejected: not on any parent
        if second_d <= TOL:
            continue                          # rejected: ambiguous (two
                                              # candidates within tolerance)
        if second_d < 10 * max(best_d, 1e-12):
            continue                          # rejected: no clear margin
        out[i] = y_orig[best]
        par[i] = best
    return out, par


def one(nm):
    dst = os.path.join(OUT, nm + ".pkl")
    if os.path.exists(dst):
        return "cached"
    sp_o = os.path.join(ROOT, "steps", nm + ".step")
    if not os.path.exists(sp_o):        # CADSynth uses step/<id>.stp  # alt_paths
        for _alt in (os.path.join(ROOT, "step", nm + ".stp"),
                     os.path.join(ROOT, "step", nm + ".step")):
            if os.path.exists(_alt):
                sp_o = _alt; break
    sp_r = os.path.join(RP_STEP, nm + ".step")
    lp = os.path.join(ROOT, "labels", nm + ".json")
    if not os.path.exists(lp):          # CADSynth uses label/  # label_alt
        lp = os.path.join(ROOT, "label", nm + ".json")
    if not all(os.path.exists(p) for p in (sp_o, sp_r, lp)):
        return "missing"
    try:
        from dataset.AAGExtractor import AAGExtractor, scale_solid_to_unit_box
        from region_graph import build, face_list

        orig, rep = read(sp_o), read(sp_r)
        if orig is None or rep is None:
            return "readfail"
        raw = json.load(open(lp))
        if isinstance(raw, dict) and "labels" in raw:      # F360seg / CADSynth
            y_o = np.asarray(raw["labels"], dtype=np.int64)
        else:                                              # MFInstSeg / MFCAD++
            lab = raw[0][1]["seg"]
            y_o = np.array([lab[str(i)] for i in range(len(lab))], dtype=np.int64)
        n_o = len(face_list(orig))
        if len(y_o) != n_o:
            return "labelmismatch"
        _tr = transfer_labels(orig, rep, y_o)
        if _tr is None:
            return "transferfail"
        y_r, parent = _tr

        g = build(scale_solid_to_unit_box(read(sp_r)))
        if g is None:
            return "nograph"
        f2r, feats, (src, dsti), ef = g
        if len(f2r) != len(y_r):
            return "sizemismatch"
        try:
            aag = AAGExtractor(sp_r, SCHEMA).process()
        except Exception:
            aag = None
        with open(dst, "wb") as fh:
            pickle.dump({"code_sha": CODE_SHA, "f2r": f2r, "feats": feats, "src": src, "dst": dsti,
                         "ef": ef, "y": y_r, "n_faces_orig": n_o, "aag": aag,
                         "parent": parent}, fh)
        return "ok" if aag is not None else "ok_noaag"
    except Exception as e:
        return f"err:{type(e).__name__}"


def main():
    names = [os.path.basename(p)[:-5] for p in
             __import__("glob").glob(os.path.join(RP_STEP, "*.step"))]
    print(f"[in] {RP_STEP}\n[out] {OUT}", flush=True)
    print(f"[repart-extract] {len(names)} parts", flush=True)
    t0 = time.time(); stats = {}
    with Pool(int(os.environ.get("NPROC", "36"))) as p:
        for i, r in enumerate(p.imap_unordered(one, names, chunksize=8)):
            stats[r] = stats.get(r, 0) + 1
            if (i + 1) % 2000 == 0:
                print(f"  {i+1}/{len(names)} {time.time()-t0:.0f}s {stats}", flush=True)
    print(f"[repart-extract] done {time.time()-t0:.0f}s {stats}", flush=True)


if __name__ == "__main__":
    main()
