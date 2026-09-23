"""Point-cloud extraction for the DGCNN baseline, and the dataset job lister.

Point clouds are trivially partition-invariant, so the comparison that matters
for a point-based baseline is clean accuracy. Sample N points area-uniformly
from a fine triangulation, keep the source face id (for per-face voting) and
the per-face labels.

`jobs_split(split)` -> [(name, step_path, labels)] enumerates a dataset split
in the benchmark layouts (MFInstSeg / MFCAD++ / CADSynth / Fusion 360) and is
shared by the perturbation-cell extractors.

Env: DS_ROOT, PC_DIR (out dir name), N_PTS (default 2048).
Repart mode: RP_IN (step dir) + RP_Y (pkl dir whose 'y' provides transferred
labels, e.g. eval_rpA_chs) -- names come from the pkl dir in that mode.
Usage: python pc_extract.py <split|rp>
"""
from __future__ import annotations
import os, sys, json, glob, pickle, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths

AAG = paths.AAGNET
sys.path.insert(0, AAG)
sys.path.insert(0, os.path.join(AAG, "dataset"))

ROOT = os.environ.get("DS_ROOT", paths.dataset("f360seg"))
OUT = os.path.join(ROOT, os.environ.get("PC_DIR", "pc"))
N_PTS = int(os.environ.get("N_PTS", "2048"))
RP_IN = os.environ.get("RP_IN")
RP_Y = os.environ.get("RP_Y")
if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)


def solid_triangles(shape):
    """[(face_idx, p0, p1, p2, area, normal)] over a fine mesh of the solid."""
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepTools import breptools
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCC.Core.TopoDS import topods
    tris = []
    ex, fi = TopExp_Explorer(shape, TopAbs_FACE), 0
    while ex.More():
        f = topods.Face(ex.Current())
        breptools.Clean(f)
        BRepMesh_IncrementalMesh(f, 0.005, False, 0.2, True)
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation(f, loc)
        if tri is not None:
            trsf = loc.Transformation()
            nodes = []
            for i in range(1, tri.NbNodes() + 1):
                p = tri.Node(i).Transformed(trsf)
                nodes.append((p.X(), p.Y(), p.Z()))
            nodes = np.asarray(nodes)
            rev = f.Orientation() == TopAbs_REVERSED
            for t in range(1, tri.NbTriangles() + 1):
                a, b, c = tri.Triangle(t).Get()
                p0, p1, p2 = nodes[a - 1], nodes[b - 1], nodes[c - 1]
                cr = np.cross(p1 - p0, p2 - p0)
                ar = 0.5 * float(np.linalg.norm(cr))
                if ar <= 0:
                    continue
                nrm = cr / max(np.linalg.norm(cr), 1e-30)
                if rev:
                    nrm = -nrm
                tris.append((fi, p0, p1, p2, ar, nrm))
        ex.Next(); fi += 1
    return tris, fi


def one(arg):
    nm, sp, y = arg
    dst = os.path.join(OUT, nm + ".npz")
    if os.path.exists(dst):
        return "cached"
    try:
        from dataset.AAGExtractor import scale_solid_to_unit_box
        from OCC.Core.STEPControl import STEPControl_Reader
        rd = STEPControl_Reader(); rd.ReadFile(sp); rd.TransferRoots()
        shape = scale_solid_to_unit_box(rd.OneShape())
        tris, nf = solid_triangles(shape)
        if not tris or (y is not None and nf != len(y)):
            return "mismatch"
        A = np.array([t[4] for t in tris])
        rng = np.random.default_rng(abs(hash(nm)) % (2**32))
        pick = rng.choice(len(tris), size=N_PTS, p=A / A.sum())
        u, v = rng.random(N_PTS), rng.random(N_PTS)
        flip = u + v > 1
        u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
        pts, pf = np.zeros((N_PTS, 6), np.float32), np.zeros(N_PTS, np.int64)
        for k, ti in enumerate(pick):
            fi, p0, p1, p2, _, nrm = tris[ti]
            pts[k, :3] = p0 + u[k] * (p1 - p0) + v[k] * (p2 - p0)
            pts[k, 3:] = nrm
            pf[k] = fi
        np.savez_compressed(dst, pts=pts, pf=pf,
                            y=np.asarray(y, np.int64) if y is not None else np.zeros(0))
        return "ok"
    except Exception as e:
        return "err:" + type(e).__name__


def jobs_split(split):
    names = [l.strip() for l in open(os.path.join(ROOT, f"{split}.txt")) if l.strip()]
    out = []
    for nm in names:
        sp = os.path.join(ROOT, "steps", nm + ".step")
        if not os.path.exists(sp):
            for alt in (os.path.join(ROOT, "step", nm + ".stp"),
                        os.path.join(ROOT, "step", nm + ".step"),
                        os.path.join(ROOT, "breps", nm + ".step")):
                if os.path.exists(alt):
                    sp = alt; break
        lp = os.path.join(ROOT, "labels", nm + ".json")
        if not os.path.exists(lp):
            lp = os.path.join(ROOT, "label", nm + ".json")
        if not (os.path.exists(sp) and os.path.exists(lp)):
            continue
        raw = json.load(open(lp))
        if isinstance(raw, dict) and "labels" in raw:
            y = raw["labels"]
        elif isinstance(raw, list) and raw and isinstance(raw[0], int):
            y = raw
        else:
            lab = raw[0][1]["seg"]
            y = [lab[str(i)] for i in range(len(lab))]
        out.append((nm, sp, y))
    return out


def jobs_rp():
    out = []
    for fp in sorted(glob.glob(os.path.join(RP_Y, "*.pkl"))):
        nm = os.path.basename(fp)[:-4]
        sp = os.path.join(RP_IN, nm + ".step")
        if not os.path.exists(sp):
            continue
        out.append((nm, sp, pickle.load(open(fp, "rb"))["y"].tolist()))
    return out


if __name__ == "__main__":
    mode = sys.argv[1]
    js = jobs_rp() if mode == "rp" else jobs_split(mode)
    print(f"[pc] {mode}: {len(js)} jobs -> {OUT}", flush=True)
    t0 = time.time(); stats = {}
    with Pool(int(os.environ.get("NPROC", str(os.cpu_count() or 4)))) as p:
        for r in p.imap_unordered(one, js, chunksize=16):
            stats[r] = stats.get(r, 0) + 1
    print(f"[{mode}] done in {time.time()-t0:.0f}s  {stats}", flush=True)
