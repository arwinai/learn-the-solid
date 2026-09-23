"""Extract canonical region graphs for one split of a dataset.

Reads <root>/steps/<name>.step and <root>/labels/<name>.json (MFInstSeg /
MFCAD++ layout; CADSynth's {"labels": [...]} layout is also accepted), builds
the region graph and writes <root>/<RG_DIR>/<name>.npz with the face->region
map, node/edge features and per-face labels. Face order in the STEP file
matches the label indexing, so labels are attached by index. Solids are scaled
to the unit box exactly as AAGNet's extractor does, so features are comparably
normalised.

Usage: DS_ROOT=$CINV_DATA/mfinstseg RG_DIR=region_graphs python region_extract.py <train|val|test>
Set the feature flags (RECOG, PART_FRAME, ...) in the environment first.
"""
from __future__ import annotations
import os, sys, json, warnings, time
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
OUT = os.path.join(ROOT, os.environ.get("RG_DIR", "region_graphs"))
os.makedirs(OUT, exist_ok=True)


def one(nm):
    from dataset.AAGExtractor import scale_solid_to_unit_box
    from region_graph import build
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone

    dst = os.path.join(OUT, nm + ".npz")
    if os.path.exists(dst):
        return "cached"
    sp = os.path.join(ROOT, "steps", nm + ".step")
    if not os.path.exists(sp):
        for alt in (os.path.join(ROOT, "step", nm + ".stp"),
                    os.path.join(ROOT, "step", nm + ".step")):
            if os.path.exists(alt):
                sp = alt; break
    lp = os.path.join(ROOT, "labels", nm + ".json")
    if not os.path.exists(lp):
        lp = os.path.join(ROOT, "label", nm + ".json")
    if not (os.path.exists(sp) and os.path.exists(lp)):
        return "missing"
    try:
        rd = STEPControl_Reader()
        if rd.ReadFile(sp) != IFSelect_RetDone:
            return "readfail"
        rd.TransferRoots()
        shape = scale_solid_to_unit_box(rd.OneShape())
        g = build(shape)
        if g is None:
            return "nograph"
        f2r, feats, (src, dst_i), ef = g
        raw = json.load(open(lp))
        if isinstance(raw, dict) and "labels" in raw:          # CADSynth format
            y = np.asarray(raw["labels"], dtype=np.int64)
        else:                                                   # MFInstSeg/MFCAD++ format
            lab = raw[0][1]["seg"]
            y = np.array([lab[str(i)] for i in range(len(lab))], dtype=np.int64)
        if len(y) != len(f2r):
            return "labelmismatch"
        np.savez_compressed(dst, f2r=f2r, feats=feats, src=src, dst=dst_i,
                            ef=ef, y=y, code_sha=np.bytes_(CODE_SHA))
        return "ok"
    except Exception as e:
        return f"err:{type(e).__name__}"


def _code_sha():
    """Provenance fingerprint: git HEAD + dirty flag, stored in every cache
    file so a mixed-generation cache is detectable."""
    import subprocess
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           cwd=os.path.dirname(os.path.abspath(__file__)),
                           capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "-uno"],
                               cwd=os.path.dirname(os.path.abspath(__file__)),
                               capture_output=True, text=True).stdout.strip()
        return h + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


CODE_SHA = _code_sha()


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    names = [l.strip() for l in open(os.path.join(ROOT, f"{split}.txt")) if l.strip()]
    print(f"[{split}] {len(names)} parts", flush=True)
    t0 = time.time(); stats = {}
    with Pool(int(os.environ.get("NPROC", str(os.cpu_count() or 4)))) as p:
        for i, r in enumerate(p.imap_unordered(one, names, chunksize=16)):
            stats[r] = stats.get(r, 0) + 1
            if (i + 1) % 2000 == 0:
                el = time.time() - t0
                print(f"  {i+1}/{len(names)}  {el:.0f}s  eta {el/(i+1)*(len(names)-i-1):.0f}s  {stats}",
                      flush=True)
    print(f"[{split}] done in {time.time()-t0:.0f}s  {stats}", flush=True)


if __name__ == "__main__":
    main()
