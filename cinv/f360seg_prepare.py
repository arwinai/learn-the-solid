"""Prepare the Fusion 360 Gallery SEGMENTATION benchmark (BRepNet's benchmark).

35,680 real designer-authored parts with per-face labels over 8 modelling
operations. Two things must be checked before it can be used:

1. FACE ORDER. The .seg labels are indexed by the order the dataset's own
   exporter enumerated faces. Our pipeline enumerates with TopExp after a STEP
   round-trip. If the orders disagree, every label lands on the wrong face and
   all downstream numbers are garbage -- silently.

   We validate SEMANTICALLY, which needs no ground-truth ordering: faces labelled
   Fillet/Chamfer must be overwhelmingly non-planar, and faces labelled
   ExtrudeEnd/CutEnd must be overwhelmingly planar. If the ordering were wrong,
   these correlations would collapse toward the base rate.

2. LABEL PURITY under region merging. This dataset labels faces by the OPERATION
   that created them, so two coplanar adjacent faces can carry DIFFERENT labels
   (an extruded end next to a cut end). Merging them would destroy supervision in
   a way MFInstSeg never exercised, so the ceiling has to be measured explicitly
   (done downstream, on the region graphs).

Inputs: <data>/f360seg/labels_raw/<name>.seg and <data>/f360seg/steps/<name>.step.
Outputs: the face-order validation printout, train/val/test split files
(seeded shuffle, 70/15/15) and <data>/f360seg/labels/<name>.json in the form
the extractors read.  Runs in the geometry environment (pythonocc).

Usage: python f360seg_prepare.py
"""
from __future__ import annotations
import os, sys, glob, json, warnings, random
warnings.filterwarnings("ignore")
import numpy as np
from multiprocessing import Pool


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths
AAG = paths.AAGNET
sys.path.insert(0, AAG); sys.path.insert(0, os.path.join(AAG, "dataset"))

D = paths.dataset("f360seg")
CLASSES = ["ExtrudeSide", "ExtrudeEnd", "CutSide", "CutEnd",
           "Fillet", "Chamfer", "RevolveSide", "RevolveEnd"]
PLANAR_EXPECTED = {1, 3}          # ExtrudeEnd, CutEnd -> flat
CURVED_EXPECTED = {4, 5}          # Fillet, Chamfer -> not flat (chamfer can be flat)


def check_one(nm):
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.GeomAbs import GeomAbs_Plane
    from region_graph import face_list
    lp = os.path.join(D, "labels_raw", nm + ".seg")
    sp = os.path.join(D, "steps", nm + ".step")
    if not (os.path.exists(lp) and os.path.exists(sp)):
        return None
    try:
        y = np.array([int(x) for x in open(lp).read().split()], dtype=np.int64)
        rd = STEPControl_Reader()
        if rd.ReadFile(sp) != IFSelect_RetDone:
            return None
        rd.TransferRoots()
        faces = face_list(rd.OneShape())
        if len(faces) != len(y):
            return {"count_mismatch": (len(faces), len(y))}
        planar = np.array([BRepAdaptor_Surface(f).GetType() == GeomAbs_Plane
                           for f in faces])
        return {"y": y, "planar": planar}
    except Exception:
        return None


def main():
    names = sorted(os.path.basename(p)[:-4]
                   for p in glob.glob(os.path.join(D, "labels_raw", "*.seg")))
    print(f"[f360seg] {len(names)} parts", flush=True)
    sample = names[:2500]
    res = []
    with Pool(34) as p:
        for r in p.imap_unordered(check_one, sample, chunksize=8):
            if r:
                res.append(r)
    mism = [r for r in res if "count_mismatch" in r]
    ok = [r for r in res if "y" in r]
    print(f"  parsed {len(res)}  face/label count mismatch {len(mism)}"
          f" ({100*len(mism)/max(len(res),1):.1f}%)")

    Y = np.concatenate([r["y"] for r in ok])
    P = np.concatenate([r["planar"] for r in ok])
    print(f"\n=== semantic validation of face ORDER ({len(Y)} faces) ===")
    print(f"  base rate planar: {100*P.mean():.1f}%")
    print(f"{'class':14s} {'n':>7s} {'% planar':>10s}")
    for c in range(8):
        m = Y == c
        if m.sum() == 0:
            continue
        print(f"{CLASSES[c]:14s} {int(m.sum()):7d} {100*P[m].mean():9.1f}%")
    pl = np.isin(Y, list(PLANAR_EXPECTED)); cu = np.isin(Y, list(CURVED_EXPECTED))
    print(f"\n  ExtrudeEnd/CutEnd planar : {100*P[pl].mean():.1f}%  (expect >> base)")
    print(f"  Fillet/Chamfer planar    : {100*P[cu].mean():.1f}%  (expect << base)")
    gap = P[pl].mean() - P[cu].mean()
    print(f"  separation               : {100*gap:.1f} points")
    print("  -> a wrong face ordering would drive this separation toward 0.")

    # splits
    rng = random.Random(0)
    keep = [n for n in names]
    rng.shuffle(keep)
    n = len(keep); a, b = int(0.7 * n), int(0.85 * n)
    for s, part in (("train", keep[:a]), ("val", keep[a:b]), ("test", keep[b:])):
        open(os.path.join(D, f"{s}.txt"), "w").write("\n".join(part) + "\n")
    print(f"\n[splits] train {a} val {b-a} test {n-b}")

    # labels/ in the json form our extractor understands
    os.makedirs(os.path.join(D, "labels"), exist_ok=True)
    for nm in names:
        t = os.path.join(D, "labels", nm + ".json")
        if os.path.exists(t):
            continue
        y = [int(x) for x in open(os.path.join(D, "labels_raw", nm + ".seg")).read().split()]
        json.dump({"file_name": nm, "labels": y}, open(t, "w"))
    print(f"[labels] wrote {len(names)} json label files")


if __name__ == "__main__":
    main()
