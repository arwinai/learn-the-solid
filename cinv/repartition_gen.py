"""Re-partition generator with several distinct partition FAMILIES.

Face partitions are not an enumerable group, so augmenting over one family
cannot confer invariance to another.  Testing that concretely needs at least
two families such that a model can be trained on one and tested on the other:

  famA  : 2 axis-aligned section planes (X then Y)      <- augmentation family
  famB  : 4 section planes incl. diagonal directions    <- held-out family
  famC  : ShapeUpgrade_UnifySameDomain, i.e. MERGING rather than splitting
  famK<k>: one axis-aligned plane, at most k faces split

famC is the adversarial case: it moves the face count DOWN, so no amount of
split-augmentation covers it.

Reads <DS_ROOT>/<split>.txt and <DS_ROOT>/steps/<name>.step, writes one
re-partitioned STEP per part to RP_OUT (default <DS_ROOT>/steps_rp<FAM>_<split>).
Volume and area of every output are checked against the input to 1e-9.
Runs in the geometry environment (pythonocc).

Env: DS_ROOT  RP_SPLIT (train|val|test)  RP_FAM (A|B|C|K<k>)  RP_OUT (dir)
Usage: RP_FAM=B RP_SPLIT=test python repartition_gen.py
"""
from __future__ import annotations
import os, sys, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths
ROOT = os.environ.get("DS_ROOT", paths.dataset("mfinstseg"))

SPLIT = os.environ.get("RP_SPLIT", "test")
FAM = os.environ.get("RP_FAM", "A")
OUT = os.environ.get("RP_OUT", os.path.join(ROOT, f"steps_rp{FAM}_{SPLIT}"))
os.makedirs(OUT, exist_ok=True)


def planes_for(fam, bb):
    """-> list of (origin, direction) for the section planes."""
    xm, ym, zm, xM, yM, zM = bb
    out = []
    if fam == "A":
        out.append(((xm + 0.5 * (xM - xm), 0, 0), (1, 0, 0)))
        out.append(((0, ym + 0.5 * (yM - ym), 0), (0, 1, 0)))
    elif fam == "B":
        for f in (0.33, 0.66):
            out.append(((xm + f * (xM - xm), ym + f * (yM - ym), 0), (1, 1, 0)))
            out.append(((xm + f * (xM - xm), 0, zm + f * (zM - zm)), (1, 0, 1)))
    return out


def split_with(shape, fam):
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section
    from OCC.Core.BRepFeat import BRepFeat_SplitShape
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_EDGE
    from OCC.Core.TopoDS import topods, TopoDS_Face
    from OCC.Core.gp import gp_Pln, gp_Pnt, gp_Dir
    from OCC.Core.ShapeUpgrade import ShapeUpgrade_UnifySameDomain

    # famK<k>: split at most k FACES -- a realistic-magnitude perturbation.
    # Designers do not slice every face; the question is how FEW splits it takes
    # to hurt.  Uses one section plane but adds only the first k ancestor edges.
    if fam.startswith("K"):
        kmax = int(fam[1:])
    else:
        kmax = 10 ** 6
    if fam == "C":
        u = ShapeUpgrade_UnifySameDomain(shape, True, True, True)
        u.Build()
        return u.Shape()

    b = Bnd_Box(); brepbndlib.Add(shape, b)
    bb = b.Get()
    big = max(bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2]) * 3
    cur = shape
    plist = planes_for("A", bb)[:1] if fam.startswith("K") else planes_for(fam, bb)
    for org, dr in plist:
        try:
            pln = gp_Pln(gp_Pnt(*org), gp_Dir(*dr))
            tool = BRepBuilderAPI_MakeFace(pln, -big, big, -big, big).Face()
            sec = BRepAlgoAPI_Section(cur, tool, False)
            sec.ComputePCurveOn1(True); sec.Approximation(True); sec.Build()
            if not sec.IsDone():
                continue
            sp = BRepFeat_SplitShape(cur)
            added = 0
            ex = TopExp_Explorer(sec.Shape(), TopAbs_EDGE)
            while ex.More():
                e = topods.Edge(ex.Current()); ex.Next()
                fc = TopoDS_Face()
                if not sec.HasAncestorFaceOn1(e, fc):
                    continue
                try:
                    sp.Add(e, topods.Face(fc)); added += 1
                except Exception:
                    pass
                if added >= kmax or added > 200:
                    break
            if added:
                sp.Build()
                if sp.Shape() is not None:
                    cur = sp.Shape()
        except Exception:
            continue
    return cur


def one(nm):
    from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.BRepGProp import brepgprop_VolumeProperties, brepgprop_SurfaceProperties
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE

    dst = os.path.join(OUT, nm + ".step")
    if os.path.exists(dst):
        return "cached"
    sp = os.path.join(ROOT, "steps", nm + ".step")
    if not os.path.exists(sp):          # CADSynth uses step/<id>.stp
        for _alt in (os.path.join(ROOT, "step", nm + ".stp"),
                     os.path.join(ROOT, "step", nm + ".step")):
            if os.path.exists(_alt):
                sp = _alt; break  # alt_paths
    if not os.path.exists(sp):
        return "missing"

    def props(sh):
        v = GProp_GProps(); brepgprop_VolumeProperties(sh, v)
        a = GProp_GProps(); brepgprop_SurfaceProperties(sh, a)
        return v.Mass(), a.Mass()

    def nf(sh):
        n = 0; ex = TopExp_Explorer(sh, TopAbs_FACE)
        while ex.More():
            n += 1; ex.Next()
        return n

    try:
        rd = STEPControl_Reader()
        if rd.ReadFile(sp) != IFSelect_RetDone:
            return "readfail"
        rd.TransferRoots()
        base = rd.OneShape()
        out = split_with(base, FAM)
        if out is None:
            return "nosplit"
        v0, a0 = props(base); v1, a1 = props(out)
        if v0 <= 0 or abs(v1 - v0) / v0 > 1e-9 or abs(a1 - a0) / a0 > 1e-9:
            return "geomchanged"
        if nf(out) == nf(base):
            return "nochange"
        w = STEPControl_Writer()
        w.Transfer(out, STEPControl_AsIs)
        return "ok" if w.Write(dst) == 1 else "writefail"
    except Exception as e:
        return f"err:{type(e).__name__}"


def main():
    names = [l.strip() for l in open(os.path.join(ROOT, f"{SPLIT}.txt")) if l.strip()]
    print(f"[rp fam={FAM} split={SPLIT}] {len(names)} parts -> {OUT}", flush=True)
    t0 = time.time(); stats = {}
    with Pool(34) as p:
        for i, r in enumerate(p.imap_unordered(one, names, chunksize=8)):
            stats[r] = stats.get(r, 0) + 1
    print(f"[rp fam={FAM} split={SPLIT}] done {time.time()-t0:.0f}s {stats}", flush=True)


if __name__ == "__main__":
    main()
