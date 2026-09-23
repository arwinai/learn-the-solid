"""Build a RE-PARTITIONED copy of the MFInstSeg test set.

This is the perturbation D2-equivariance provably cannot fix.  UV-Net (CVPR 2021,
Sec 4.5) already showed and fixed sensitivity to UV-grid re-parameterisation; a
changed FACE PARTITION is a different animal -- it changes the node set, the
adjacency, and the trim masks, so no per-face grid canonicalisation touches it.

Each test solid has some of its faces split with BRepFeat_SplitShape, which cuts
faces along curves lying ON them.  The solid is unchanged: we verify volume AND
area agree to 1e-9 (the BRepAlgoAPI_Splitter trap -- that one preserves volume
while silently adding internal walls and +25% area -- is avoided).

Labels transfer exactly: a split face's pieces inherit its label, and the
child->parent face map (recorded downstream by repart_extract.py) keeps
per-face mIoU comparable.

Reads <DS_ROOT>/test.txt and <DS_ROOT>/steps/<name>.step, writes
<DS_ROOT>/steps_repartitioned/<name>.step.  Runs in the geometry environment
(pythonocc).

Env: DS_ROOT  REPART_FRAC (number of section planes = round(2 * frac), default 1.0)
Usage: python repartition_testset.py
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
OUT_STEP = os.path.join(ROOT, "steps_repartitioned")
os.makedirs(OUT_STEP, exist_ok=True)
FRAC = float(os.environ.get("REPART_FRAC", "1.0"))


def split_faces(shape, frac, seed):
    """Split FACES along section curves lying on them (boundary preserved).

    Section edges lie exactly ON the faces they cross, so BRepFeat_SplitShape
    subdivides faces without moving any surface -- unlike BRepAlgoAPI_Splitter,
    which preserves volume but silently inserts internal walls (+25% area).
    `seed` is accepted for interface compatibility; the construction is
    deterministic.
    """
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section
    from OCC.Core.BRepFeat import BRepFeat_SplitShape
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from OCC.Core.TopoDS import topods, TopoDS_Face
    from OCC.Core.gp import gp_Pln, gp_Pnt, gp_Dir

    k = max(1, int(round(2 * frac)))
    b = Bnd_Box(); brepbndlib.Add(shape, b); xm, ym, zm, xM, yM, zM = b.Get()
    big = max(xM - xm, yM - ym, zM - zm) * 3
    cur = shape
    for i in range(k):
        f = (i + 1) / (k + 1)
        pln = (gp_Pln(gp_Pnt(xm + f * (xM - xm), 0, 0), gp_Dir(1, 0, 0)) if i % 2 == 0
               else gp_Pln(gp_Pnt(0, ym + f * (yM - ym), 0), gp_Dir(0, 1, 0)))
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
            # the edge MUST be attached to the face it actually lies on; the
            # Section algorithm records that ancestor.  Guessing the face (e.g.
            # "first one that doesn't raise") silently corrupts the solid
            # (volume changes of tens of percent).
            fc = TopoDS_Face()
            if not sec.HasAncestorFaceOn1(e, fc):
                continue
            try:
                sp.Add(e, topods.Face(fc)); added += 1
            except Exception:
                pass
            if added > 150:
                break
        if added:
            try:
                sp.Build()
                if sp.Shape() is not None:
                    cur = sp.Shape()
            except Exception:
                pass
    return cur if cur is not shape else None


def one(nm):
    from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.BRepGProp import brepgprop_VolumeProperties, brepgprop_SurfaceProperties
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE

    dst = os.path.join(OUT_STEP, nm + ".step")
    if os.path.exists(dst):
        return "cached"
    sp = os.path.join(ROOT, "steps", nm + ".step")
    if not os.path.exists(sp):
        return "missing"

    def props(sh):
        v = GProp_GProps(); brepgprop_VolumeProperties(sh, v)
        a = GProp_GProps(); brepgprop_SurfaceProperties(sh, a)
        return v.Mass(), a.Mass()

    def nfaces(sh):
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
        out = split_faces(base, FRAC, seed=abs(hash(nm)) % (2**31))
        if out is None:
            return "nosplit"
        v0, a0 = props(base); v1, a1 = props(out)
        if v0 <= 0 or abs(v1 - v0) / v0 > 1e-9 or abs(a1 - a0) / a0 > 1e-9:
            return "geomchanged"
        if nfaces(out) <= nfaces(base):
            return "nogain"
        w = STEPControl_Writer()
        w.Transfer(out, STEPControl_AsIs)
        if w.Write(dst) != 1:
            return "writefail"
        return "ok"
    except Exception as e:
        return f"err:{type(e).__name__}"


def main():
    names = [l.strip() for l in open(os.path.join(ROOT, "test.txt")) if l.strip()]
    print(f"[repartition] {len(names)} test parts, frac={FRAC}", flush=True)
    t0 = time.time(); stats = {}
    with Pool(36) as p:
        for i, r in enumerate(p.imap_unordered(one, names, chunksize=8)):
            stats[r] = stats.get(r, 0) + 1
            if (i + 1) % 2000 == 0:
                print(f"  {i+1}/{len(names)} {time.time()-t0:.0f}s {stats}", flush=True)
    print(f"[repartition] done {time.time()-t0:.0f}s {stats}", flush=True)


if __name__ == "__main__":
    main()
