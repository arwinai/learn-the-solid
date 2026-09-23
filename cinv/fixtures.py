"""Construction-variant fixtures: the same solid built two different ways.

Each `*_pair` function returns two `TopoDS_Shape`s that bound the SAME solid
(identical volume and area to 1e-9) but whose boundaries are partitioned
differently -- e.g. a ring made as one revolved profile vs. as a cylinder minus
a cylinder, or an L-profile extruded whole vs. as two fused blocks. They need no
dataset and are used by the verification scripts (region_check.py,
fixture_agreement.py, surface_graph.py's self-test) and by `main()` below,
which additionally runs AAGNet's own STEP->graph extractor on every variant.

Usage: python fixtures.py            # writes aag_variants.pkl next to this file
"""
from __future__ import annotations
import os, sys, pickle, json, tempfile, warnings
warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths

AAG = paths.AAGNET

from OCC.Core.BRepPrimAPI import (BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder,
                                  BRepPrimAPI_MakePrism, BRepPrimAPI_MakeRevol)
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCC.Core.BRepBuilderAPI import (BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire,
                                     BRepBuilderAPI_MakeFace)
from OCC.Core.BRepGProp import brepgprop_VolumeProperties, brepgprop_SurfaceProperties
from OCC.Core.GProp import GProp_GProps
from OCC.Core.gp import gp_Ax1, gp_Ax2, gp_Circ, gp_Dir, gp_Pnt, gp_Vec, gp_Pln
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs


# ---------------------------------------------------------------- shape builders
def props(shape):
    v = GProp_GProps(); brepgprop_VolumeProperties(shape, v)
    a = GProp_GProps(); brepgprop_SurfaceProperties(shape, a)
    return v.Mass(), a.Mass()


def _wire(pts):
    mw = BRepBuilderAPI_MakeWire()
    for a, b in zip(pts, pts[1:] + [pts[0]]):
        if np.allclose(a, b):
            continue
        mw.Add(BRepBuilderAPI_MakeEdge(gp_Pnt(*a), gp_Pnt(*b)).Edge())
    return mw.Wire()


def ring_pair(R, r, h):
    """A: cut(cyl, cyl).   B: revolve(rectangle).   Same solid."""
    a = BRepAlgoAPI_Cut(
        BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), R, h).Shape(),
        BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, -1), gp_Dir(0, 0, 1)), r, h + 2).Shape()).Shape()
    f = BRepBuilderAPI_MakeFace(_wire([(r, 0, 0), (R, 0, 0), (R, 0, h), (r, 0, h)])).Face()
    b = BRepPrimAPI_MakeRevol(f, gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))).Shape()
    return a, b


def slab_hole_pair(w, d_, h, r):
    """A: box - cyl.   B: extrude(face with hole in it).   Same solid."""
    a = BRepAlgoAPI_Cut(
        BRepPrimAPI_MakeBox(gp_Pnt(-w / 2, -d_ / 2, 0), w, d_, h).Shape(),
        BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, -1), gp_Dir(0, 0, 1)), r, h + 2).Shape()).Shape()
    outer = _wire([(-w / 2, -d_ / 2, 0), (w / 2, -d_ / 2, 0), (w / 2, d_ / 2, 0), (-w / 2, d_ / 2, 0)])
    mf = BRepBuilderAPI_MakeFace(gp_Pln(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), outer)
    hw = BRepBuilderAPI_MakeWire(BRepBuilderAPI_MakeEdge(
        gp_Circ(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), r)).Edge()).Wire()
    hw.Reverse(); mf.Add(hw)
    b = BRepPrimAPI_MakePrism(mf.Face(), gp_Vec(0, 0, h)).Shape()
    return a, b


def lprofile_pair(a_, b_, t, h):
    """A: fuse(box, box).   B: extrude(L-shaped wire).   Same solid."""
    A = BRepAlgoAPI_Fuse(BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), a_, t, h).Shape(),
                         BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), t, b_, h).Shape()).Shape()
    prof = _wire([(0, 0, 0), (a_, 0, 0), (a_, t, 0), (t, t, 0), (t, b_, 0), (0, b_, 0)])
    B = BRepPrimAPI_MakePrism(BRepBuilderAPI_MakeFace(prof).Face(), gp_Vec(0, 0, h)).Shape()
    return A, B


def block_with_hole(hole_x=0.0, r=0.25):
    s = BRepPrimAPI_MakeBox(gp_Pnt(-1, -0.6, 0), 2.0, 1.2, 0.5).Shape()
    t = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(hole_x, 0, -0.5), gp_Dir(0, 0, 1)), r, 2).Shape()
    return BRepAlgoAPI_Cut(s, t).Shape()


# ---------------------------------------------------------------- extraction
_TMP = tempfile.mkdtemp(prefix="aagstep_")


def aag(shape, tag):
    """Write shape to STEP, run AAGNet's extractor on it (needs CINV_AAGNET)."""
    sys.path.insert(0, AAG); sys.path.insert(0, os.path.join(AAG, "dataset"))
    from dataset.AAGExtractor import AAGExtractor
    schema = json.load(open(os.path.join(AAG, "feature_lists/all.json")))
    path = os.path.join(_TMP, f"{tag}.step")
    w = STEPControl_Writer()
    w.Transfer(shape, STEPControl_AsIs)
    if w.Write(path) != 1:
        return None
    try:
        return AAGExtractor(path, schema).process()
    except Exception as e:
        print(f"    [extract failed {tag}] {e}")
        return None


def main():
    out = {"pairs": [], "singles": {}}
    rng = np.random.default_rng(0)
    n_ok = 0
    for i in range(8):
        R = rng.uniform(0.8, 1.4); r = rng.uniform(0.25, 0.55); h = rng.uniform(0.3, 0.9)
        fams = [("ring", ring_pair(R, r, h)),
                ("slab_hole", slab_hole_pair(2 * R, 1.6 * R, h, r)),
                ("Lprofile", lprofile_pair(1.5 * R, 1.2 * R, 0.3 * R, h))]
        for name, (A, B) in fams:
            va, aa = props(A); vb, ab = props(B)
            # geometric identity is the whole premise -- enforce it hard
            if va <= 0 or abs(va - vb) / va > 1e-9 or abs(aa - ab) / aa > 1e-9:
                print(f"  [skip] {name}{i}: vol {va:.6f}/{vb:.6f} area {aa:.6f}/{ab:.6f}")
                continue
            ga = aag(A, f"{name}{i}_A"); gb = aag(B, f"{name}{i}_B")
            if ga is None or gb is None:
                continue
            out["pairs"].append({"family": name, "idx": i, "A": ga, "B": gb,
                                 "nfA": ga["graph"]["num_nodes"], "nfB": gb["graph"]["num_nodes"]})
            n_ok += 1
            print(f"  {name}{i}: faces {ga['graph']['num_nodes']}/{gb['graph']['num_nodes']} "
                  f"vol {va:.6f} area {aa:.6f}  OK", flush=True)

    # R3 reference shapes: real functional changes on one base part
    refs = {"base": block_with_hole(),
            "hole_moved_0.45": block_with_hole(hole_x=0.45),
            "hole_moved_0.20": block_with_hole(hole_x=0.20),
            "hole_r_0.30": block_with_hole(r=0.30),
            "hole_r_0.35": block_with_hole(r=0.35),
            "a_ring": ring_pair(1.0, 0.4, 0.5)[0]}
    for k, sh in refs.items():
        g = aag(sh, k)
        if g is not None:
            out["singles"][k] = g
            print(f"  ref {k}: faces {g['graph']['num_nodes']}", flush=True)

    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aag_variants.pkl")
    with open(dst, "wb") as f:
        pickle.dump(out, f)
    print(f"\n[fixtures] {n_ok} valid pairs + {len(out['singles'])} refs -> {dst}")


if __name__ == "__main__":
    main()
