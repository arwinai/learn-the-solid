"""Richer node features for the canonical region graph.

Per-face UV grids (as in AAGNet) carry far more information than a handful of
integrals, but they are tied to the parameterisation. These features follow
the same principle as the representation itself -- replace grid sampling of
the PARAMETER domain with INTEGRALS over the SURFACE:

    int n dA        orientation (for a plane this recovers area * normal exactly)
    int n(x)n dA    orientation spread (angular extent of a curved region)
    int H dA, int K dA, int H^2 dA    curvature content
    boundary length, sharp fraction, loop count
    third-order spatial moments, bbox extents

Every one is a sum over the faces of the region, so it is additive and cannot
depend on how the region was partitioned.  Integrals are evaluated on a fine
triangulation, so invariance holds to mesh tolerance rather than exactly -- the
residual is measured (region_check.py, graph_agreement.py) rather than assumed.
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
import numpy as np

from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepLProp import BRepLProp_SLProps
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_WIRE, TopAbs_REVERSED
from OCC.Core.TopoDS import topods

N_EXTRA = 17   # bbox extents dropped: even analytic AddOptimal bounds are
               # conservative for curved faces, and differently so for a face vs
               # its halves (residual 1e-4). Redundant with principal moments.
N_EXTRA_UNUSED = 18   # 3 + 6 + 3 + 3 + 3   (third moments dropped: midpoint
               # quadrature over triangles is mesh-dependent, so they were the
               # only feature that would break exact partition-invariance)


def face_integrals(face, deflection=0.0015):
    """Exact-ish surface integrals over ONE face, to be summed across a region.

    Returns dict with:
      n1   [3]  int n dA
      n2   [6]  int n_i n_j dA (upper triangle)
      curv [3]  int H dA, int K dA, int H^2 dA
      m3   [3]  int (x-0)^3 dA per axis (third spatial moment about origin)
      bbox [6]  min xyz / max xyz of the sampled region
      area [1]
    """
    out = {"n1": np.zeros(3), "n2": np.zeros(6), "curv": np.zeros(3),
           "m3": np.zeros(3), "lo": np.full(3, np.inf), "hi": np.full(3, -np.inf),
           "area": 0.0, "xx": np.zeros(6), "xn": np.zeros(9), "hx": np.zeros(3),
           "kx": np.zeros(3), "tri_pts": [], "tri_areas": [], "tri_verts": [],
           "tri_H": [], "tri_K": []}
    loc = TopLoc_Location()
    # ALWAYS remesh finely: whatever triangulation the STEP/kernel left behind has
    # face-dependent density, and midpoint quadrature error that differs between a
    # face and its two halves is exactly what destroys partition-invariance.
    from OCC.Core.BRepTools import breptools
    breptools.Clean(face)
    BRepMesh_IncrementalMesh(face, deflection, False, 0.1, True)
    tri = BRep_Tool.Triangulation(face, loc)
    if tri is None:
        return out
    trsf = loc.Transformation()
    has_uv = tri.HasUVNodes()
    surf = BRepAdaptor_Surface(face)
    rev = face.Orientation() == TopAbs_REVERSED

    nodes = []
    for i in range(1, tri.NbNodes() + 1):
        p = tri.Node(i).Transformed(trsf)
        nodes.append(np.array([p.X(), p.Y(), p.Z()]))
    nodes = np.array(nodes)
    if len(nodes) == 0:
        return out
    # exact tight analytic bounds; union over a region's faces reproduces the
    # region's box regardless of how it was split (triangulation nodes do not
    # reach the true extremes of a curved surface)
    try:
        from OCC.Core.Bnd import Bnd_Box
        from OCC.Core.BRepBndLib import brepbndlib
        bb = Bnd_Box(); bb.SetGap(0.0)
        brepbndlib.AddOptimal(face, bb, True, False)
        x0, y0, z0, x1, y1, z1 = bb.Get()
        out["lo"] = np.array([x0, y0, z0]); out["hi"] = np.array([x1, y1, z1])
    except Exception:
        out["lo"] = nodes.min(0); out["hi"] = nodes.max(0)

    for t in range(1, tri.NbTriangles() + 1):
        a, b, c = tri.Triangle(t).Get()
        p0, p1, p2 = nodes[a - 1], nodes[b - 1], nodes[c - 1]
        cross = np.cross(p1 - p0, p2 - p0)
        ar = 0.5 * float(np.linalg.norm(cross))
        if ar <= 0:
            continue
        cen = (p0 + p1 + p2) / 3.0
        # normal + curvature from the ANALYTIC surface at the triangle centroid
        n = None; H = K = 0.0
        if has_uv:
            uv = (np.array(tri.UVNode(a).Coord()) + np.array(tri.UVNode(b).Coord())
                  + np.array(tri.UVNode(c).Coord())) / 3.0
            try:
                pr = BRepLProp_SLProps(surf, uv[0], uv[1], 2, 1e-7)
                if pr.IsNormalDefined():
                    n = np.array(pr.Normal().Coord())
                    if pr.IsCurvatureDefined():
                        H = float(pr.MeanCurvature()); K = float(pr.GaussianCurvature())
            except Exception:
                n = None
        if n is None:
            n = cross / max(np.linalg.norm(cross), 1e-30)
        if rev:
            # MeanCurvature() is signed against the surface's NATURAL orientation,
            # so a reversed face must flip H along with the normal.  (K and H^2 are
            # sign-invariant; H is the only curvature feature that is not.)
            n = -n
            H = -H
        out["area"] += ar
        out["tri_pts"].append(cen); out["tri_areas"].append(ar)
        out["tri_verts"].append((p0, p1, p2))
        out["tri_H"].append(H); out["tri_K"].append(K)
        out["n1"] += ar * n
        out["n2"] += ar * np.array([n[0]*n[0], n[1]*n[1], n[2]*n[2],
                                    n[0]*n[1], n[0]*n[2], n[1]*n[2]])
        out["curv"] += ar * np.array([H, K, H * H])
        out["m3"] += ar * cen ** 3
        x = cen
        out["xx"] += ar * np.array([x[0]*x[0], x[1]*x[1], x[2]*x[2],
                                    x[0]*x[1], x[0]*x[2], x[1]*x[2]])
        out["xn"] += ar * np.outer(x, n).ravel()
        out["hx"] += ar * H * x
        out["kx"] += ar * K * x
    return out


def boundary_feats(face):
    """[n_wires, total wire length placeholder] -- loop count is canonical."""
    n_w = 0
    ex = TopExp_Explorer(face, TopAbs_WIRE)
    while ex.More():
        n_w += 1; ex.Next()
    return n_w


def pack(acc):
    """dict of accumulated integrals -> flat feature vector [N_EXTRA]."""
    v = np.zeros(N_EXTRA, dtype=np.float32)
    a = max(acc["area"], 1e-12)
    v[0:3] = acc["n1"] / a                    # mean normal (area-normalised)
    v[3:9] = acc["n2"] / a                    # orientation spread
    v[9:12] = acc["curv"] / a                 # mean H, K, H^2
    v[12] = acc["n_loops"]
    v[13] = acc["blen"]
    v[14] = acc["bsharp"]
    v[15] = acc.get("bconcave", 0.0) / max(acc["blen"], 1e-12)
    v[16] = acc.get("bconvex", 0.0) / max(acc["blen"], 1e-12)
    return v


def new_acc():
    return {"n1": np.zeros(3), "n2": np.zeros(6), "curv": np.zeros(3),
            "m3": np.zeros(3), "lo": np.full(3, np.inf), "hi": np.full(3, -np.inf),
            "area": 0.0, "n_loops": 0.0, "blen": 0.0, "bsharp": 0.0,
            "bconcave": 0.0, "bconvex": 0.0,
            "xx": np.zeros(6), "xn": np.zeros(9), "hx": np.zeros(3),
            "kx": np.zeros(3), "tri_pts": [], "tri_areas": [], "tri_verts": [],
            "tri_H": [], "tri_K": []}


def add(acc, f):
    acc["n1"] += f["n1"]; acc["n2"] += f["n2"]; acc["curv"] += f["curv"]
    acc["m3"] += f["m3"]; acc["area"] += f["area"]
    acc["xx"] += f["xx"]; acc["xn"] += f["xn"]
    acc["hx"] += f["hx"]; acc["kx"] += f["kx"]
    acc["tri_pts"] += f["tri_pts"]; acc["tri_areas"] += f["tri_areas"]
    acc["tri_verts"] += f["tri_verts"]
    acc["tri_H"] += f["tri_H"]; acc["tri_K"] += f["tri_K"]
    acc["lo"] = np.minimum(acc["lo"], f["lo"])
    acc["hi"] = np.maximum(acc["hi"], f["hi"])
    return acc
