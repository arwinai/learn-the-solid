"""The proposed input representation: the CANONICAL REGION GRAPH.

Node = a maximal CONNECTED region of a single underlying analytic surface.

Two faces belong to the same node iff they lie on the same analytic surface AND
are edge-connected.  This is determined entirely by the solid's point set, so it
is invariant to how a kernel chose to partition the boundary -- but unlike
whole-surface grouping it does NOT fuse two disjoint pockets machined to the
same depth, so per-face labels survive.

Because the region is itself canonical, ANY functional of it is invariant -- we
are not restricted to additive integrals.  Node features therefore include area,
area-weighted centroid, second moments (region shape), boundary length, loop
count, surface type and intrinsic parameters.  Edge features describe the shared
boundary between two regions: total arc length, dihedral angle, convexity.
"""
from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from surface_graph import surface_key, area_of, centroid_of, TYPES, RECOG_AUX
import region_feats as RF2

from OCC.Core.TopExp import TopExp_Explorer, topexp
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.TopoDS import topods
from OCC.Core.TopTools import (TopTools_IndexedDataMapOfShapeListOfShape,
                               TopTools_ListIteratorOfListOfShape,
                               TopTools_IndexedMapOfShape)
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, \
    GeomAbs_Sphere, GeomAbs_Torus
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_LinearProperties, brepgprop_SurfaceProperties
from OCC.Core.GCPnts import GCPnts_AbscissaPoint
from OCC.Core.BRepLProp import BRepLProp_SLProps
from OCC.Core.BRep import BRep_Tool

SURF_NAMES = ["plane", "cyl", "cone", "sphere", "torus", "other"]


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]; a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def face_list(shape):
    fs, ex = [], TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        fs.append(topods.Face(ex.Current())); ex.Next()
    return fs


def edge_face_map(shape):
    m = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, m)
    return m


def edge_length(e):
    g = GProp_GProps(); brepgprop_LinearProperties(e, g)
    return g.Mass()


def dihedral(face_a, face_b, edge):
    """Angle between the two surface normals at the middle of the shared edge."""
    try:
        c = BRepAdaptor_Curve(edge)
        t = 0.5 * (c.FirstParameter() + c.LastParameter())
        p = c.Value(t)
        ns = []
        for f in (face_a, face_b):
            s = BRepAdaptor_Surface(f)
            from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface
            sas = ShapeAnalysis_Surface(BRep_Tool.Surface(f))
            uv = sas.ValueOfUV(p, 1e-6)
            pr = BRepLProp_SLProps(s, uv.X(), uv.Y(), 1, 1e-6)
            if not pr.IsNormalDefined():
                return 0.0
            n = np.array(pr.Normal().Coord())
            if f.Orientation() == 1:                  # TopAbs_REVERSED
                n = -n
            ns.append(n)
        d = float(np.clip(np.dot(ns[0], ns[1]), -1, 1))
        ang = float(np.degrees(np.arccos(d)))
        # SIGN via occwl's EdgeDataExtractor -- the same convexity computation
        # AAGNet's own dataset pipeline uses. (First attempt probed a point
        # offset along the mean normal with a solid classifier; that direction
        # points into the void for BOTH edge types, so it cannot discriminate.)
        try:
            from occwl.edge_data_extractor import EdgeDataExtractor, EdgeConvexity
            from occwl.edge import Edge as _OE
            from occwl.face import Face as _OF
            ed = EdgeDataExtractor(_OE(edge), [_OF(face_a), _OF(face_b)],
                                   num_samples=5, use_arclength_params=False)
            if ed.good:
                cv = ed.edge_convexity(angle_tol_rads=np.radians(1.0))
                if cv == EdgeConvexity.CONCAVE:
                    ang = -ang
        except Exception:
            pass
        return ang
    except Exception:
        return 0.0


_SOLID_FOR_SIGN = [None]


def build(shape, sharp_deg=1.0):
    """-> (face_to_region, node_feats [R,F], edges (src,dst), edge_feats [E,K])."""
    # ISO_SCALE=1: rotation-INVARIANT global scale. Upstream extractors apply
    # AAGNet's scale_solid_to_unit_box, whose axis-aligned bounding box makes
    # the scale factor rotation-DEPENDENT. Rescale here so the total boundary
    # area is exactly 1: the net scale becomes 1/sqrt(original area) no matter
    # what the AABB step did, since a prior uniform scale s multiplies area by
    # s^2 and is exactly cancelled by 1/sqrt(A).
    if os.environ.get("ISO_SCALE"):
        from OCC.Core.gp import gp_Trsf
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
        _g = GProp_GProps()
        if os.environ.get("CONE_INTRINSIC"):
            # FACE-WISE HYBRID area. GProp's default quadrature has ~1e-3
            # relative area error on NURBS faces, so a solid and its
            # NurbsConvert twin get slightly different global scales -- a
            # ~1e-4 mismatch in every length column that would bury the
            # <=1e-6 intrinsic cone-radius parity this flag exists to
            # provide. A spline face that RECOGNIZES as analytic (a kernel
            # re-expression of an exact quadric -- the only way NurbsConvert
            # noise enters) therefore integrates adaptively (Eps=1e-9).
            # Everything else keeps the default quadrature: analytic faces
            # accumulate exactly like the whole-shape call (all-analytic
            # parts keep their pre-flag scale bit-identically), and genuine
            # freeform faces keep their old (imprecise but PAIRED -- the
            # same NURBS on both legs) default value, so clean freeform-
            # bearing parts are untouched too. (Broader rules measured:
            # whole-shape Eps shifts plain analytic parts by up to ~1e-4,
            # all-spline-adaptive shifts 35/300 clean f360 parts.)
            _ANA = (GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
                    GeomAbs_Sphere, GeomAbs_Torus)
            _QUAD = ("plane", "cyl", "cone", "sphere", "torus")
            for _f in face_list(shape):
                _gf = GProp_GProps()
                try:
                    if (BRepAdaptor_Surface(_f).GetType() not in _ANA
                            and surface_key(_f)[0] in _QUAD):
                        from OCC.Core.BRepGProp import brepgprop as _bg
                        _bg.SurfaceProperties(_f, _gf, 1e-9)
                    else:
                        brepgprop_SurfaceProperties(_f, _gf)
                except Exception:
                    brepgprop_SurfaceProperties(_f, _gf)
                _g.Add(_gf)
        else:
            brepgprop_SurfaceProperties(shape, _g)
        _A = _g.Mass()
        if _A > 0:
            _tr = gp_Trsf()
            _tr.SetScaleFactor(1.0 / float(np.sqrt(_A)))
            shape = BRepBuilderAPI_Transform(shape, _tr, True).Shape()
    _SOLID_FOR_SIGN[0] = shape
    faces = face_list(shape)
    nf = len(faces)
    if nf == 0:
        return None
    keys = [surface_key(f) for f in faces]
    dsu = DSU(nf)
    idx = {f.__hash__(): i for i, f in enumerate(faces)}

    m = edge_face_map(shape)
    shared = []                                        # (i, j, edge, length, dihedral)
    for k in range(1, m.Size() + 1):
        e = topods.Edge(m.FindKey(k))
        lst = m.FindFromIndex(k)
        it = TopTools_ListIteratorOfListOfShape(lst)
        ff = []
        while it.More():
            ff.append(idx.get(it.Value().__hash__())); it.Next()
        ff = [x for x in ff if x is not None]
        if len(ff) != 2:
            continue
        i, j = ff[0], ff[1]
        ang = dihedral(faces[i], faces[j], e)
        v0 = topexp.FirstVertex(e, True).__hash__()
        v1 = topexp.LastVertex(e, True).__hash__()
        shared.append((i, j, edge_length(e), ang, v0, v1))
        # same analytic surface AND edge-connected -> one region
        if keys[i] == keys[j] and not os.environ.get("NO_MERGE"):
            dsu.union(i, j)

    roots = {}
    f2r = np.empty(nf, dtype=np.int64)
    for i in range(nf):
        r = dsu.find(i)
        if r not in roots:
            roots[r] = len(roots)
        f2r[i] = roots[r]
    R = len(roots)

    # ---- node features (all functionals of the canonical region) ----
    # Accumulate true surface integrals per region with GProp_GProps.Add, which
    # combines mass, centroid AND inertia correctly about a common origin.  This
    # is additive over faces, so it cannot depend on how the region was split.
    feats = np.zeros((R, 6 + 1 + 3 + 3 + 4 + 3), dtype=np.float32)
    gp_acc = [None] * R
    # CONE_INTRINSIC: a parallel HIGH-PRECISION accumulator over cone-region
    # faces only. The intrinsic cone radius r_c projects the region centroid
    # onto the cone axis, and GProp's default quadrature has ~1e-3 relative
    # error on (trimmed, split) NURBS faces whose axial component would leak
    # into r_c (measured 6e-6 on a split+NurbsConvert boolean fixture). The
    # Eps overload is adaptive; it is additive over faces exactly like the
    # default one, so partition invariance is untouched. Only feats[:,13] of
    # cone regions reads this accumulator -- every other column still comes
    # from gp_acc, bit-identical to the flag being off.
    CONE_INTRINSIC = bool(os.environ.get("CONE_INTRINSIC"))
    cone_hp = [None] * R
    for i, f in enumerate(faces):
        r = f2r[i]
        g = GProp_GProps()
        brepgprop_SurfaceProperties(f, g)
        if gp_acc[r] is None:
            gp_acc[r] = g
        else:
            gp_acc[r].Add(g)
        if CONE_INTRINSIC and keys[i][0] == "cone":
            gh = GProp_GProps()
            try:
                from OCC.Core.BRepGProp import brepgprop as _bg
                _bg.SurfaceProperties(f, gh, 1e-10)
            except Exception:
                brepgprop_SurfaceProperties(f, gh)
            if cone_hp[r] is None:
                cone_hp[r] = gh
            else:
                cone_hp[r].Add(gh)
    area = np.array([gp_acc[r].Mass() for r in range(R)])
    cen = np.array([list(gp_acc[r].CentreOfMass().Coord()) for r in range(R)])
    cen_cone = {r: np.array(list(cone_hp[r].CentreOfMass().Coord()))
                for r in range(R) if cone_hp[r] is not None}
    for i, f in enumerate(faces):
        r = f2r[i]
        k = keys[i]
        t = SURF_NAMES.index(k[0]) if k[0] in SURF_NAMES else 5
        feats[r, t] = 1.0
        feats[r, 6] = area[r]
        feats[r, 7:10] = cen[r]
        s = BRepAdaptor_Surface(f); ty = s.GetType()
        # sign-free axis normally; under PART_FRAME keep the raw axis and
        # apply abs() after rotating into the solid's canonical frame
        _ax = (lambda v: np.asarray(v)) if os.environ.get("PART_FRAME") \
            else (lambda v: np.abs(v))
        if ty == GeomAbs_Cylinder:
            cy = s.Cylinder()
            feats[r, 10:13] = _ax(cy.Axis().Direction().Coord())
            feats[r, 13] = cy.Radius()
        elif ty == GeomAbs_Cone:
            co = s.Cone()
            feats[r, 10:13] = _ax(co.Axis().Direction().Coord())
            # SA_ABS=1: |semi-angle|. OCC's sign is tied to the axis-direction
            # convention -- construction data NurbsConvert destroys and
            # recognition cannot reliably reconstruct (2.3% of converted parts
            # flipped this channel by up to ~1.3 rad). The sign carries no
            # geometry not already in the frame-expressed axis features.
            feats[r, 14] = abs(co.SemiAngle()) if os.environ.get("SA_ABS") \
                else co.SemiAngle()
            if os.environ.get("CONE_INTRINSIC"):
                # INTRINSIC local radius at the REGION centroid's axis station.
                # OCC's RefRadius is anchored at the surface's parametric
                # origin -- construction data NurbsConvert destroys, so a
                # recognized cone cannot reproduce it. With apex q, unit axis
                # a, semi-angle alpha and the region's accumulated area
                # centroid c (a point-set property), r_c = |tan(alpha) *
                # ((c - q) . a)| depends only on the surface's point set and
                # the region -- and it is the same for every face of the
                # region, so setting it per face is per-region.
                q = np.asarray(co.Apex().Coord(), dtype=float)
                a_cn = np.asarray(co.Axis().Direction().Coord(), dtype=float)
                c_cn = cen_cone.get(r, cen[r])
                feats[r, 13] = abs(np.tan(float(co.SemiAngle()))
                                   * float(np.dot(c_cn - q, a_cn)))
            else:
                feats[r, 13] = co.RefRadius()
        elif ty == GeomAbs_Sphere:
            feats[r, 13] = s.Sphere().Radius()
        elif ty == GeomAbs_Torus:
            to = s.Torus()
            feats[r, 10:13] = _ax(to.Axis().Direction().Coord())
            feats[r, 13] = to.MajorRadius(); feats[r, 15] = to.MinorRadius()
        elif ty == GeomAbs_Plane:
            feats[r, 10:13] = _ax(s.Plane().Axis().Direction().Coord())
        elif k[0] in ("plane", "cyl", "sphere", "cone", "torus"):
            # surface recognized as analytic (RECOG) but adaptor still reports
            # a spline: fill axis/radius from the canonical key so features
            # match a natively-analytic build
            if k[0] == "plane":
                feats[r, 10:13] = _ax(np.asarray(k[1:4], dtype=float))
            elif k[0] == "cyl":
                feats[r, 13] = float(k[1])
                feats[r, 10:13] = _ax(np.asarray(k[2:5], dtype=float))
            elif k[0] == "sphere":
                feats[r, 13] = float(k[1])
            elif k[0] == "cone":
                # native layout: (name, refR, semiAngle, dir3, loc3)
                # SA_ABS: |semi-angle| here too (see native branch)
                feats[r, 14] = abs(float(k[2])) if os.environ.get("SA_ABS") \
                    else float(k[2])
                feats[r, 10:13] = _ax(np.asarray(k[3:6], dtype=float))
                if os.environ.get("CONE_INTRINSIC"):
                    aux = RECOG_AUX.get(k)
                    if aux is not None:
                        # unrounded fitted apex/axis/semi-angle (see native
                        # branch for the r_c definition)
                        q_cn = np.asarray(aux[0], dtype=float)
                        a_cn = np.asarray(aux[1], dtype=float)
                        c_cn = cen_cone.get(r, cen[r])
                        feats[r, 13] = abs(np.tan(float(aux[2]))
                                           * float(np.dot(c_cn - q_cn, a_cn)))
                    else:
                        # fallback from the (rounded) key: the signed radius
                        # along the key axis is R(z) = refR + tan(sa) * z with
                        # z measured from the key anchor loc, and |R| is the
                        # true local radius under _cone_key's sign convention
                        d_cn = np.asarray(k[3:6], dtype=float)
                        p_cn = np.asarray(k[6:9], dtype=float)
                        c_cn = cen_cone.get(r, cen[r])
                        feats[r, 13] = abs(float(k[1]) + np.tan(float(k[2]))
                                           * float(np.dot(c_cn - p_cn, d_cn)))
                else:
                    feats[r, 13] = float(k[1])
            elif k[0] == "torus":
                # native layout: (name, majorR, minorR, center3) -- the key
                # holds no axis, the fit's axis is kept in RECOG_AUX
                feats[r, 13] = float(k[1]); feats[r, 15] = float(k[2])
                d_ax = RECOG_AUX.get(k)
                if d_ax is not None:
                    feats[r, 10:13] = _ax(np.asarray(d_ax, dtype=float))

    # ---- region shape: principal moments of inertia about the region centroid.
    #      Taken from the ACCUMULATED GProp, so they are exact surface integrals
    #      over the whole region and independent of any face subdivision.
    for r in range(R):
        try:
            pp = gp_acc[r].PrincipalProperties()
            m = np.sort(np.array(list(pp.Moments())))
            feats[r, 16:19] = m
        except Exception:
            pass

    # ---- edges between DISTINCT regions ----
    agg = {}
    EHIST = bool(os.environ.get("EHIST"))
    NEB = 6
    for i, j, L, ang, _v0, _v1 in shared:
        ri, rj = int(f2r[i]), int(f2r[j])
        if ri == rj:
            continue
        key = (min(ri, rj), max(ri, rj))
        a = agg.setdefault(key, [0.0] * (4 + (NEB if EHIST else 0)))
        a[0] += L                       # total shared arc length (additive -> canonical)
        a[1] += L * abs(ang)            # length-weighted dihedral magnitude
        a[2] += L if abs(ang) > sharp_deg else 0.0
        a[3] += L if ang < -sharp_deg else 0.0     # CONCAVE share (cut-like)
        if EHIST:
            # per-boundary SIGNED dihedral histogram, soft-binned (same knife-
            # edge argument as the node-level BHIST: +-90deg is ubiquitous)
            t = float(np.clip((ang + 180.0) / 360.0 * NEB - 0.5, 0.0, NEB - 1.0))
            i0 = int(t); w1 = t - i0; i1 = min(i0 + 1, NEB - 1)
            a[4 + i0] += (1.0 - w1) * L
            a[4 + i1] += w1 * L
    # ---- richer integral features + region-boundary features ----
    accs = [RF2.new_acc() for _ in range(R)]
    for i, f in enumerate(faces):
        RF2.add(accs[f2r[i]], RF2.face_integrals(f))
    # region boundary = edges whose two faces fall in DIFFERENT regions
    bd = {r: [] for r in range(R)}
    for i, j, L, ang, v0, v1 in shared:
        ri, rj = int(f2r[i]), int(f2r[j])
        if ri == rj:
            continue
        for r in (ri, rj):
            bd[r].append((L, ang, v0, v1))
    for r in range(R):
        accs[r]["blen"] = sum(x[0] for x in bd[r])
        accs[r]["bsharp"] = sum(x[0] for x in bd[r] if abs(x[1]) > sharp_deg)
        accs[r]["bconcave"] = sum(x[0] for x in bd[r] if x[1] < -sharp_deg)
        accs[r]["bconvex"] = sum(x[0] for x in bd[r] if x[1] > sharp_deg)
        # loop count = connected components of the boundary edge set (union-find
        # on shared vertices) -- a property of the region, not of any face split
        vs = {}
        for _, _, v0, v1 in bd[r]:
            for v in (v0, v1):
                if v not in vs:
                    vs[v] = len(vs)
        if vs:
            d2 = DSU(len(vs))
            for _, _, v0, v1 in bd[r]:
                d2.union(vs[v0], vs[v1])
            accs[r]["n_loops"] = len({d2.find(k) for k in range(len(vs))})
    extra = np.stack([RF2.pack(accs[r]) for r in range(R)]).astype(np.float32)

    # exact second-moment tensor int x(x)x dA, recovered from the ACCUMULATED
    # GProp inertia matrix (analytic, additive) rather than from triangle
    # midpoint quadrature -- quadrature versions of this are mesh-dependent and
    # silently destroy partition-invariance.
    xx = np.zeros((R, 6), dtype=np.float32)
    for r in range(R):
        try:
            M = gp_acc[r].MatrixOfInertia()
            I = np.array([[M.Value(i + 1, j + 1) for j in range(3)] for i in range(3)])
            s2 = np.trace(I) / 2.0
            T = np.eye(3) * s2 - I
            a_r = max(area[r], 1e-12)
            xx[r] = np.array([T[0, 0], T[1, 1], T[2, 2],
                              T[0, 1], T[0, 2], T[1, 2]]) / a_r
        except Exception:
            pass
    feats = np.concatenate([feats, extra, xx], axis=1)

    if os.environ.get("PART_FRAME"):
        # Express the frame-dependent feature columns in the SOLID's own
        # inertia frame instead of the file's. The whole-solid inertia is a
        # boundary integral, hence partition-invariant, so this preserves
        # every invariance proof; and it removes the file-frame (rotation)
        # dependence that costs 23 mIoU under random SO(3). Ambiguities are
        # handled exactly like the region grid's: axis signs are blended by
        # skew weight (Lipschitz), and within near-degenerate eigen-subspaces
        # features are rotation-averaged (vectors' ambiguous components -> 0,
        # tensors -> their subspace-rotational mean), which IS the invariant.
        A_tot = float(area.sum())
        C = (area[:, None] * cen).sum(0) / max(A_tot, 1e-12)
        # GProp's MatrixOfInertia is expressed about EACH accumulated
        # system's own center of mass, so compose regions to the shared
        # centroid C with per-region parallel-axis terms
        # I_C = sum_r [ I_r + A_r (|d_r|^2 E - d_r d_r^T) ], d_r = c_r - C.
        Ic = np.zeros((3, 3))
        for r in range(R):
            M = gp_acc[r].MatrixOfInertia()
            I_r = np.array([[M.Value(i2 + 1, j2 + 1) for j2 in range(3)]
                            for i2 in range(3)])
            d_r = cen[r] - C
            Ic += I_r + area[r] * (np.dot(d_r, d_r) * np.eye(3)
                                   - np.outer(d_r, d_r))
        w_eig, V = np.linalg.eigh(0.5 * (Ic + Ic.T))

        # skew along each axis (for sign weights). Centroid quadrature is
        # exact only for LINEAR integrands, and the mesher emits two giant
        # triangles on large planes -- the resulting bias in int x^3 dA can flip
        # w_sign between a face and its split halves (the same issue as in the
        # canonical grid). x is linear over a flat triangle, so int x^2 and int x^3
        # have exact closed forms from the vertex values: no subdivision.
        TV = np.concatenate([np.asarray(accs[r]["tri_verts"], dtype=np.float64)
                             for r in range(R) if len(accs[r]["tri_verts"])])
        TA = np.concatenate([np.asarray(accs[r]["tri_areas"]).reshape(-1)
                             for r in range(R) if len(accs[r]["tri_areas"])])
        LV = np.einsum("tvi,ia->tva", TV - C, V)      # [T,3verts,3axes]
        # exact: int l^2 = (A/6)(sum li^2 + sum_{i<j} li lj)
        s2 = (LV ** 2).sum(1) + (LV[:, 0] * LV[:, 1] + LV[:, 0] * LV[:, 2]
                                 + LV[:, 1] * LV[:, 2])
        m2 = (TA[:, None] / 6.0 * s2).sum(0)
        rms_g = np.sqrt(np.maximum(m2 / max(A_tot, 1e-12), 1e-18))
        # exact: int l^3 = (A/10)(sum li^3 + sum_{i!=j} li^2 lj + l1 l2 l3)
        sum_l = LV.sum(1)
        sum_l2 = (LV ** 2).sum(1)
        sum_l3 = (LV ** 3).sum(1)
        cross = (LV[:, 0] * LV[:, 1] * LV[:, 2])
        s3 = sum_l3 + (sum_l2 * sum_l - sum_l3) + cross
        m3 = (TA[:, None] / 10.0 * s3).sum(0)
        TOL_LO, TOL_HI = 0.01, 0.04
        w_sign = np.ones(3)
        sn_abs = np.zeros(3)
        for a in range(3):
            sn = float(m3[a]) / (max(A_tot, 1e-12) * rms_g[a] ** 3)
            sn_abs[a] = abs(sn)
            w_sign[a] = float(np.clip((abs(sn) - TOL_LO) / (TOL_HI - TOL_LO), 0, 1))
            if m3[a] < 0:
                V[:, a] = -V[:, a]
        # Enforce a RIGHT-HANDED frame: per-axis skew sign
        # fixes alone leave det(V) = +-1, and a mirror image then produces the
        # SAME frame hence identical features. Restore det=+1 by negating the
        # EFFECTIVE sign-weight of the least skew-confident axis: because the
        # feature blend below enumerates both signs of every axis with weights
        # (1 +- w_sign)/2, negating a column of V is exactly equivalent to
        # negating that axis's w_sign, so this reproduces the hard flip when
        # the minimum is unique. Under proper rotations skew stats are
        # invariant so nothing changes; under reflection the weighting becomes
        # asymmetric on a different axis than the mirror plane, so chiral
        # pairs differ whenever skews are confident and untied.
        # Tie handling: a hard argmin switches abruptly when two skew
        # magnitudes are near-equal, so the det-fix is DISTRIBUTED over
        # near-tied axes. The tie WINDOW shrinks with the smallest skew,
        # W = min(s_min, BAND): the det = +-1 transition happens exactly
        # where some skew crosses zero (s_min -> 0), and there W -> 0
        # concentrates the fix entirely on the crossing axis, whose w_sign
        # is 0 -- so the det branch matches the det-positive side EXACTLY
        # at the transition (a fixed window would leave a bounded jump).
        # At an exact tie with confident skews the tied axes share the fix
        # symmetrically and chirality discrimination degrades continuously
        # to zero -- the same regime as the blend band. No hard guarantee
        # is claimed at ties or degeneracies.
        if np.linalg.det(V) < 0:
            BAND, DELTA = 0.03, 1e-4
            s_min = float(sn_abs.min())
            W = max(min(s_min, BAND), 1e-30)
            t = np.clip(1.0 - (sn_abs - s_min) / W, 0.0, 1.0) \
                / (sn_abs + DELTA)
            t = t / t.sum()
            w_sign = (1.0 - 2.0 * t) * w_sign

        # degeneracy weights per adjacent pair (0=distinct, 1=fully degenerate)
        scale = max(abs(w_eig[2]), 1e-12)
        d01 = float(np.clip(1.0 - (abs(w_eig[1] - w_eig[0]) / scale - 0.01) / 0.04, 0, 1))
        d12 = float(np.clip(1.0 - (abs(w_eig[2] - w_eig[1]) / scale - 0.01) / 0.04, 0, 1))

        if os.environ.get("PF_DEBUG"):
            print(f"[pf] A={A_tot:.6f} C={np.round(C,6)} eig={np.round(w_eig,6)} "
                  f"d01={d01:.3f} d12={d12:.3f} w_sign={np.round(w_sign,4)} "
                  f"sn={np.round(sn_abs,5)} det={int(np.sign(np.linalg.det(V)))}",
                  flush=True)

        S6 = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]

        def sym_to_mat(v):
            m = np.zeros(v.shape[:-1] + (3, 3))
            for k, (i2, j2) in enumerate(S6):
                m[..., i2, j2] = v[..., k]; m[..., j2, i2] = v[..., k]
            return m

        def mat_to_sym(m):
            return np.stack([m[..., i2, j2] for i2, j2 in S6], axis=-1)

        cen_w = feats[:, 7:10] - C
        axis_w = feats[:, 10:13]                      # raw axis (see below)
        n1_w = feats[:, 20:23]
        n2_w = sym_to_mat(feats[:, 23:29].astype(np.float64))
        mu = feats[:, 7:10].astype(np.float64)
        xx_w = sym_to_mat(feats[:, 37:43].astype(np.float64))
        # xx (cols 37:43) is the region-CENTRAL second moment per unit area
        # (GProp's MatrixOfInertia is about the region's own centroid);
        # shifting it to the part centroid is the parallel-axis term
        # + outer(c_r - C), not the raw-moment identity.
        xx_c = xx_w + np.einsum("ri,rj->rij", mu - C, mu - C)

        from itertools import product
        acc_cen = np.zeros_like(cen_w); acc_axis = np.zeros_like(axis_w)
        acc_n1 = np.zeros_like(n1_w)
        acc_n2 = np.zeros((R, 6)); acc_xx = np.zeros((R, 6))
        for flips in product((0, 1), repeat=3):
            wt = 1.0
            for a, fl in enumerate(flips):
                wt *= (1 - w_sign[a]) / 2 if fl else (1 + w_sign[a]) / 2
            if wt < 1e-9:
                continue
            Vs = V * np.array([-1 if f else 1 for f in flips])[None, :]
            c_s, a_s, n_s = cen_w @ Vs, axis_w @ Vs, n1_w @ Vs
            n2_s = np.einsum("ia,rij,jb->rab", Vs, n2_w, Vs)
            xx_s = np.einsum("ia,rij,jb->rab", Vs, xx_c, Vs)
            # Degeneracy averaging: SEQUENTIAL application of the two plane
            # averages is order-dependent when the bands overlap -- a partial
            # (0,1) blend leaks the arbitrary in-plane basis into component 0
            # before a full (1,2) average, which cannot remove it (a
            # near-cube is a counterexample). Instead blend INDEPENDENT
            # projections of the ORIGINAL value: with a = d01, b = d12 and
            # exact plane averages P01/P12 (and P3 = full isotropic),
            #   out = (1-a)(1-b) X + a(1-b) P01(X) + (1-a)b P12(X) + ab P3(X)
            # -- a convex combination; at b = 1 only P12 and P3 survive
            # (both independent of the (1,2) basis), symmetrically at a = 1,
            # and a = b = 1 is exactly isotropic. Operator-level invariance
            # under degenerate-subspace basis changes holds to ~1e-16.
            a_w, b_w = d01, d12
            if a_w > 0 or b_w > 0:
                def plane_avg_vec(u, i2, j2):
                    out = u.copy()
                    out[..., i2] = 0.0
                    out[..., j2] = 0.0
                    return out

                def plane_avg_mat(m, i2, j2):
                    out = np.zeros_like(m)
                    k3 = 3 - i2 - j2
                    md = 0.5 * (m[..., i2, i2] + m[..., j2, j2])
                    out[..., i2, i2] = md
                    out[..., j2, j2] = md
                    out[..., k3, k3] = m[..., k3, k3]
                    return out

                def blend_vec(u):
                    return ((1 - a_w) * (1 - b_w) * u
                            + a_w * (1 - b_w) * plane_avg_vec(u, 0, 1)
                            + (1 - a_w) * b_w * plane_avg_vec(u, 1, 2))
                            # + a_w*b_w*0 (SO(3) mean of a vector)

                def blend_mat(m):
                    tr3 = np.einsum("rii->r", m) / 3.0
                    iso = tr3[:, None, None] * np.eye(3)[None]
                    return ((1 - a_w) * (1 - b_w) * m
                            + a_w * (1 - b_w) * plane_avg_mat(m, 0, 1)
                            + (1 - a_w) * b_w * plane_avg_mat(m, 1, 2)
                            + a_w * b_w * iso)

                c_s = blend_vec(c_s)
                a_s = blend_vec(a_s)
                n_s = blend_vec(n_s)
                n2_s = blend_mat(n2_s)
                xx_s = blend_mat(xx_s)
            acc_cen += wt * c_s; acc_axis += wt * np.abs(a_s); acc_n1 += wt * n_s
            acc_n2 += wt * mat_to_sym(n2_s); acc_xx += wt * mat_to_sym(xx_s)
        feats[:, 7:10] = acc_cen
        feats[:, 10:13] = acc_axis
        feats[:, 20:23] = acc_n1
        feats[:, 23:29] = acc_n2
        feats[:, 37:43] = acc_xx

    if os.environ.get("BHIST"):
        # 8-bin length-weighted histogram of SIGNED dihedral angle over the
        # region boundary -- additive over boundary edges, hence partition-
        # invariant exactly like blen/bconcave. SOFT (linear) binning is
        # mandatory: cylinder-plane boundaries sit at exactly +-90deg, and a
        # hard bin edge there flips bins on mesh-epsilon angle differences.
        NB = 8
        bh = np.zeros((R, NB), dtype=np.float32)
        for r in range(R):
            tot = max(accs[r]["blen"], 1e-12)
            for Lx, angx, _v0, _v1 in bd[r]:
                t = float(np.clip((angx + 180.0) / 360.0 * NB - 0.5, 0.0, NB - 1.0))
                i0 = int(t); w1 = t - i0; i1 = min(i0 + 1, NB - 1)
                bh[r, i0] += (1.0 - w1) * Lx / tot
                bh[r, i1] += w1 * Lx / tot
        feats = np.concatenate([feats, bh], axis=1)

    if os.environ.get("CHIST"):
        # 6+6-bin area-weighted histograms of mean (H) and Gaussian (K)
        # curvature over the region -- the DISTRIBUTION, not just the mean.
        # Additive over triangles, hence over faces. Curvature has units
        # 1/length (1/length^2 for K), so values are made scale-free with
        # s = sqrt(total part area) -- a whole-solid quantity, identical
        # under any partition -- then squashed with atan and soft-binned.
        NC = 6
        s_part = np.sqrt(max(float(area.sum()), 1e-12))
        ch = np.zeros((R, 2 * NC), dtype=np.float32)
        for r in range(R):
            ta = np.asarray(accs[r]["tri_areas"])
            if len(ta) == 0:
                continue
            a_r = max(float(ta.sum()), 1e-12)
            for off, vals, sc in ((0, accs[r]["tri_H"], s_part),
                                  (NC, accs[r]["tri_K"], s_part * s_part)):
                z = np.arctan(np.asarray(vals) * sc) / (np.pi / 2)   # [-1, 1]
                t = np.clip((z + 1.0) * 0.5 * NC - 0.5, 0.0, NC - 1.0)
                i0 = np.floor(t).astype(int); w1 = t - i0
                i1 = np.minimum(i0 + 1, NC - 1)
                np.add.at(ch[r], off + i0, (1.0 - w1) * ta / a_r)
                np.add.at(ch[r], off + i1, w1 * ta / a_r)
        feats = np.concatenate([feats, ch], axis=1)

    if os.environ.get("CANON_GRID"):
        from region_canon_grid import canon_grid, N_GRID
        cg = np.zeros((R, N_GRID), dtype=np.float32)
        for r in range(R):
            tp = np.array(accs[r]["tri_pts"]); ta = np.array(accs[r]["tri_areas"])
            if len(tp) == 0:
                continue
            M = gp_acc[r].MatrixOfInertia()
            I = np.array([[M.Value(i2 + 1, j2 + 1) for j2 in range(3)]
                          for i2 in range(3)])
            cg[r] = canon_grid(tp, ta, area[r], cen[r], I,
                               accs[r].get("tri_verts"))
        feats = np.concatenate([feats, cg], axis=1)

    src, dst, ef = [], [], []
    for (ri, rj), av in agg.items():
        L, LA, Ls, Lc = av[0], av[1], av[2], av[3]
        ang = LA / max(L, 1e-12)
        e = [L, ang / 180.0, Ls / max(L, 1e-12), Lc / max(L, 1e-12)]
        if EHIST:
            e += [h / max(L, 1e-12) for h in av[4:]]
        src += [ri, rj]; dst += [rj, ri]; ef += [e, e]
    if not src:
        return None
    return f2r, feats, (np.array(src), np.array(dst)), np.array(ef, dtype=np.float32)
