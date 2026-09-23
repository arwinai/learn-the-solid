"""A B-rep-NATIVE canonical input: the maximal-analytic-surface graph.

Idea: the face partition is arbitrary, but the underlying ANALYTIC SURFACE each
face lies on is not.  A cut and an extrude both produce the same cylinder; two
coplanar faces of a fused L-bracket lie on one plane.  So group faces by their
underlying surface (type + intrinsic parameters, canonicalised), and describe
each surface by AREA INTEGRALS over its trimmed region -- which are additive, so
splitting a face leaves them unchanged.

Unlike smoothness-based patch merging, this does NOT collapse under filleting:
a fillet ADDS a new surface whose area -> 0 as r -> 0, rather than merging two
existing patches discontinuously.

This module provides `surface_key` (the canonical identity of a face's
underlying surface, with optional analytic recognition of spline-stored
quadrics, RECOG=1) and the per-surface accumulators used by region_graph.py.
Running it directly executes a self-test on the construction-variant fixtures:
  A. do two constructions of the same solid yield the same surface multiset?
  B. is it stable under a fillet sweep (continuity)?
"""
from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
from OCC.Core.TopoDS import topods
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.GeomAbs import (GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
                              GeomAbs_Sphere, GeomAbs_Torus,
                              GeomAbs_BSplineSurface, GeomAbs_BezierSurface,
                              GeomAbs_SurfaceOfRevolution,
                              GeomAbs_SurfaceOfExtrusion,
                              GeomAbs_Line, GeomAbs_Circle)
from OCC.Core.BRepGProp import brepgprop_SurfaceProperties
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepFilletAPI import BRepFilletAPI_MakeFillet
from OCC.Core.gp import gp_Pnt

TYPES = {GeomAbs_Plane: "plane", GeomAbs_Cylinder: "cyl", GeomAbs_Cone: "cone",
         GeomAbs_Sphere: "sphere", GeomAbs_Torus: "torus",
         GeomAbs_BSplineSurface: "bspline", GeomAbs_BezierSurface: "bezier"}


def surface_key(face, tol=6):
    """Canonical identity of the UNDERLYING analytic surface, independent of trim.

    Signs are canonicalised (a plane and its opposite orientation are one surface),
    so the key depends only on the point set the surface occupies in space.
    """
    s = BRepAdaptor_Surface(face)
    t = s.GetType()
    name = TYPES.get(t, "other")
    if t == GeomAbs_Plane:
        pl = s.Plane(); a, b, c = pl.Axis().Direction().Coord()
        d = pl.Axis().Location().XYZ().Dot(pl.Axis().Direction().XYZ())
        v = np.array([a, b, c, d])
        if v[np.argmax(np.abs(v) > 1e-9)] < 0:      # fix global sign
            v = -v
        return (name,) + tuple(np.round(v, tol))
    if t == GeomAbs_Cylinder:
        cy = s.Cylinder(); ax = cy.Axis()
        d = np.array(ax.Direction().Coord()); p = np.array(ax.Location().Coord())
        if d[np.argmax(np.abs(d) > 1e-9)] < 0:
            d = -d
        p = p - d * np.dot(p, d)                    # closest point to origin on axis
        return (name, round(cy.Radius(), tol)) + tuple(np.round(d, tol)) + tuple(np.round(p, tol))
    if t == GeomAbs_Cone:
        # Canonicalize on the APEX, not OCC's stored reference station: the
        # same cone re-anchored at a different station changes RefRadius and
        # Location together. Routing through
        # _cone_key also unifies native and recognized cone keys.
        co = s.Cone(); ax = co.Axis()
        apex = np.array(co.Apex().Coord())
        d = np.array(ax.Direction().Coord())
        sa = float(co.SemiAngle())
        if sa < 0:                      # (d, sa) == (-d, -sa); enforce sa > 0
            d = -d; sa = -sa
        return _cone_key(apex, d, sa, tol)
    if t == GeomAbs_Sphere:
        sp = s.Sphere()
        return (name, round(sp.Radius(), tol)) + tuple(np.round(sp.Location().Coord(), tol))
    if t == GeomAbs_Torus:
        to = s.Torus(); tax = to.Axis()
        d = np.array(tax.Direction().Coord())
        if d[np.argmax(np.abs(d) > 1e-9)] < 0:   # torus is axis-flip symmetric
            d = -d
        return (name, round(to.MajorRadius(), tol),
                round(to.MinorRadius(), tol)) + \
               tuple(np.round(to.Axis().Location().Coord(), tol)) + \
               tuple(np.round(d, tol))
    # SWEPT-NOTATION RECOGNITION (env SWEEP_RECOG=1): kernels can store the
    # same quadric a THIRD way besides the analytic types and B-splines --
    # Geom_SurfaceOfRevolution / Geom_SurfaceOfLinearExtrusion, defined by a
    # basis curve plus an axis/direction (FreeCAD's Part API, for example, emits
    # these). A revolved line IS a cylinder/cone/plane and a revolved
    # meridian circle IS a torus/sphere; an extruded line IS a plane and a
    # perpendicularly-extruded circle IS a cylinder. Without this branch a
    # revolved-line cone never merges/matches an analytic cone. Non-analytic
    # sweeps (freeform profiles, oblique circle extrusions, skew revolved
    # lines) keep their swept-type keys below.
    if os.environ.get("SWEEP_RECOG") and t in (GeomAbs_SurfaceOfRevolution,
                                               GeomAbs_SurfaceOfExtrusion):
        try:
            k = _recognize_sweep(s, tol)
        except Exception:
            k = None
        if k is not None:
            return k

    # Analytic RECOGNITION (env RECOG=1): kernel translation re-expresses
    # analytic surfaces as NURBS (BRepBuilderAPI_NurbsConvert reproduces this
    # locally); a spline that IS exactly a plane/cylinder/sphere must key as
    # one or merging breaks completely under cross-kernel round-trips
    # (measured: split+NurbsConvert -> zero merges survive). Quadrics are
    # representable exactly as rational B-splines, so recognition residuals
    # are ~1e-9 when true and O(1) when not: a wide, safe margin.
    if os.environ.get("RECOG") and t in (GeomAbs_BSplineSurface,
                                         GeomAbs_BezierSurface):
        k = _recognize_analytic(s, tol)
        if k is not None:
            return k

    # Non-analytic (B-spline / Bezier): key on the UNTRIMMED surface's control
    # net + knots + degrees.  Trimming or splitting a face leaves the underlying
    # Geom surface untouched, so this is partition-invariant -- whereas the old
    # area-based fallback was NOT (two halves of a split spline face got
    # different keys and never merged back).
    try:
        if t == GeomAbs_BSplineSurface:
            b = s.BSpline()
        elif t == GeomAbs_BezierSurface:
            b = s.Bezier()
        else:
            b = None
        if b is not None:
            nu, nv = b.NbUPoles(), b.NbVPoles()
            poles = []
            for i in range(1, nu + 1):
                for j in range(1, nv + 1):
                    poles.extend(np.round(b.Pole(i, j).Coord(), tol))
            return (name, b.UDegree(), b.VDegree(), nu, nv) + tuple(poles)
    except Exception:
        pass
    return (name, round(area_of(face), tol))




def _plane_key(n, d, tol):
    v = np.array([n[0], n[1], n[2], d])
    if v[np.argmax(np.abs(v) > 1e-9)] < 0:
        v = -v
    return ("plane",) + tuple(np.round(v, tol))


def _cyl_key(r, d, p, tol):
    d = np.asarray(d, float); p = np.asarray(p, float)
    if d[np.argmax(np.abs(d) > 1e-9)] < 0:
        d = -d
    p = p - d * np.dot(p, d)
    return ("cyl", round(float(r), tol)) + tuple(np.round(d, tol)) + tuple(np.round(p, tol))


# auxiliary fit data of RECOG-recognized surfaces, keyed by the surface key.
# Tori: the native torus key stores no direction (the center is intrinsic),
# but the feature backfill in region_graph needs the axis column filled
# exactly like a native torus face -> value = axis tuple(d).
# Cones: under CONE_INTRINSIC the backfill needs the UNROUNDED fitted apex/
# axis/semi-angle (the key's entries are rounded) -> value = (apex3, dir3, sa).
# Written on recognition, read in the same process.
RECOG_AUX = {}


def _cone_key(apex, d_open, ang, tol):
    """Native-layout cone key (name, refR, semiAngle, dir3, loc3) from a fit.

    A native GeomAbs_Cone keys on OCC's stored (RefRadius @ Location) whose
    anchor is construction data that NurbsConvert destroys, so the fitted key
    uses the same canonicalization _cyl_key already uses: sign-fix the axis
    direction, anchor at the closest point of the axis to the origin, and put
    the (signed) parametric radius at that anchor in the RefRadius slot --
    which reproduces the native values exactly for origin-anchored cones
    (BRepPrimAPI_MakeCone and both its semi-angle signs verified).

    apex: fitted apex point; d_open: unit axis, radius GROWING along it;
    ang: semi-angle magnitude in (0, pi/2).
    """
    q = np.asarray(apex, float); d = np.asarray(d_open, float)
    sa = float(ang)
    if d[np.argmax(np.abs(d) > 1e-9)] < 0:      # same sign fix as _cyl_key;
        d = -d; sa = -sa                        # radius then grows along -d
    p = q - d * np.dot(q, d)                    # closest point to origin on axis
    refr = np.tan(sa) * float(np.dot(d, p - q))  # R(v) = refR + v sin(sa) => this
    return ("cone", round(float(refr), tol), round(sa, tol)) + \
        tuple(np.round(d, tol)) + tuple(np.round(p, tol))


def _torus_key(R, r, c, tol, axis=None):
    """Native-layout torus key (name, majorR, minorR, center3, axis3). The
    axis is part of surface identity (same center+radii, different axis is a
    different torus); unoriented, sign-fixed."""
    k = ("torus", round(float(R), tol), round(float(r), tol)) + \
        tuple(np.round(np.asarray(c, float), tol))
    if axis is not None:
        d = np.asarray(axis, float)
        if d[np.argmax(np.abs(d) > 1e-9)] < 0:
            d = -d
        k = k + tuple(np.round(d, tol))
    return k


# An exact re-expression of a quadric fits at machine precision (a cylinder
# converts to an exact rational NURBS), while a patch that merely LOOKS like
# another quadric sits many orders above. EXACT_TOL is that fingerprint, and
# DEGEN_MAX rejects a fit whose radius is not commensurate with the patch it
# covers -- a least-squares sphere through nearly-coplanar samples is
# rank-deficient and can report residual 0 with a radius 1e12x the extent.
EXACT_TOL = 1e-12
DEGEN_MAX = 1e3
# A fit can also run away in a parameter that is NOT its radius: a nearly
# cylindrical patch admits a torus whose centre sits 1e9 away, which the
# radius test above does not see. Coordinates here are part-normalised (unit
# boundary area about the origin), so every legitimate key parameter is O(1);
# PARAM_MAX rejects a candidate whose key carries anything absurd.
PARAM_MAX = 1e3


def _sane(key, rad_over_extent):
    """Is this candidate physically commensurate with the patch and the part?"""
    if rad_over_extent > DEGEN_MAX:
        return False
    return all(abs(v) <= PARAM_MAX for v in key[1:]
               if isinstance(v, (int, float)))


def _recognize_analytic(adaptor, tol, K=6, rtol=1e-6):
    """If the (spline) surface is exactly a plane/cylinder/sphere/cone/torus
    over the face's parameter box, return the ANALYTIC key it would have had
    natively; else None. Sample K*K points+normals, fit, verify residual.

    A small or anisotropic patch can satisfy several fits: a sliver of a
    cylinder is planar to any fixed tolerance, and a thin band on a cylinder of
    radius R is matched by a sphere of the same radius to ~t^2/8R over axial
    extent t. Returning the first acceptable type would make the key depend on
    the order the fits appear below -- an artefact of this file rather than a
    property of the solid. Candidates are therefore collected, and the first
    (simplest) admissible one wins UNLESS another reproduces the patch to
    machine precision with a non-degenerate radius, which is the signature of
    an exact re-expression of that very surface."""
    from OCC.Core.BRepLProp import BRepLProp_SLProps
    u0, u1 = adaptor.FirstUParameter(), adaptor.LastUParameter()
    v0, v1 = adaptor.FirstVParameter(), adaptor.LastVParameter()
    if not all(np.isfinite([u0, u1, v0, v1])):
        return None
    P, N = [], []
    for iu in range(K):
        for iv in range(K):
            u = u0 + (u1 - u0) * (iu + 0.5) / K
            v = v0 + (v1 - v0) * (iv + 0.5) / K
            pr = BRepLProp_SLProps(adaptor, u, v, 1, 1e-9)
            if not pr.IsNormalDefined():
                return None
            P.append(pr.Value().Coord()); N.append(pr.Normal().Coord())
    P, N = np.asarray(P), np.asarray(N)
    scale = max(np.ptp(P, axis=0).max(), 1e-9)

    cands = []            # (rel residual, radius/extent, key, aux)

    # PLANE: normals parallel + points coplanar
    n0 = N.mean(0); n0 /= max(np.linalg.norm(n0), 1e-12)
    if np.abs(np.abs(N @ n0) - 1).max() < rtol:
        d = float((P @ n0).mean())
        e = float(np.abs(P @ n0 - d).max()) / scale
        if e < rtol:
            cands.append((e, 0.0, _plane_key(n0, d, tol), None))

    # SPHERE: |p - c| = r, linear least squares
    A = np.c_[2 * P, np.ones(len(P))]
    b = (P ** 2).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c, r2 = sol[:3], sol[3] + (sol[:3] ** 2).sum()
    if r2 > 0:
        r = np.sqrt(r2)
        e = float(np.abs(np.linalg.norm(P - c, axis=1) - r).max()) / scale
        if e < rtol:
            cands.append((e, float(r) / scale,
                          ("sphere", round(float(r), tol))
                          + tuple(np.round(c, tol)), None))

    # CYLINDER: normals orthogonal to axis; constant distance to axis
    Cn = np.cov(N.T)
    w, V = np.linalg.eigh(Cn)
    a = V[:, 0]                                     # least-variance direction
    if np.abs(N @ a).max() < rtol:
        Pp = P - np.outer(P @ a, a)                 # project out axis
        A2 = np.c_[2 * Pp, np.ones(len(Pp))]
        b2 = (Pp ** 2).sum(1)
        sol2, *_ = np.linalg.lstsq(A2, b2, rcond=None)
        c2 = sol2[:3]
        c2 = c2 - a * np.dot(c2, a)
        r = float(np.linalg.norm(Pp - c2, axis=1).mean())
        e = float(np.abs(np.linalg.norm(Pp - c2, axis=1) - r).max()) / scale
        if e < rtol:
            cands.append((e, r / scale, _cyl_key(r, a, c2, tol), None))

    # CONE: same least-variance axis, but every normal makes one CONSTANT
    # NONZERO angle with it (the complement of the semi-angle); the apex q is
    # the intersection of the tangent planes, n_i . (p_i - q) = 0 (linear).
    ca = N @ a
    c0 = float(ca.mean())
    if np.abs(ca - c0).max() < rtol and rtol < abs(c0) < 1.0 - 1e-9:
        q, *_ = np.linalg.lstsq(N, (N * P).sum(1), rcond=None)
        ang = float(np.arcsin(abs(c0)))             # |n . axis| = sin(semi-angle)
        rel = P - q
        h = rel @ a                                 # signed distance from apex
        rad = np.linalg.norm(rel - np.outer(h, a), axis=1)
        # single nappe only (both-nappe patches cannot key deterministically)
        e = max(float(np.abs((N * rel).sum(1)).max()),
                float(np.abs(rad - np.abs(h) * np.tan(ang)).max())) / scale
        if h.min() * h.max() > 0 and e < rtol:
            d_open = a if h.mean() > 0 else -a      # radius grows away from apex
            k = _cone_key(q, d_open, ang, tol)
            cands.append((e, float(rad.max()) / scale, k,
                          (tuple(np.asarray(q, float)),
                           tuple(np.asarray(d_open, float)), float(ang))))

    # TORUS: the axis is the common transversal of every normal line
    # (Plucker: (p x n) . d + n . m = 0, homogeneous linear in (d, m)); the
    # meridian section (rho, h) then lies on a circle -- linear in
    # (2R, 2h0, r^2 - R^2 - h0^2). Both fits are exact on a true torus and
    # the circle check has an O(1) margin on any non-circular revolved
    # profile, which the axis fit alone would accept.
    Pm = P.mean(0)
    Pc = (P - Pm) / scale                           # centered, scale-free
    M6 = np.c_[np.cross(Pc, N), N]
    sv = np.linalg.svd(M6, compute_uv=False)
    if sv[-1] < rtol * max(sv[0], 1e-12):
        _, _, Vt = np.linalg.svd(M6)
        d = Vt[-1][:3]; nd = float(np.linalg.norm(d))
        if nd > 1e-9:
            d = d / nd
            m = Vt[-1][3:] / nd
            m = m - d * np.dot(m, d)
            q = np.cross(d, m) * scale + Pm         # point on the fitted axis
            rel = P - q
            h = rel @ d
            rho = np.linalg.norm(rel - np.outer(h, d), axis=1)
            A3 = np.c_[2 * rho, 2 * h, np.ones(len(P))]
            b3 = rho ** 2 + h ** 2
            sol3, *_ = np.linalg.lstsq(A3, b3, rcond=None)
            R, h0 = float(sol3[0]), float(sol3[1])
            rr2 = float(sol3[2]) + R * R + h0 * h0
            if R > rtol * scale and rr2 > 0:
                r = float(np.sqrt(rr2))
                res = np.abs(np.sqrt((rho - R) ** 2 + (h - h0) ** 2) - r)
                e = float(res.max()) / scale
                if e < rtol:
                    k = _torus_key(R, r, q + h0 * d, tol, axis=d)
                    cands.append((e, R / scale, k, tuple(d)))

    if not cands:
        return None
    # Simplest admissible type by default. An exact fit may override it, but
    # only if it is non-degenerate: the ordering currently PROTECTS against
    # degenerate fits (a torus branch reached only when everything else fails
    # can key a near-cylinder with a centre 1e9 away), so removing the
    # ordering means making that protection explicit.
    best = cands[0]
    exact = [c for c in cands if c[0] < EXACT_TOL and _sane(c[2], c[1])]
    if exact:
        best = min(exact, key=lambda c: c[0])
    if best[3] is not None:
        RECOG_AUX[best[2]] = best[3]
    return best[2]


# SWEEP_RECOG diagnostics: when env SWEEP_DEBUG is set, every swept-surface
# recognition attempt appends a dict here (curve-fit and surface-verify
# margins, accept/reject and the deciding reason). Read by the gate scripts
# only; extraction never looks at it.
SWEEP_LOG = []


def _sweep_log(**kw):
    if os.environ.get("SWEEP_DEBUG"):
        SWEEP_LOG.append(kw)


def _curve_as_line_or_circle(c, rtol=1e-6, M=25):
    """Basis-curve recognition for the swept branch.

    Adaptor3d curve -> ("line", (point, unit_dir), rel_margin) |
    ("circle", (center, unit_normal, radius), rel_margin) | None.

    Native GeomAbs_Line/Circle read exactly (margin 0). Anything else is
    sampled at M points and fitted with the same 1e-6*scale accept the
    surface recognizer uses -- some exporters store extrusion
    profiles as degree-1 B-splines that ARE lines. The line fit runs first:
    a perfect line also satisfies a circle fit with r -> inf, never the
    reverse, so the order is load-bearing."""
    t = c.GetType()
    if t == GeomAbs_Line:
        l = c.Line()
        return ("line", (np.asarray(l.Location().Coord(), float),
                         np.asarray(l.Direction().Coord(), float)), 0.0)
    if t == GeomAbs_Circle:
        ci = c.Circle()
        return ("circle", (np.asarray(ci.Location().Coord(), float),
                           np.asarray(ci.Axis().Direction().Coord(), float),
                           float(ci.Radius())), 0.0)
    try:
        t0, t1 = c.FirstParameter(), c.LastParameter()
    except Exception:
        return None
    if not (np.isfinite(t0) and np.isfinite(t1)) or t1 <= t0:
        return None
    Q = np.array([c.Value(t0 + (t1 - t0) * (i + 0.5) / M).Coord()
                  for i in range(M)])
    scale = max(np.ptp(Q, axis=0).max(), 1e-9)
    m = Q.mean(0)
    _, _, Vt = np.linalg.svd(Q - m)
    d = Vt[0]
    res_line = float(np.linalg.norm((Q - m) - np.outer((Q - m) @ d, d),
                                    axis=1).max())
    if res_line < rtol * scale:
        return ("line", (m, d), res_line / scale)
    n = Vt[2]
    res_pl = float(np.abs((Q - m) @ n).max())
    if res_pl < rtol * scale:
        e1, e2 = Vt[0], Vt[1]
        xy = np.c_[(Q - m) @ e1, (Q - m) @ e2]
        A = np.c_[2 * xy, np.ones(len(xy))]
        sol, *_ = np.linalg.lstsq(A, (xy ** 2).sum(1), rcond=None)
        cc, r2 = sol[:2], float(sol[2] + (sol[:2] ** 2).sum())
        if r2 > 0:
            r = float(np.sqrt(r2))
            c3 = m + cc[0] * e1 + cc[1] * e2
            res_c = float(np.abs(np.linalg.norm(Q - c3, axis=1) - r).max())
            if res_c < rtol * scale:
                return ("circle", (c3, n, r), max(res_pl, res_c) / scale)
            _sweep_log(stage="curve", verdict="reject-circle-residual",
                       line_rel=res_line / scale, circle_rel=res_c / scale)
            return None
    _sweep_log(stage="curve", verdict="reject-nonplanar",
               line_rel=res_line / scale, planarity_rel=res_pl / scale)
    return None


def _recognize_sweep(adaptor, tol, K=6, rtol=1e-6):
    """SWEEP_RECOG=1: recognize a GeomAbs_SurfaceOfRevolution /
    GeomAbs_SurfaceOfExtrusion whose sweep IS exactly an analytic surface and
    return the key it would have had natively; else None.

    The mapping is SYMBOLIC from the basis curve + axis/direction:
      revolution of a line:   parallel -> cylinder (r = line-axis distance);
                              perpendicular -> plane (annulus, normal = axis);
                              oblique intersecting -> cone (apex at the
                              intersection, semi-angle = line/axis angle);
                              oblique skew -> hyperboloid, REJECT.
      revolution of a circle: plane through the axis -> torus (center-off-
                              axis) or sphere (center ON the axis); any other
                              circle placement REJECT.
      extrusion of a line:    plane (unless swept along itself);
      extrusion of a circle:  cylinder ONLY when the direction is
                              perpendicular to the circle plane; oblique
                              extrusion is an elliptic cylinder, REJECT.
    The candidate is then VERIFIED against K*K face sample points at the same
    1e-6*scale margin the spline recognizer uses (exact notations sit at
    ~1e-15, non-analytic sweeps at O(1)). Keys and RECOG_AUX backfill are
    EXACTLY the native-analytic-branch ones (_plane/_cyl/_cone/_torus_key)."""
    t = adaptor.GetType()
    u0, u1 = adaptor.FirstUParameter(), adaptor.LastUParameter()
    v0, v1 = adaptor.FirstVParameter(), adaptor.LastVParameter()
    if not all(np.isfinite([u0, u1, v0, v1])):
        return None
    P = np.array([adaptor.Value(u0 + (u1 - u0) * (iu + 0.5) / K,
                                v0 + (v1 - v0) * (iv + 0.5) / K).Coord()
                  for iu in range(K) for iv in range(K)])
    scale = max(np.ptp(P, axis=0).max(), 1e-9)
    try:
        bc = adaptor.BasisCurve()
    except Exception:
        return None
    rec = _curve_as_line_or_circle(bc, rtol)
    if rec is None:
        return None
    kind, prm, cmarg = rec

    key = resid = aux = None
    why = "no-analytic-map"
    if t == GeomAbs_SurfaceOfRevolution:
        ax = adaptor.AxeOfRevolution()
        pa = np.asarray(ax.Location().Coord(), float)
        da = np.asarray(ax.Direction().Coord(), float)
        da /= max(np.linalg.norm(da), 1e-12)
        if kind == "line":
            p0 = np.asarray(prm[0], float)
            dl = np.asarray(prm[1], float)
            dl /= max(np.linalg.norm(dl), 1e-12)
            w = np.cross(dl, da)
            sin_a = float(np.linalg.norm(w))
            cos_a = abs(float(np.dot(dl, da)))
            rel = p0 - pa
            if sin_a < rtol:                     # parallel -> cylinder
                r = float(np.linalg.norm(rel - da * np.dot(rel, da)))
                if r > rtol * scale:
                    rp = P - pa
                    resid = float(np.abs(np.linalg.norm(
                        rp - np.outer(rp @ da, da), axis=1) - r).max())
                    key = _cyl_key(r, da, pa, tol)
                else:
                    why = "line-on-axis-degenerate"
            elif cos_a < rtol:                   # perpendicular -> plane
                dpl = float(np.dot(p0, da))      # constant along the line
                resid = float(np.abs(P @ da - dpl).max())
                key = _plane_key(da, dpl, tol)
            else:
                # oblique: must INTERSECT the axis, else hyperboloid
                gap = abs(float(np.dot(rel, w))) / max(sin_a, 1e-12)
                if gap < rtol * scale:
                    st, *_ = np.linalg.lstsq(np.c_[dl, -da], pa - p0,
                                             rcond=None)
                    q = pa + st[1] * da          # apex: on axis AND line
                    ang = float(np.arccos(np.clip(cos_a, 0.0, 1.0)))
                    h = (P - q) @ da
                    if h.min() * h.max() > 0:    # single nappe only
                        d_open = da if h.mean() > 0 else -da
                        rad = np.linalg.norm((P - q) - np.outer(h, da),
                                             axis=1)
                        resid = float(np.abs(rad - np.abs(h)
                                             * np.tan(ang)).max())
                        key = _cone_key(q, d_open, ang, tol)
                        aux = (tuple(q.tolist()), tuple(d_open.tolist()),
                               float(ang))
                    else:
                        why = "cone-face-spans-both-nappes"
                else:
                    why = f"skew-line-hyperboloid(gap_rel={gap / scale:.3g})"
        else:                                    # circle
            c0 = np.asarray(prm[0], float)
            ncr = np.asarray(prm[1], float)
            rc = float(prm[2])
            ncr /= max(np.linalg.norm(ncr), 1e-12)
            # meridian: the circle's plane must CONTAIN the axis
            if abs(float(np.dot(ncr, da))) < rtol and \
                    abs(float(np.dot(ncr, pa - c0))) < rtol * scale:
                relc = c0 - pa
                hR = float(np.dot(relc, da))
                Rmaj = float(np.linalg.norm(relc - hR * da))
                if Rmaj < rtol * scale:          # center ON axis -> sphere
                    resid = float(np.abs(np.linalg.norm(P - c0, axis=1)
                                         - rc).max())
                    key = ("sphere", round(rc, tol)) + tuple(np.round(c0, tol))
                else:                            # torus (incl. spindle)
                    ctr = pa + hR * da
                    rp = P - ctr
                    h = rp @ da
                    rho = np.linalg.norm(rp - np.outer(h, da), axis=1)
                    resid = float(np.abs(np.sqrt((rho - Rmaj) ** 2 + h ** 2)
                                         - rc).max())
                    key = _torus_key(Rmaj, rc, ctr, tol)
                    aux = tuple(da.tolist())
            else:
                why = "circle-plane-not-through-axis"
    else:                                        # linear extrusion
        dd = np.asarray(adaptor.Direction().Coord(), float)
        dd /= max(np.linalg.norm(dd), 1e-12)
        if kind == "line":
            p0 = np.asarray(prm[0], float)
            dl = np.asarray(prm[1], float)
            n = np.cross(dl, dd)
            nn = float(np.linalg.norm(n))
            if nn > rtol:                        # -> plane
                n /= nn
                dpl = float(np.dot(p0, n))
                resid = float(np.abs(P @ n - dpl).max())
                key = _plane_key(n, dpl, tol)
            else:
                why = "line-extruded-along-itself"
        else:                                    # circle
            c0 = np.asarray(prm[0], float)
            ncr = np.asarray(prm[1], float)
            rc = float(prm[2])
            ncr /= max(np.linalg.norm(ncr), 1e-12)
            obl = float(np.linalg.norm(np.cross(dd, ncr)))
            if obl < rtol:                       # perpendicular -> cylinder
                rp = P - c0
                resid = float(np.abs(np.linalg.norm(
                    rp - np.outer(rp @ ncr, ncr), axis=1) - rc).max())
                key = _cyl_key(rc, ncr, c0, tol)
            else:                                # NOT a cylinder
                why = f"oblique-circle-extrusion(sin={obl:.3g})"
    if key is None or resid >= rtol * scale:
        _sweep_log(stage="surface", verdict="reject",
                   surf="rev" if t == GeomAbs_SurfaceOfRevolution else "ext",
                   curve=kind, curve_rel=cmarg,
                   why=why if key is None else "surface-residual",
                   resid_rel=None if resid is None else resid / scale)
        return None
    if aux is not None:
        RECOG_AUX[key] = aux
    _sweep_log(stage="surface", verdict="accept",
               surf="rev" if t == GeomAbs_SurfaceOfRevolution else "ext",
               curve=kind, curve_rel=cmarg, map=key[0],
               resid_rel=resid / scale)
    return key


def area_of(face):
    g = GProp_GProps(); brepgprop_SurfaceProperties(face, g)
    return g.Mass()


def centroid_of(face):
    g = GProp_GProps(); brepgprop_SurfaceProperties(face, g)
    return np.array(g.CentreOfMass().Coord())


def surface_graph(shape):
    """Faces -> dict{surface_key: (total_area, area-weighted centroid, n_faces)}.

    Area and area-weighted centroid are INTEGRALS over the trimmed region, so
    they are additive: splitting a face into pieces leaves them unchanged.
    """
    acc = {}
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        f = topods.Face(ex.Current()); ex.Next()
        k = surface_key(f)
        a = area_of(f); c = centroid_of(f)
        if k in acc:
            a0, c0, n = acc[k]
            acc[k] = (a0 + a, (c0 * a0 + c * a) / (a0 + a), n + 1)
        else:
            acc[k] = (a, c, 1)
    return acc


def signature(g):
    """Order-free comparable signature: sorted (type, area, centroid) tuples."""
    return sorted((k[0], round(v[0], 6)) + tuple(np.round(v[1], 6)) for k, v in g.items())


def fillet_vertical_edges(shape, rad, max_edges=4):
    mk = BRepFilletAPI_MakeFillet(shape)
    ex = TopExp_Explorer(shape, TopAbs_EDGE); n = 0
    while ex.More() and n < max_edges:
        e = topods.Edge(ex.Current()); ex.Next()
        c = BRepAdaptor_Curve(e)
        p0 = c.Value(c.FirstParameter()); p1 = c.Value(c.LastParameter())
        if (abs(p0.X()-p1.X()) < 1e-9 and abs(p0.Y()-p1.Y()) < 1e-9
                and abs(p0.Z()-p1.Z()) > 1e-6
                and abs(abs(p0.X())-1.0) < 1e-6 and abs(abs(p0.Y())-0.6) < 1e-6):
            try:
                mk.Add(rad, e); n += 1
            except Exception:
                pass
    if n == 0:
        return None
    mk.Build()
    return mk.Shape() if mk.IsDone() else None


def n_faces(shape):
    n = 0; ex = TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        n += 1; ex.Next()
    return n


def _selftest():
    from fixtures import (ring_pair, slab_hole_pair, lprofile_pair,
                          block_with_hole, props)
    print("=== A. does the surface graph collapse construction differences? ===")
    print(f"{'family':10s} {'faces A/B':>10s} {'surfaces A/B':>13s} {'signature match':>16s}")
    rng = np.random.default_rng(0)
    n_match = n_tot = 0
    for i in range(8):
        R = rng.uniform(0.8, 1.4); r = rng.uniform(0.25, 0.55); h = rng.uniform(0.3, 0.9)
        for name, (A, B) in [("ring", ring_pair(R, r, h)),
                             ("slab_hole", slab_hole_pair(2*R, 1.6*R, h, r)),
                             ("Lprofile", lprofile_pair(1.5*R, 1.2*R, 0.3*R, h))]:
            va, aa = props(A); vb, ab = props(B)
            if va <= 0 or abs(va-vb)/va > 1e-9 or abs(aa-ab)/aa > 1e-9:
                continue
            ga, gb = surface_graph(A), surface_graph(B)
            ok = signature(ga) == signature(gb)
            n_match += ok; n_tot += 1
            if i < 2:
                print(f"{name:10s} {n_faces(A):4d}/{n_faces(B):<5d} {len(ga):6d}/{len(gb):<6d} "
                      f"{'MATCH' if ok else 'differ':>16s}")
    print(f"  -> {n_match}/{n_tot} pairs have an identical surface-graph signature")
    
    print("\n=== B. continuity under filleting ===")
    base = block_with_hole()
    g0 = surface_graph(base)
    print(f"  r=0.000 : faces {n_faces(base):3d}  surfaces {len(g0):3d}   (baseline)")
    for rad in (0.001, 0.01, 0.05, 0.10, 0.20, 0.30):
        f = fillet_vertical_edges(base, rad)
        if f is None:
            continue
        g = surface_graph(f)
        new = [k for k in g if k not in g0]
        new_area = sum(g[k][0] for k in new)
        tot = sum(v[0] for v in g.values())
        print(f"  r={rad:<5} : faces {n_faces(f):3d}  surfaces {len(g):3d}   "
              f"new surfaces {len(new):2d} carrying {100*new_area/tot:5.2f}% of area")
    print("  (a fillet ADDS surfaces whose area -> 0 smoothly; it does not merge existing ones)")


if __name__ == "__main__":
    _selftest()
