"""Canonical-frame spatial grid: recover within-region detail without UV grids.

Integral-only features have a cost on real parts with operation labels
(Fusion 360 segmentation): telling ExtrudeSide from CutSide often hinges on
fine within-face shape that a handful of integrals discards, which is exactly
what a UV grid captures.

UV grids fail invariance because the PARAMETERISATION is arbitrary. But a
region's own geometry defines a frame: the eigenvectors of its inertia tensor.
Binning the region's surface area into a KxKxK grid expressed in that frame
yields UV-grid-like spatial detail that re-partitioning cannot change, because
both the frame and the binned mass are functions of the region, not the faces.

Two details decide whether this is actually invariant:

  * SIGN of each principal axis is ambiguous. We fix it with the third moment of
    area along the axis (skewness): flip so it is non-negative. Skewness is
    quadrature-based and mesh-dependent, but it is only used for a DISCRETE sign
    choice, so mesh error is harmless unless the skew is near zero -- in that
    case the region is near-symmetric along that axis and BOTH signs bin the
    mass identically, so the choice does not matter.
  * DEGENERATE eigenvalues (surfaces of revolution): any rotation of the
    degenerate pair is valid. We detect near-degeneracy and, within the
    degenerate subspace, replace the grid by its rotational average (bin by
    radius instead of by the two ambiguous coordinates). Cylindrical regions get
    a radius x axis grid, which is exactly their natural symmetry.
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
import numpy as np

import os
K = int(os.environ.get("CG_K", "4"))   # grid resolution per axis
DEGEN_RTOL = 0.05           # eigenvalue closeness => treat as degenerate
N_GRID = K * K * K          # dense branch size; degenerate branch padded to same


def region_frame(area, cen, I):
    """Principal frame from the accumulated inertia matrix.

    GProp's MatrixOfInertia for the accumulated region system is already
    expressed about the region's own center of mass, so it is the region's
    central inertia and is used as-is (no parallel-axis shift)."""
    Ic = I
    w, V = np.linalg.eigh(0.5 * (Ic + Ic.T))
    return w, V                                    # ascending eigenvalues


MAX_TRIS = 2_000_000   # hard budget: unbounded 4x-per-round splitting reached
                       # 264GB RSS on one pathological part and the OOM killer
                       # took the whole training queue down with it. Past the
                       # budget we accept coarser quadrature on that region
                       # (graceful residual) instead of an OOM (dead box).


def subdivide(P0, P1, P2, A, max_diam):
    """Vectorised: split ALL oversize triangles each round (numpy), not one at a
    time (the Python-loop version took 78 min on one split's extraction)."""
    P0, P1, P2, A = map(np.asarray, (P0, P1, P2, A))
    for _ in range(12):
        d = np.maximum(np.linalg.norm(P1 - P0, axis=1),
                       np.maximum(np.linalg.norm(P2 - P1, axis=1),
                                  np.linalg.norm(P0 - P2, axis=1)))
        big = (d > max_diam) & (A > 1e-14)
        if not big.any() or len(A) + 3 * int(big.sum()) > MAX_TRIS:
            break
        b0, b1, b2, ba = P0[big], P1[big], P2[big], A[big] / 4.0
        m01, m12, m20 = (b0 + b1) / 2, (b1 + b2) / 2, (b2 + b0) / 2
        P0 = np.concatenate([P0[~big], b0, m01, m20, m01])
        P1 = np.concatenate([P1[~big], m01, b1, m12, m12])
        P2 = np.concatenate([P2[~big], m20, m12, b2, m20])
        A  = np.concatenate([A[~big], ba, ba, ba, ba])
    return (P0 + P1 + P2) / 3.0, A


def _subdivide_old(P0, P1, P2, A, max_diam):
    """Midpoint-subdivide triangles until diameter <= max_diam.
    On PLANES the mesher emits two giant triangles regardless of deflection
    (chordal error is zero there), so one-point-per-triangle quadrature is
    mesh-dependent no matter how fine the deflection. Bounding the triangle
    diameter bounds the quadrature error by geometry, not by the mesher's whim."""
    out_c, out_a = [], []
    stack = list(zip(P0, P1, P2, A))
    while stack:
        p0, p1, p2, a = stack.pop()
        d = max(np.linalg.norm(p1 - p0), np.linalg.norm(p2 - p1),
                np.linalg.norm(p0 - p2))
        if d <= max_diam or a < 1e-14:
            out_c.append((p0 + p1 + p2) / 3.0); out_a.append(a)
            continue
        m01, m12, m20 = (p0 + p1) / 2, (p1 + p2) / 2, (p2 + p0) / 2
        q = a / 4.0
        stack += [(p0, m01, m20, q), (m01, p1, m12, q),
                  (m20, m12, p2, q), (m01, m12, m20, q)]
    return np.array(out_c), np.array(out_a)


def canon_grid(tri_pts, tri_areas, area, cen, I, tri_verts=None):
    """[T,3] triangle centroids + areas -> K^3 canonical area-mass grid.

    Additive over faces by construction: the bins are computed from the SUM of
    per-triangle mass, and frame/extent depend only on accumulated integrals.
    """
    w, V = region_frame(area, cen, I)
    if tri_verts is not None and len(tri_verts):
        rms0 = np.sqrt(max(area, 1e-12))
        P = np.asarray(tri_verts)
        tri_pts, tri_areas = subdivide(P[:, 0], P[:, 1], P[:, 2],
                                       np.asarray(tri_areas), 0.15 * rms0 / K)
    tri_pts = np.asarray(tri_pts); tri_areas = np.asarray(tri_areas)
    X = (tri_pts - cen) @ V                        # coords in principal frame

    # fix axis signs by skewness of the area mass. On a NEAR-SYMMETRIC region
    # the skew is mesh-noise and a discrete sign choice flips between a face
    # and its re-partitioned halves (measured ~7e-2 planar residual; a hard
    # ambiguity THRESHOLD merely moves the knife edge to the threshold itself
    # -- solids-stable 98.9% instead of 99.6%). So the choice is made
    # CONTINUOUS: blend the signed and sign-averaged grids with a weight
    # w(|skew|) that ramps 0->1 over [TOL_LO, TOL_HI]. Lipschitz in the mesh
    # everywhere: symmetric regions get the exact average, clearly-skewed
    # regions the old signed grid, and nothing jumps in between.
    TOL_LO, TOL_HI = 0.01, 0.04
    rms_a = np.sqrt(np.maximum(np.sum(tri_areas[:, None] * X ** 2, 0) /
                               max(area, 1e-12), 1e-18))
    w_ax = np.ones(3)
    for a in range(3):
        s = float(np.sum(tri_areas * X[:, a] ** 3))
        s_norm = s / (max(area, 1e-12) * rms_a[a] ** 3)
        w_ax[a] = float(np.clip((abs(s_norm) - TOL_LO) / (TOL_HI - TOL_LO),
                                0.0, 1.0))
        if s < 0:
            X[:, a] = -X[:, a]          # preferred sign always applied first

    # detect degenerate pairs (rotational symmetry)
    degen01 = abs(w[0] - w[1]) <= DEGEN_RTOL * max(abs(w[2]), 1e-12)
    degen12 = abs(w[1] - w[2]) <= DEGEN_RTOL * max(abs(w[2]), 1e-12)

    # robust per-axis extent: rms radius (integral!), not min/max (mesh-dependent)
    rms = np.sqrt(np.maximum(np.sum(tri_areas[:, None] * X ** 2, 0) /
                             max(area, 1e-12), 1e-18))
    H = 2.5 * rms                                   # bin range = +-H

    def soft(v, h):
        """Soft (linear) bin assignment: (idx0, idx1, w1). Hard binning is a
        knife edge -- a planar face's mass sits exactly on a bin boundary and
        different triangulations drop it into different cells (measured residual
        0.5). Linear interpolation makes the grid Lipschitz in the sample
        positions, so mesh differences produce O(mesh) error instead."""
        t = np.clip((v / h + 1) * 0.5 * K - 0.5, 0.0, K - 1.0)
        i0 = np.floor(t).astype(int)
        w1 = t - i0
        i1 = np.minimum(i0 + 1, K - 1)
        return i0, i1, w1

    def bin_once(Xs):
        g = np.zeros((K, K, K), dtype=np.float64)
        m = tri_areas
        if degen01 and degen12:
            # TRIPLE degeneracy (spherically symmetric inertia): every axis of
            # the eigh basis is arbitrary, so any use of individual coordinates
            # is noise -- measured as the dominant residual on real parts
            # (median eigen-gap of grid-offending regions = 0.0). Only the 3D
            # radius is well-defined: bin purely radially.
            r3 = np.sqrt((Xs ** 2).sum(1))
            hr = 2.5 * float(np.sqrt(max(np.sum(m * r3 ** 2) /
                                         max(area, 1e-12), 1e-18)))
            a0, a1, aw = soft(2 * r3 - hr, hr)
            zz = np.zeros(len(m), int)
            for ia, wa in ((a0, 1 - aw), (a1, aw)):
                np.add.at(g, (ia, zz, zz), m * wa)
        elif degen01 or degen12:
            i, j = (0, 1) if degen01 else (1, 2)
            k3 = 3 - i - j
            r = np.sqrt(Xs[:, i] ** 2 + Xs[:, j] ** 2)
            hr = 2.5 * float(np.sqrt(max(np.sum(m * r ** 2) / max(area, 1e-12),
                                         1e-18)))
            a0, a1, aw = soft(2 * r - hr, hr)      # map r in [0,hr] onto [-h,h]
            b0, b1, bw = soft(Xs[:, k3], H[k3])
            for ia, wa in ((a0, 1 - aw), (a1, aw)):
                for ib, wb in ((b0, 1 - bw), (b1, bw)):
                    np.add.at(g, (ia, ib, np.zeros(len(m), int)), m * wa * wb)
        else:
            idx = [soft(Xs[:, a], H[a]) for a in range(3)]
            for da in (0, 1):
                ia = idx[0][da]; wa = idx[0][2] if da else 1 - idx[0][2]
                for db in (0, 1):
                    ib = idx[1][db]; wb = idx[1][2] if db else 1 - idx[1][2]
                    for dc in (0, 1):
                        ic = idx[2][dc]; wc = idx[2][2] if dc else 1 - idx[2][2]
                        np.add.at(g, (ia, ib, ic), m * wa * wb * wc)
        return g

    from itertools import product
    g = np.zeros((K, K, K), dtype=np.float64)
    for flips in product((0, 1), repeat=3):
        wt = 1.0
        for a, fl in enumerate(flips):
            wt *= (1.0 - w_ax[a]) / 2.0 if fl else (1.0 + w_ax[a]) / 2.0
        if wt < 1e-9:
            continue
        Xs = X
        if any(flips):
            Xs = X.copy()
            for a, fl in enumerate(flips):
                if fl:
                    Xs[:, a] = -Xs[:, a]
        g += wt * bin_once(Xs)
    return (g / max(area, 1e-12)).ravel().astype(np.float32)
