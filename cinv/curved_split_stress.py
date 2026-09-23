"""Stress test: are the region features invariant when a CURVED face is split?

Several features (int n dA, int n(x)n dA, int H dA, bbox) are evaluated by
quadrature over a triangulation. For a PLANAR face the integrand is constant, so
quadrature is exact and invariance is free. For a CURVED face it is not, and
construction-variant fixtures whose curved faces happen not to be split between
the two constructions do not exercise this. This script splits cylindrical and
spherical faces explicitly and reports the residual per feature group.

Usage: python curved_split_stress.py   (geometry environment; export the
       paper's feature flags first -- the column layout assumes them)
"""
from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from region_graph import build, face_list
from repartition_testset import split_faces
from fixtures import block_with_hole, ring_pair, props

from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeSphere
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.gp import gp_Ax2, gp_Pnt, gp_Dir

GROUPS = [("type+area+centroid+axis+params", 0, 16),
          ("principal moments (GProp, exact)", 16, 19),
          ("mean normal  int n dA", 20, 23),
          ("orientation  int n(x)n dA", 23, 29),
          ("curvature    int H,K,H2 dA", 29, 32),
          ("loops/blen/bsharp", 32, 35),
          ("boundary convexity shares", 35, 37),
          ("second moments (GProp, exact)", 37, 43),
          ("dihedral hist (BHIST)", 43, 51),
          ("canon grid (BHIST+CG layout)", 51, 115)]


def cyl_shapes():
    yield "block_with_hole", block_with_hole()
    yield "ring", ring_pair(1.0, 0.4, 0.5)[0]
    yield "plain_cylinder", BRepPrimAPI_MakeCylinder(
        gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), 0.7, 1.2).Shape()
    yield "sphere_minus_cyl", BRepAlgoAPI_Cut(
        BRepPrimAPI_MakeSphere(1.0).Shape(),
        BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, -2), gp_Dir(0, 0, 1)), 0.3, 4).Shape()
    ).Shape()


print(f"{'shape':18s} {'faces':>12s} {'regions':>10s}   worst residual by feature group")
print("-" * 100)
for nm, sh in cyl_shapes():
    rp = split_faces(sh, 1.0, 0)
    if rp is None:
        print(f"{nm:18s} split failed"); continue
    v0, a0 = props(sh); v1, a1 = props(rp)
    if v0 <= 0 or abs(v1 - v0) / v0 > 1e-9 or abs(a1 - a0) / a0 > 1e-9:
        print(f"{nm:18s} SKIP: geometry changed dV={abs(v1-v0)/v0:.1e} dA={abs(a1-a0)/a0:.1e}")
        continue
    ga, gb = build(sh), build(rp)
    if ga is None or gb is None:
        print(f"{nm:18s} no graph"); continue
    fa, fb = ga[1], gb[1]
    if fa.shape != fb.shape:
        print(f"{nm:18s} REGION COUNT DIFFERS {fa.shape[0]} vs {fb.shape[0]}")
        continue
    # Hungarian matching on ALL feature columns: symmetric twin regions (e.g.
    # a cylinder's two caps) can be identical in any prefix of columns, and a
    # greedy argmin then pairs both twins with the same row, reporting a huge
    # fake residual where the graphs actually agree as multisets.
    from scipy.optimize import linear_sum_assignment
    D = ((fa[:, None, :] - fb[None, :, :]) ** 2).sum(-1)
    ri, ci = linear_sum_assignment(D)
    diff = np.abs(fa[ri] - fb[ci])
    print(f"{nm:18s} {len(face_list(sh)):5d}->{len(face_list(rp)):<5d} "
          f"{fa.shape[0]:10d}")
    for gname, lo, hi in GROUPS:
        d = float(diff[:, lo:hi].max())
        flag = "  <-- NOT INVARIANT" if d > 1e-6 else ""
        print(f"    {gname:34s} {d:.3e}{flag}")
