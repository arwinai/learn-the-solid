"""Region-graph agreement on synthetic fixture pairs, measured with the same
protocol as graph_agreement.py (sigma units, exact/sampled families,
lexicographic matching) so the fixture rows are comparable with the
benchmark-scale rows.

Two tiers:
  constr. fixtures -- 24 construction-variant pairs (ring / slab-with-hole /
                      L-profile at 8 random size draws), the same generator
                      pairs region_check.py uses;
  curved-split     -- 4 hand-built solids whose curved faces are cut with
                      curved_split_stress.split_faces.

Requires the geometry environment (pythonocc). Prints per-tier pair counts and
p50/p90/p99/max residuals of the exact and sampled feature families.

Usage: PART_FRAME=1 ... python fixture_agreement.py
"""
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeSphere
from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt

from fixtures import (block_with_hole, lprofile_pair, props,
                      ring_pair, slab_hole_pair)
from curved_split_stress import split_faces
from graph_agreement import compare
from region_graph import build


def constr_pairs():
    rng = np.random.default_rng(0)
    for _ in range(8):
        R = rng.uniform(0.8, 1.4)
        r = rng.uniform(0.25, 0.55)
        h = rng.uniform(0.3, 0.9)
        yield "ring", ring_pair(R, r, h)
        yield "slab_hole", slab_hole_pair(2 * R, 1.6 * R, h, r)
        yield "Lprofile", lprofile_pair(1.5 * R, 1.2 * R, 0.3 * R, h)


def curved_pairs():
    shapes = [("block_with_hole", block_with_hole()),
              ("ring", ring_pair(1.0, 0.4, 0.5)[0]),
              ("plain_cylinder", BRepPrimAPI_MakeCylinder(
                  gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), 0.7, 1.2).Shape()),
              ("sphere_minus_cyl", BRepAlgoAPI_Cut(
                  BRepPrimAPI_MakeSphere(1.0).Shape(),
                  BRepPrimAPI_MakeCylinder(
                      gp_Ax2(gp_Pnt(0, 0, -2), gp_Dir(0, 0, 1)),
                      0.3, 4).Shape()).Shape())]
    for nm, sh in shapes:
        rp = split_faces(sh, 1.0, 0)
        if rp is not None:
            yield nm, (sh, rp)


def run(tier, gen):
    ex, sa, n_pair, n_same = [], [], 0, 0
    for nm, (A, B) in gen:
        va, aa = props(A)
        vb, ab = props(B)
        if va <= 0 or abs(va - vb) / va > 1e-9 or abs(aa - ab) / aa > 1e-9:
            print(f"  {tier}/{nm}: SKIP, geometry differs")
            continue
        ga, gb = build(A), build(B)
        if ga is None or gb is None:
            print(f"  {tier}/{nm}: SKIP, no graph")
            continue
        n_pair += 1
        eq, e, s = compare(np.asarray(ga[1]), np.asarray(gb[1]))[:3]
        if not eq:
            print(f"  {tier}/{nm}: REGION COUNT DIFFERS "
                  f"{ga[1].shape[0]} vs {gb[1].shape[0]}")
            continue
        n_same += 1
        ex.append(e)
        sa.append(s)

    def pcts(x):
        x = np.array(x)
        return "/".join([f"{np.percentile(x, q):.3g}" for q in (50, 90, 99)]
                        + [f"{x.max():.3g}"])

    print(f"{tier}: N={n_pair} identical={n_same}")
    print(f"  exact   p50/p90/p99/max = {pcts(ex)}")
    print(f"  sampled p50/p90/p99/max = {pcts(sa)}")


if __name__ == "__main__":
    run("constr. fixtures", constr_pairs())
    run("curved-split", curved_pairs())
