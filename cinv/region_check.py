"""Two checks on the canonical region graph.

1. INVARIANCE  : do two constructions of the same solid give the same graph?
                 Measured on the synthetic fixture pairs (ring, slab-with-hole,
                 L-profile) with Hungarian matching of region features.
2. LABEL PURITY: on real MFInstSeg parts, do all faces merged into one region
                 share a label?  Impure regions lose supervision under
                 majority vote, which bounds the achievable per-face accuracy.

Requires the geometry environment (pythonocc). Reads MFInstSeg test parts from
paths.dataset("mfinstseg") (test.txt, steps/, labels/) and prints both
summaries to stdout.

Usage: python region_check.py
"""
from __future__ import annotations
import os, sys, json, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths
from region_graph import build, face_list
from fixtures import ring_pair, slab_hole_pair, lprofile_pair, props

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone


ROOT = paths.dataset("mfinstseg")


def sig(feats, edges, ef):
    """Order-free signature of the whole graph."""
    n = sorted(tuple(np.round(r, 5)) for r in feats)
    e = sorted(tuple(np.round(r, 5)) for r in ef)
    return (n, e)


print("=== 1. invariance across construction strategies ===")
rng = np.random.default_rng(0)
ok = tot = 0; worst = 0.0
for i in range(8):
    R = rng.uniform(0.8, 1.4); r = rng.uniform(0.25, 0.55); h = rng.uniform(0.3, 0.9)
    for name, (A, B) in [("ring", ring_pair(R, r, h)),
                         ("slab_hole", slab_hole_pair(2*R, 1.6*R, h, r)),
                         ("Lprofile", lprofile_pair(1.5*R, 1.2*R, 0.3*R, h))]:
        va, aa = props(A); vb, ab = props(B)
        if va <= 0 or abs(va-vb)/va > 1e-9 or abs(aa-ab)/aa > 1e-9:
            continue
        ga, gb = build(A), build(B)
        if ga is None or gb is None:
            continue
        tot += 1
        match = ga[1].shape == gb[1].shape
        resid = float("inf")
        if match:
            # Hungarian matching on feature distance: twin symmetric regions
            # can tie in any column prefix, so order-based pairing is wrong.
            from scipy.optimize import linear_sum_assignment
            D = ((ga[1][:, None, :] - gb[1][None, :, :]) ** 2).sum(-1)
            ri, ci = linear_sum_assignment(D)
            resid = float(np.abs(ga[1][ri] - gb[1][ci]).max())
            worst = max(worst, resid)
        ok += match
        if i < 2:
            print(f"  {name:10s} faces {len(face_list(A)):3d}/{len(face_list(B)):<3d} "
                  f"regions {ga[1].shape[0]:3d}/{gb[1].shape[0]:<3d} "
                  f"{'MATCH' if match else 'DIFFER'} resid={resid:.2e}")
print(f"  -> {ok}/{tot} identical region structure, worst matched residual {worst:.3e}")


print("\n=== 2. label purity on real MFInstSeg parts ===")
names = [l.strip() for l in open(os.path.join(ROOT, "test.txt")) if l.strip()][:300]
n_done = n_face = n_region = 0
impure_regions = 0
lost_faces = 0
merged_hist = {}
for nm in names:
    sp = os.path.join(ROOT, "steps", nm + ".step")
    lp = os.path.join(ROOT, "labels", nm + ".json")
    if not (os.path.exists(sp) and os.path.exists(lp)):
        continue
    rd = STEPControl_Reader()
    if rd.ReadFile(sp) != IFSelect_RetDone:
        continue
    rd.TransferRoots()
    shape = rd.OneShape()
    g = build(shape)
    if g is None:
        continue
    f2r, feats = g[0], g[1]
    lab = json.load(open(lp))[0][1]["seg"]
    if len(lab) != len(f2r):
        continue                                   # face order mismatch -> skip
    y = np.array([lab[str(i)] for i in range(len(f2r))])
    Rn = feats.shape[0]
    for r in range(Rn):
        ys = y[f2r == r]
        c = len(ys)
        merged_hist[c] = merged_hist.get(c, 0) + 1
        if len(set(ys.tolist())) > 1:
            impure_regions += 1
            # majority vote -> how many face labels would be wrong
            vals, cnts = np.unique(ys, return_counts=True)
            lost_faces += c - cnts.max()
    n_done += 1; n_face += len(f2r); n_region += Rn

print(f"  parts processed         : {n_done}")
print(f"  faces {n_face}  ->  regions {n_region}   "
      f"(compression {n_face/max(n_region,1):.3f}x)")
print(f"  regions merging >1 face : {sum(v for k,v in merged_hist.items() if k>1)}"
      f" / {n_region}")
print(f"  IMPURE regions          : {impure_regions} ({100*impure_regions/max(n_region,1):.3f}%)")
print(f"  faces whose label is unrecoverable after merging: {lost_faces}"
      f" ({100*lost_faces/max(n_face,1):.3f}%)")
print(f"  => CEILING on per-face accuracy = {100*(1-lost_faces/max(n_face,1)):.3f}%")
print(f"  merge-size histogram    : {dict(sorted(merged_hist.items())[:8])}")
