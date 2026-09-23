"""Report label-transfer rejections per dataset x perturbation cell.

The evaluation scripts (eval_region_pkldir, eval_baselines_pkldir,
paired_churn_all, f360_paired_consistency) mask faces whose label transfer was
rejected (y / parent == -1) with identical support for every model. This
script reports the size of that excluded support, so that the effect of the
exclusion is visible alongside the metrics. Geometry-free (numpy only).

For every eval cell pkl dir this prints, per dataset x perturbation:
  parts          -- solids with a parent/y vector
  faces          -- total child faces
  rejected       -- faces with no unambiguous on-surface parent (y == -1)
  rej_face_pct   -- rejected / faces
  rej_area_pct   -- MEAN over parts of the within-part rejected-AREA
                    fraction (normalized areas from AAGNet face attr col 5)
  area%fw        -- the same fractions pooled with FACE-COUNT weights; NOT a
                    physical surface-area fraction (raw areas are not stored),
                    and it can move under face splitting at constant area
  parts_any_rej  -- solids containing at least one rejection

Usage: python report_rejections.py [suffix]   (default pfr_iso5a)
"""
from __future__ import annotations
import glob
import os
import pickle
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths

SUFFIX = sys.argv[1] if len(sys.argv) > 1 else "pfr_iso5a"
ROOTS = {ds: paths.dataset(ds)
         for ds in ("mfinstseg", "mfcadpp", "cadsynth", "f360seg")}
AREA_COL = 5   # graph_face_attr col 5 = face area (per-part normalization
               # cancels in fractions; correlation 1.000000 with GProp area)

print(f"{'dataset':10s} {'cell':26s} {'parts':>6s} {'faces':>8s} "
      f"{'rejected':>8s} {'rej_face%':>9s} {'rej_area%':>9s} "
      f"{'area%fw':>9s} {'parts_rej':>9s}")
print("-" * 100)
for ds, root in ROOTS.items():
    cells = sorted(glob.glob(os.path.join(root, f"eval_*_{SUFFIX}")))
    for cd in cells:
        cell = os.path.basename(cd)
        nP = nF = nR = nPR = 0
        fr_parts = []          # per-part rejected-area fractions
        a_rej = a_tot = 0.0    # pooled (per-part-normalized units)
        for fp in sorted(glob.glob(os.path.join(cd, "*.pkl"))):
            d = pickle.load(open(fp, "rb"))
            y = d.get("y")
            if y is None:
                continue
            y = np.asarray(y)
            rej = y < 0
            nP += 1
            nF += len(y)
            nR += int(rej.sum())
            nPR += int(rej.any())
            aag = d.get("aag")
            if aag is not None:
                fa = np.asarray(aag["graph_face_attr"], dtype=np.float64)
                if len(fa) == len(y):
                    ar = fa[:, AREA_COL]
                    tot = float(ar.sum())
                    if tot > 0:
                        fr = float(ar[rej].sum()) / tot
                        fr_parts.append(fr)
                        # pooled over parts in NORMALIZED units: weight each
                        # part's fraction by its face count (raw areas are not
                        # stored, so a true area-weighted pool is unavailable)
                        a_rej += fr * len(ar)
                        a_tot += float(len(ar))
        if nP == 0:
            continue
        area_mean = 100 * np.mean(fr_parts) if fr_parts else float("nan")
        area_pool = 100 * a_rej / a_tot if a_tot else float("nan")   # face-weighted
        print(f"{ds:10s} {cell:26s} {nP:6d} {nF:8d} {nR:8d} "
              f"{100*nR/max(nF,1):8.3f}% {area_mean:8.3f}% "
              f"{area_pool:8.3f}% {nPR:9d}", flush=True)
