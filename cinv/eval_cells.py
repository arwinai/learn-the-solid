"""Evaluate every model on an arbitrary list of perturbation cells.

The nuisance suite has several cells (re-partition families, rotation, spline
re-expression, compositions) and not all of them exist for every dataset, so
this takes the cell directories as arguments and reports each model on exactly
the directories given. Every model is scored on the same pkl graphs, so all of
them see identical solids and identical transferred labels; a missing model
prints '--' rather than being silently dropped. Our model is reported as the
mean and standard deviation over the seed checkpoints present in <CKPT>.
Learning environment (torch/dgl), with the baseline checkouts on the path.

Usage:
  python eval_cells.py <dataset-slug> <cell-dir> [<cell-dir> ...]
"""
from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import paths  # noqa: E402
from eval_matrix import eval_base, eval_region, make  # noqa: E402

DATA = paths.DATA
AAG = paths.AAGNET


def region_ckpts(slug):
    """The three seeds of our primary model for this dataset, if present."""
    out = []
    for suf in ("", "_s1", "_s2"):
        p = paths.ckpt(f"region_{slug}_pfr_iso4{suf}")
        if os.path.exists(p):
            out.append(p)
    return out


def baseline_ckpt(arch, slug):
    if arch == "aagnet":
        if slug == "mfinstseg":
            return os.path.join(AAG, "weights", "weight_on_MFInstseg.pth")
        for t in (f"aagnet_{slug}", f"aagnet_{slug}_seed1"):
            p = os.path.join(paths.CKPT, f"{t}.pth")
            if os.path.exists(p):
                return p
    if arch == "uvnet":
        for t in (f"uvnet_{slug}", "uvnet_mfi"):
            p = os.path.join(paths.CKPT, f"{t}.pth")
            if os.path.exists(p):
                return p
    return None


def main(slug, cells):
    root = os.path.join(DATA, slug)
    te = None
    tp = os.path.join(root, "test.txt")
    if os.path.exists(tp):
        te = [l.strip() for l in open(tp) if l.strip()]

    names = [os.path.basename(c).replace("eval_", "").replace("_pfr_iso4", "")
             .replace("_pfr_iso5", "") for c in cells]
    print(f"\n{slug}: {len(cells)} cell(s)")
    print(f"{'model':12s} " + " ".join(f"{n:>10s}" for n in names))
    print("-" * (13 + 11 * len(names)))

    for arch in ("aagnet", "uvnet"):
        ck = baseline_ckpt(arch, slug)
        if ck is None:
            print(f"{arch:12s} " + " ".join(f"{'--':>10s}" for _ in cells))
            continue
        r = eval_base(arch, ck, cells, te)
        print(f"{arch:12s} " + " ".join(
            f"{(v if v is not None else float('nan')):10.4f}" for v in r))

    cks = region_ckpts(slug)
    if not cks:
        print(f"{'ours':12s} " + " ".join(f"{'--':>10s}" for _ in cells))
        return
    per_seed = [eval_region(c, cells, te) for c in cks]
    arr = np.array([[np.nan if v is None else v for v in row]
                    for row in per_seed], dtype=float)
    mean, sd = np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)
    print(f"{'ours':12s} " + " ".join(f"{m:10.4f}" for m in mean))
    print(f"{'  (sd)':12s} " + " ".join(f"{s:10.4f}" for s in sd)
          + f"   over {len(cks)} seeds")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(2)
    main(sys.argv[1], sys.argv[2:])
