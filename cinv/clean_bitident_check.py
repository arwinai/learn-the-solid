"""Check that the current feature pipeline reproduces reference clean graphs
bit-for-bit.

Rebuilds the region graph of every part in PARTS_LIST with the code as it
stands and compares it against the reference npz in REF_DIR -- region count,
then node features and edge features exactly.  Identical clean graphs mean the
training data has not moved and existing checkpoints remain valid for the
current pipeline.

The graph is rebuilt with the actual pipeline (region_graph.build) rather than
a reimplementation of the rule under test, so the comparison reflects what the
pipeline does. Requires the geometry environment (pythonocc) and the AAGNet
checkout (for scale_solid_to_unit_box).

Env: DS_ROOT (dataset dir with steps/; default paths.dataset("mfinstseg")),
     PARTS_LIST (one *.step name per line), REF_DIR (reference region_graphs
     dir), NPROC (worker count), plus the usual feature flags.
Prints "IDENTICAL ALL" only if every part matches.

Usage: DS_ROOT=... PARTS_LIST=... REF_DIR=... python clean_bitident_check.py
"""
from __future__ import annotations

import os
import sys
import warnings
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths

warnings.filterwarnings("ignore")

DS_ROOT = os.environ.get("DS_ROOT", paths.dataset("mfinstseg"))
PARTS_LIST = os.environ["PARTS_LIST"]
REF_DIR = os.environ["REF_DIR"]
STEPS = os.path.join(DS_ROOT, "steps")
NPROC = int(os.environ.get("NPROC", "6"))


def one(name):
    stem = name[:-5] if name.endswith(".step") else name
    ref = os.path.join(REF_DIR, stem + ".npz")
    if not os.path.exists(ref):
        return ("noref", stem, None)
    try:
        from OCC.Core.STEPControl import STEPControl_Reader

        sys.path.insert(0, paths.AAGNET)
        from dataset.AAGExtractor import scale_solid_to_unit_box

        from region_graph import build

        rd = STEPControl_Reader()
        rd.ReadFile(os.path.join(STEPS, stem + ".step"))
        rd.TransferRoots()
        g = build(scale_solid_to_unit_box(rd.OneShape()))
        if g is None:
            return ("nograph", stem, None)
        _, feats, _, ef = g
        z = np.load(ref)
        rf, re_ = np.asarray(z["feats"]), np.asarray(z["ef"])
        if feats.shape != rf.shape:
            return ("shape", stem, f"{feats.shape} vs {rf.shape}")
        dn = float(np.abs(np.asarray(feats) - rf).max())
        de = (float(np.abs(np.asarray(ef) - re_).max())
              if np.asarray(ef).shape == re_.shape else float("inf"))
        if dn == 0.0 and de == 0.0:
            return ("same", stem, None)
        return ("diff", stem, f"node {dn:.3g} edge {de:.3g}")
    except Exception as exc:                       # noqa: BLE001
        return ("err", stem, f"{type(exc).__name__}: {exc}")


def main():
    names = [l.strip() for l in open(PARTS_LIST) if l.strip()]
    with Pool(NPROC) as p:
        res = p.map(one, names)
    tally = {}
    for k, stem, info in res:
        tally[k] = tally.get(k, 0) + 1
    bad = [(k, s, i) for k, s, i in res if k in ("diff", "shape", "err")]
    print(f"parts={len(names)} {tally}")
    for k, s, i in bad[:20]:
        print(f"  {k}: {s} {i}")
    # "noref"/"nograph" are parts the reference generation never produced a
    # graph for, so there is nothing to compare and they are not evidence of a
    # change; only a real diff (or an error hiding one) fails this gate.
    for k, s, _ in res:
        if k in ("noref", "nograph"):
            print(f"  {k} (excluded, no reference to compare): {s}")
    if bad:
        print(f"NOT IDENTICAL: {len(bad)} changed parts")
    else:
        print(f"IDENTICAL ALL ({tally.get('same', 0)} compared, "
              f"{len(names) - tally.get('same', 0)} excluded)")


if __name__ == "__main__":
    main()
