"""Check that analytic surface recognition is stable under NurbsConvert.

For each part, the surface-type census (from surface_graph.surface_key) of the
clean solid is compared with that of its NurbsConvert twin. The two must match,
because conversion changes the surface representation, not the geometry. The
check calls the pipeline's own surface_key on the converted solids rather than
re-implementing the fits. Requires the geometry environment (pythonocc) and the
AAGNet checkout (for scale_solid_to_unit_box); reads STEP files from
paths.dataset("mfinstseg")/steps.

Usage: python verify_exactfit.py <part> [<part> ...]      (MFInstSeg names)
"""
from __future__ import annotations

import os
import sys
import warnings
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths

warnings.filterwarnings("ignore")

from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_NurbsConvert
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer

sys.path.insert(0, paths.AAGNET)
from dataset.AAGExtractor import scale_solid_to_unit_box

from surface_graph import surface_key


STEPS = os.path.join(paths.dataset("mfinstseg"), "steps")


def census(shape):
    c = Counter()
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        k = surface_key(ex.Current())
        c[k[0] if isinstance(k, tuple) else "other"] += 1
        ex.Next()
    return c


def main(parts):
    bad = 0
    for nm in parts:
        rd = STEPControl_Reader()
        rd.ReadFile(f"{STEPS}/{nm}.step")
        rd.TransferRoots()
        raw = rd.OneShape()
        a = census(scale_solid_to_unit_box(raw))
        b = census(scale_solid_to_unit_box(
            BRepBuilderAPI_NurbsConvert(raw, True).Shape()))
        ok = a == b
        bad += 0 if ok else 1
        print(f"{nm}: {'MATCH' if ok else 'MISMATCH'}")
        print(f"  clean     {dict(sorted(a.items()))}")
        print(f"  converted {dict(sorted(b.items()))}")
        if not ok:
            for k in sorted(set(a) | set(b)):
                if a.get(k, 0) != b.get(k, 0):
                    print(f"    {k}: clean {a.get(k, 0)} vs "
                          f"converted {b.get(k, 0)}")
    print("EXACTFIT OK" if bad == 0 else f"EXACTFIT FAILED on {bad} part(s)")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
