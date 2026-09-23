"""Getting a single solid out of a STEP file or a constructed shape.

Both OpenCASCADE's STEP reader and FreeCAD's exporter routinely hand back a
COMPOUND that wraps one solid, and the shared normalisation used by this code
and by the AAGNet baseline (`occwl.Solid`) asserts a `TopoDS_Solid`. Every
entry point therefore needs the same unwrap, so it lives here once.

`as_solid` deliberately refuses a compound holding several solids rather than
silently returning the first. A part built as several disjoint bodies is not
one solid, and quietly taking one of them would compare different objects; the
intended handling is to evaluate each body separately and pool the per-face
results (see fc_export_bodies.py).
"""
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_SOLID
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import TopoDS_Solid, topods


def solids(shape):
    """Every solid in `shape`, in traversal order."""
    out, ex = [], TopExp_Explorer(shape, TopAbs_SOLID)
    while ex.More():
        out.append(topods.Solid(ex.Current()))
        ex.Next()
    return out


def as_solid(shape, allow_multi=False):
    """The single solid in `shape`.

    Raises ValueError when there is none, or when there are several and
    `allow_multi` is False -- see the module docstring for why that is not a
    silent first-solid pick.
    """
    if isinstance(shape, TopoDS_Solid):
        return shape
    found = solids(shape)
    if not found:
        raise ValueError("shape contains no solid (open shell or empty result)")
    if len(found) > 1 and not allow_multi:
        raise ValueError(f"shape contains {len(found)} solids; evaluate them "
                         "separately and pool, or pass allow_multi=True")
    return found[0]


def load_step(path, allow_multi=False):
    """Read a STEP file and return its single solid."""
    rd = STEPControl_Reader()
    rd.ReadFile(path)
    rd.TransferRoots()
    return as_solid(rd.OneShape(), allow_multi=allow_multi)
