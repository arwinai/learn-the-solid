"""Reproducible per-part draws for the random nuisance cells.

The rotation applied to a part is seeded by crc32 of its name.  Unlike
Python's per-process salted `str.__hash__`, crc32 is stable across processes,
runs and machines, so

  * a cell is reproducible: re-running a script on the same parts applies the
    same rotations, and any number can be regenerated exactly;
  * every model is compared on the same solids: the region-graph extractor
    (rot_extract.py) and the STEP emitter that feeds BRepNet
    (emit_nuisance_steps.py) run as separate processes yet draw the identical
    rotation for a given part, which is what a paired consistency measurement
    requires.

Usage: from nuisance_seed import rot_params; axis, angle = rot_params(name)
"""
from __future__ import annotations

import math
from zlib import crc32

import numpy as np


def stable_seed(name):
    """A process-independent 32-bit seed for a part name."""
    return crc32(str(name).encode("utf-8")) & 0xFFFFFFFF


def rot_params(name, min_angle=0.2):
    """The (axis, angle) drawn for `name`: unit axis, angle in [min_angle, pi].

    Same name -> same rotation, in any process. Callers must not reseed.
    """
    rng = np.random.default_rng(stable_seed(name))
    v = rng.normal(size=3)
    v /= np.linalg.norm(v)
    return v, float(rng.uniform(min_angle, math.pi))
