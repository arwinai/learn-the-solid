"""Extract BRepNet's test metrics from a run as one machine-readable line.

BRepNet's `eval/test.py` reports only through PyTorch Lightning's printed
metric table, and `train/train.py` additionally writes `test_results.json`.
Both formats are parsed here so that downstream scripts never have to grep a
box-drawing table, and so that a missing number is an explicit `nan` rather
than an empty string.

Input: a BRepNet stdout log or test_results.json. Exit status is 1 when no
metric could be parsed.

Usage:
  python bn_parse_metrics.py <log-or-json> [<label>]
Prints:
  <label> accuracy=<float> mean_iou=<float>
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

KEYS = ("test/accuracy", "test/mean_iou")


def from_json(p):
    d = json.load(open(p))
    rows = d.get("results", d) if isinstance(d, dict) else d
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        rows = rows[0]
    if not isinstance(rows, dict):
        return {}
    return {k: float(rows[k]) for k in KEYS if k in rows}


def from_log(p):
    txt = Path(p).read_text(errors="replace")
    out = {}
    for k in KEYS:
        # the metric name and its value share a line in Lightning's table
        m = re.findall(re.escape(k) + r"[^0-9\-]*(-?[0-9]*\.?[0-9]+)", txt)
        if m:
            out[k] = float(m[-1])          # last = final epoch's test pass
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    p = Path(sys.argv[1])
    label = sys.argv[2] if len(sys.argv) > 2 else p.name
    got = {}
    if p.exists():
        try:
            got = from_json(p) if p.suffix == ".json" else from_log(p)
        except Exception as e:
            print(f"{label} parse_error={type(e).__name__}")
            return 1
    acc = got.get("test/accuracy", float("nan"))
    iou = got.get("test/mean_iou", float("nan"))
    print(f"{label} accuracy={acc:.4f} mean_iou={iou:.4f}")
    return 0 if got else 1


if __name__ == "__main__":
    raise SystemExit(main())
