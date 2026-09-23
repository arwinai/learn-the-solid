"""Build a BRepNet dataset.json for a perturbation-cell evaluation.

The output keeps the clean standardization (the normalization the model was
trained under) and sets the test list to the cell stems that have both
features (npz) and labels (.seg) and whose face counts agree.

The count check exists because a face-count mismatch between a .seg file and
its npz does not fail at load time; it raises an IndexError
(labels[new_to_old]) inside BRepNet's dataloader mid-epoch and aborts the
evaluation. Composed cells can produce a few such parts (the label transfer
counts faces on OCC's read of the composed STEP, while BRepNet's extractor
occasionally enumerates a different count on the same file). Mismatches are
dropped and written to <rp_dir>/label_mismatch.json, mirroring
brepnet_prep.py.

Usage: python brepnet_rp_dataset.py <clean_dataset.json> <rp_npz_dir> <out.json>
"""
import glob
import json
import os
import sys

import numpy as np

clean_json, rp_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(clean_json))

stems, dropped = [], []
for f in sorted(glob.glob(os.path.join(rp_dir, "*.npz"))):
    nm = os.path.basename(f)[:-4]
    seg = os.path.join(rp_dir, nm + ".seg")
    if not os.path.exists(seg):
        continue
    try:
        with np.load(f) as z:
            nf = int(z["face_features"].shape[0])
        ny = sum(1 for line in open(seg) if line.strip())
    except Exception:
        dropped.append(nm)
        continue
    (stems if nf == ny else dropped).append(nm)

d["test_set"] = stems
for k in ("training_set", "validation_set"):
    if k in d:
        d[k] = d[k][:2]     # minimal non-empty lists; unused by test but loaded
json.dump(d, open(out, "w"))
print(f"crafted {out}: test={len(stems)}, dropped {len(dropped)} "
      f"count-mismatched or unreadable", flush=True)
if dropped:
    json.dump(dropped, open(os.path.join(rp_dir, "label_mismatch.json"), "w"))
