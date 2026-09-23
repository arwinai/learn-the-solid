"""Write BRepNet .seg label files + train_test.json for a dataset.

Face ordering: BRepNet's EntityMapper enumerates faces in plain TopExp order,
identical to the label indexing used here (face areas agree to 1e-14; a
shuffled control disagrees by 5 orders of magnitude).

Face counts are checked per part: on some parts (observed on CADSynth)
BRepNet's extraction has more faces than the label file, and such a part
raises an IndexError (labels[new_to_old]) inside BRepNet's dataloader
mid-epoch. Mismatched parts are excluded and recorded in label_mismatch.json,
the same disclosure as intake failures, rather than written.

Modes:
  clean: DS_ROOT + NPZ_DIR -> .seg next to npz from dataset labels,
         train_test.json from {train,test}.txt (only stems with an npz)
  rp:    NPZ_DIR + RP_Y (repart pkl dir with transferred 'y') -> .seg only

Env: NPZ_DIR (BRepNet npz dir, required), DS_ROOT (dataset root; default
paths.dataset("mfinstseg")), RP_Y (selects rp mode).

Usage: NPZ_DIR=... [DS_ROOT=... | RP_Y=...] python brepnet_prep.py
"""
from __future__ import annotations
import os, sys, json, glob, pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths

ROOT = os.environ.get("DS_ROOT", paths.dataset("mfinstseg"))
NPZ = os.environ["NPZ_DIR"]
RP_Y = os.environ.get("RP_Y")


def npz_face_count(nm):
    try:
        with np.load(os.path.join(NPZ, nm + ".npz")) as d:
            return int(d["face_features"].shape[0])
    except Exception:
        return -1


def main():
    stems = {os.path.basename(f)[:-4] for f in glob.glob(os.path.join(NPZ, "*.npz"))}
    n, mismatched = 0, []
    if RP_Y:
        for fp in glob.glob(os.path.join(RP_Y, "*.pkl")):
            nm = os.path.basename(fp)[:-4]
            if nm not in stems:
                continue
            y = np.asarray(pickle.load(open(fp, "rb"))["y"], np.int64)
            if len(y) != npz_face_count(nm):
                mismatched.append(nm)
                continue
            np.savetxt(os.path.join(NPZ, nm + ".seg"), y, fmt="%d")
            n += 1
        print(f"[prep] rp: wrote {n} seg files in {NPZ}, "
              f"excluded {len(mismatched)} face-count mismatches", flush=True)
        if mismatched:
            json.dump(mismatched, open(os.path.join(NPZ, "label_mismatch.json"), "w"))
        return
    import pc_extract as pe
    pe.ROOT = ROOT
    tt = {}
    for split, key in (("train", "train"), ("val", "train"), ("test", "test")):
        try:
            jobs = pe.jobs_split(split)
        except FileNotFoundError:
            continue
        keep = []
        for nm, _sp, y in jobs:
            if nm not in stems:
                continue
            y = np.asarray(y, np.int64)
            if len(y) != npz_face_count(nm):
                mismatched.append(nm)
                continue
            np.savetxt(os.path.join(NPZ, nm + ".seg"), y, fmt="%d")
            keep.append(nm); n += 1
        tt.setdefault(key, []).extend(keep)
    json.dump(tt, open(os.path.join(NPZ, "train_test.json"), "w"))
    print(f"[prep] clean: wrote {n} seg files, split sizes "
          f"{ {k: len(v) for k, v in tt.items()} }, "
          f"excluded {len(mismatched)} face-count mismatches", flush=True)
    if mismatched:
        json.dump(mismatched, open(os.path.join(NPZ, "label_mismatch.json"), "w"))


if __name__ == "__main__":
    main()
