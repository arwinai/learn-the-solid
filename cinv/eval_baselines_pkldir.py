"""AAGNet / UV-Net accuracy + macro mIoU on a perturbation-cell pkl directory.

Scores the baselines on a cell's AAGNet-format graphs so that every table
column comes from the same pipeline generation. Labels are the cell's
transferred y with rejected faces (y = -1) masked -- the identical support
every other model is scored on. Checkpoints default to the published AAGNet
weights and <CKPT>/uvnet_mfi.pth (override with AAG_CK / UV_CK); a missing
checkpoint skips that model. Learning environment (torch/dgl), with the AAGNet
and UV-Net checkouts on the path.

Usage: python eval_baselines_pkldir.py <cell_dir>
"""
from __future__ import annotations

import glob
import os
import pickle
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths                                            # noqa: E402
from train_baseline import make, fwd, aag_graph            # noqa: E402
from region_train import macro_miou                     # noqa: E402

DEV = torch.device("cuda")
CELL = sys.argv[1]

WEIGHTS = {
    "aagnet": os.environ.get(
        "AAG_CK", os.path.join(paths.AAGNET, "weights", "weight_on_MFInstseg.pth")),
    "uvnet": os.environ.get("UV_CK", os.path.join(paths.CKPT, "uvnet_mfi.pth")),
}

models = {}
for arch, ckp in WEIGHTS.items():
    if not os.path.exists(ckp):
        continue
    m = make(arch)
    sd = torch.load(ckp, map_location=DEV)
    m.load_state_dict(sd.get("model", sd.get("state_dict", sd)), strict=False)
    models[arch] = m.eval()

ys = {a: [] for a in models}
ps = {a: [] for a in models}
n_parts = {a: 0 for a in models}
with torch.no_grad():
    for fp in sorted(glob.glob(os.path.join(CELL, "*.pkl"))):
        d = pickle.load(open(fp, "rb"))
        y = np.asarray(d.get("y", []))
        aag = d.get("aag")
        if aag is None or len(y) == 0:
            continue
        ny = aag["graph"]["num_nodes"]
        if ny != len(y):
            continue
        g = aag_graph(aag, np.zeros(ny, np.int64))
        if g is None:
            continue
        g = g.to(DEV)
        vm = y >= 0          # identical support: rejected transfers masked
        if not vm.any():
            continue
        for arch, m in models.items():
            o = fwd(m, arch, g).argmax(-1).cpu().numpy()
            ys[arch].append(y[vm])
            ps[arch].append(o[vm])
            n_parts[arch] += 1

for arch in models:
    if not ys[arch]:
        print(f"SUMMARY baseline {arch} {os.path.basename(CELL)}: no parts")
        continue
    Y, P = np.concatenate(ys[arch]), np.concatenate(ps[arch])
    print(f"SUMMARY baseline {arch} {os.path.basename(CELL)}: "
          f"acc={float((Y == P).mean()):.4f} mIoU={macro_miou(Y, P):.4f} "
          f"n={n_parts[arch]}", flush=True)
