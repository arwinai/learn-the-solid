"""Per-class IoU for a saved RegionNet checkpoint on a dataset root's test split.

Prints per-class IoU with support counts alongside the macro mean, in the
dataset's own label order, so that a gap can be attributed to specific
classes. Class names are only labels for reading; the row order is the label
integers the graphs carry, and the support column is the check that the naming
lines up. Learning environment (torch/dgl).

Usage: python eval_perclass.py <ckpt> <root> [dataset]
       dataset selects the name list (currently "f360seg"); omit for bare indices.
"""
from __future__ import annotations
import os
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from region_train import (N_CLASSES, RegionDataset, RegionNet, collate,
                             macro_miou)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Label order is the dataset's own, imported from f360seg_prepare.CLASSES so
# that the names stay aligned with the label integers; the support counts
# printed below make any misalignment visible.
from f360seg_prepare import CLASSES as F360_CLASSES


NAMES = {"f360seg": F360_CLASSES}


@torch.no_grad()
def main(ckpt, root, ds=None):
    c = torch.load(ckpt, map_location=DEV)
    cfg, in_dim = c.get("cfg", {}), c["in_dim"]
    edim = c["model"]["convs.0.fc_edge.weight"].shape[1]
    m = RegionNet(in_dim, cfg.get("d", 384), cfg.get("layers", 6),
                  cfg.get("heads", 6), edge_dim=edim).to(DEV)
    m.load_state_dict(c["model"])
    m.eval()
    mu, sd = c["mu"].to(DEV), c["sd"].to(DEV)
    dl = torch.utils.data.DataLoader(RegionDataset("test", root), 256,
                                     num_workers=6, collate_fn=collate)
    ys, ps = [], []
    for g, f2r, y in dl:
        g = g.to(DEV)
        x = (g.ndata["x"] - mu) / sd
        ps.append(m(g, x, g.edata["e"]).argmax(-1)[f2r.to(DEV)].cpu().numpy())
        ys.append(y.numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    names = NAMES.get(ds or "", [str(i) for i in range(N_CLASSES)])
    print(f"PERCLASS ckpt={ckpt} root={root} n_faces={len(y)} "
          f"acc={(y == p).mean():.4f} mIoU={macro_miou(y, p):.4f}")
    for k in range(N_CLASSES):
        t, q = (y == k), (p == k)
        u = (t | q).sum()
        if u == 0:
            continue
        nm = names[k] if k < len(names) else str(k)
        print(f"  {k} {nm:<12s} n={int(t.sum()):7d}  "
              f"IoU={(t & q).sum() / u:.4f}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
