"""Evaluate a saved RegionNet checkpoint on a dataset root's test split.

Loads the checkpoint written by region_train.py (model weights, normalisation
statistics, configuration), rebuilds the network with the edge-feature width
stored in the weights, and prints face-level accuracy and macro mIoU on
<root>/test.txt. Learning environment (torch/dgl).

Usage: python eval_region_ckpt.py <ckpt> <root>
"""
from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from region_train import RegionNet, RegionDataset, collate, macro_miou


DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def main(ckpt, root):
    c = torch.load(ckpt, map_location=DEV)
    cfg, in_dim = c.get("cfg", {}), c["in_dim"]
    edim = c["model"]["convs.0.fc_edge.weight"].shape[1]
    m = RegionNet(in_dim, cfg.get("d", 384), cfg.get("layers", 6),
                  cfg.get("heads", 6), edge_dim=edim).to(DEV)
    m.load_state_dict(c["model"]); m.eval()
    mu, sd = c["mu"].to(DEV), c["sd"].to(DEV)
    te = RegionDataset("test", root)
    dl = torch.utils.data.DataLoader(te, 256, num_workers=6, collate_fn=collate)
    accs, ys, ps = [], [], []
    for g, f2r, y in dl:
        g = g.to(DEV)
        x = (g.ndata["x"] - mu) / sd
        pf = m(g, x, g.edata["e"]).argmax(-1)[f2r.to(DEV)].cpu().numpy()
        ys.append(y.numpy()); ps.append(pf)
    y, p = np.concatenate(ys), np.concatenate(ps)
    print(f"SUMMARY ckpt={ckpt} root={root} test acc={(y==p).mean():.4f} "
          f"mIoU={macro_miou(y, p):.4f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
