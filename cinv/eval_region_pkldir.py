"""Evaluate a RegionNet checkpoint on a directory of perturbation-cell pkls.

Each pkl carries a region graph (feats/src/dst/ef/f2r) and transferred face
labels y, with rejected transfers marked -1 and excluded from the metrics.
Prints face-level accuracy and macro mIoU plus a region-level mIoU (one
majority vote per region), which is insensitive to the face-count reweighting
that face splitting induces. Learning environment (torch/dgl).

Usage: python eval_region_pkldir.py <ckpt> <pkl_dir>
"""
import os, sys, glob, pickle, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, dgl
from region_train import RegionNet, macro_miou

DEV = torch.device("cuda")
ck = torch.load(sys.argv[1], map_location=DEV); cfg = ck.get("cfg", {})
edim = ck["model"]["convs.0.fc_edge.weight"].shape[1]
m = RegionNet(ck["in_dim"], cfg.get("d", 384), cfg.get("layers", 6),
              cfg.get("heads", 6), edge_dim=edim).to(DEV)
m.load_state_dict(ck["model"]); m.eval()
MU, SD = ck["mu"].to(DEV), ck["sd"].to(DEV)
ys, ps, yr_l, pr_l = [], [], [], []
with torch.no_grad():
    for fp in sorted(glob.glob(os.path.join(sys.argv[2], "*.pkl"))):
        p = pickle.load(open(fp, "rb"))
        g = dgl.graph((p["src"], p["dst"]), num_nodes=p["feats"].shape[0]).to(DEV)
        x = (torch.from_numpy(p["feats"]).float().to(DEV) - MU) / SD
        pr = m(g, x, torch.from_numpy(p["ef"]).float().to(DEV)).argmax(-1)
        f2r = torch.from_numpy(p["f2r"]).to(DEV)
        pf = pr[f2r]
        yv = np.asarray(p["y"])
        valid = yv >= 0        # label transfer marks ambiguous faces with
                               # y = -1; exclude them from the metrics
        ys.append(yv[valid]); ps.append(pf.cpu().numpy()[valid])
        # region-level: one vote per region, majority face label as truth --
        # immune to the face-count reweighting that splitting induces
        yy = yv; f2 = np.asarray(p["f2r"])
        for r in range(int(f2.max()) + 1):
            mk = (f2 == r) & valid
            if mk.any():
                yr_l.append(np.bincount(yy[mk]).argmax())
                pr_l.append(int(pr[r]))
y, p = np.concatenate(ys), np.concatenate(ps)
yr, prg = np.asarray(yr_l), np.asarray(pr_l)
print(f"SUMMARY dir={sys.argv[2]} n_faces={len(y)} acc={(y==p).mean():.4f} "
      f"mIoU={macro_miou(y, p):.4f}  region-mIoU={macro_miou(yr, prg):.4f} "
      f"(n_regions={len(yr)})", flush=True)
