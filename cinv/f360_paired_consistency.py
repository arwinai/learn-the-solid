"""Paired prediction consistency on Fusion 360 segmentation (F360seg).

Compares a RegionNet checkpoint's predictions on the clean and re-partitioned
graphs of the SAME solid, mapping child faces back to their parents. This is
the invariance metric that is immune to the face-granularity reweighting that
affects mIoU on re-partitioned sets. Rejected transfers (parent = -1) are
masked. <clean_rgdir> is a directory name under the dataset root (default
paths.dataset("f360seg"), override with PC_ROOT). Learning environment
(torch/dgl).

Usage: python f360_paired_consistency.py <ckpt> <clean_rgdir> <repart_pkl_dir>
"""
import os, sys, glob, pickle, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths
import numpy as np, torch, dgl
from region_train import RegionNet


DEV = torch.device("cuda")
D = os.environ.get("PC_ROOT", paths.dataset("f360seg"))
ck = torch.load(sys.argv[1], map_location=DEV); cfg = ck.get("cfg", {})
edim = ck["model"]["convs.0.fc_edge.weight"].shape[1]
m = RegionNet(ck["in_dim"], cfg.get("d", 384), cfg.get("layers", 6),
              cfg.get("heads", 6), edge_dim=edim).to(DEV)
m.load_state_dict(ck["model"]); m.eval()
MU, SD = ck["mu"].to(DEV), ck["sd"].to(DEV)

@torch.no_grad()
def pred(z):
    g = dgl.graph((z["src"], z["dst"]), num_nodes=z["feats"].shape[0]).to(DEV)
    x = (torch.from_numpy(np.asarray(z["feats"])).float().to(DEV) - MU) / SD
    p = m(g, x, torch.from_numpy(np.asarray(z["ef"])).float().to(DEV)).argmax(-1)
    return p[torch.from_numpy(np.asarray(z["f2r"])).to(DEV)].cpu().numpy()

agree = tot = stable = n = 0
for fp in sorted(glob.glob(os.path.join(sys.argv[3], "*.pkl"))):
    nm = os.path.basename(fp)[:-4]
    cz = f"{D}/{sys.argv[2]}/{nm}.npz"
    if not os.path.exists(cz):
        continue
    d = pickle.load(open(fp, "rb"))
    par = d.get("parent")
    if par is None:
        continue
    pc, pp = pred(np.load(cz)), pred(d)
    if par.max() >= len(pc):
        continue
    vp = par >= 0     # rejected transfers are -1; mask identically
    par = par[vp]; pp = pp[vp]
    same = pp == pc[par]
    agree += int(same.sum()); tot += len(same)
    stable += int(same.all()); n += 1
print(f"SUMMARY dir={sys.argv[3]} faces same-pred {agree}/{tot} = {agree/max(tot,1):.4f}"
      f"  solids 100% stable {stable}/{n} = {stable/max(n,1):.4f}", flush=True)
