"""PAIRED consistency: same solid, two B-reps -- does the model say the same thing?

Accuracy on a re-partitioned test set is an AGGREGATE comparison against ground
truth.  A model could hold its aggregate score while flipping individual
predictions, so aggregate accuracy is not by itself a measure of construction
invariance.

This is the task-level analogue of the embedding-distance experiment: for each
solid, predict on the clean B-rep and on the re-partitioned one, map every
child face back to its parent face, and measure the fraction of faces whose
predicted label is UNCHANGED.  100% = the model's output is a function of the
solid; lower = the output depends on how the solid was drawn.

Inputs: the MFInstSeg root (clean AAGNet graphs, region_graphs/*.npz) and the
paired_rpA / paired_rpB pkl directories (re-partitioned graphs with the
child->parent map). Prints, per family and model, the face-level agreement and
the fraction of solids that are 100% stable. Learning environment (torch/dgl),
with the AAGNet checkout on the path.

Usage: python paired_consistency.py [--region <ckpt>] [--limit 3000]
"""
from __future__ import annotations
import os, sys, glob, pickle, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch, dgl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths
AAG = paths.AAGNET
sys.path.insert(0, AAG)
from models.inst_segmentors import AAGNetSegmentor
from utils.data_utils import load_statistics
from dataloader.mfinstseg import MFInstSegDataset
from region_train import RegionNet, N_CLASSES


ROOT = os.environ.get("DS_ROOT", paths.dataset("mfinstseg"))
DEV = torch.device("cuda")
STAT = load_statistics(os.path.join(AAG, "weights", "attr_stat.json"))


def norm_aag(g):
    g.ndata["x"] = ((g.ndata["x"] - STAT["mean_face_attr"]) / STAT["std_face_attr"]).float()
    g.edata["x"] = ((g.edata["x"] - STAT["mean_edge_attr"]) / STAT["std_edge_attr"]).float()
    return g


def aag_from_dict(d):
    g = dgl.graph(tuple(d["graph"]["edges"]), num_nodes=d["graph"]["num_nodes"])
    g.ndata["x"] = torch.tensor(np.array(d["graph_face_attr"]), dtype=torch.float32)
    g.ndata["grid"] = torch.tensor(np.array(d["graph_face_grid"]), dtype=torch.float32)
    g.edata["x"] = torch.tensor(np.array(d["graph_edge_attr"]), dtype=torch.float32)
    if g.edata["x"].size(0) == 0:
        return None
    return norm_aag(g)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default=paths.ckpt("region_big"))
    ap.add_argument("--limit", type=int, default=3000)
    args = ap.parse_args()

    am = AAGNetSegmentor(
        num_classes=N_CLASSES, arch="AAGNetGraphEncoder",
        edge_attr_dim=12, node_attr_dim=10, edge_attr_emb=64, node_attr_emb=64,
        edge_grid_dim=0, node_grid_dim=7, edge_grid_emb=0, node_grid_emb=64,
        num_layers=3, delta=2, mlp_ratio=2, drop=0.25, drop_path=0.25,
        head_hidden_dim=64, conv_on_edge=False, use_uv_gird=True,
        use_edge_attr=True, use_face_attr=True).to(DEV)
    sd = torch.load(os.path.join(paths.AAGNET, "weights", "weight_on_MFInstseg.pth"),
                    map_location=DEV)
    am.load_state_dict(sd.get("model", sd.get("state_dict", sd)), strict=False)
    am.eval()

    ck = torch.load(args.region, map_location=DEV)
    cfg = ck.get("cfg", {})
    rm = RegionNet(ck["in_dim"], cfg.get("d", 384), cfg.get("layers", 6),
                   cfg.get("heads", 6)).to(DEV)
    rm.load_state_dict(ck["model"]); rm.eval()
    MU, SD = ck["mu"].to(DEV), ck["sd"].to(DEV)

    # clean AAGNet graphs, indexed by name
    te = MFInstSegDataset(root_dir=ROOT, split="test", center_and_scale=False,
                          normalize=True, random_rotate=False, dataset_type="full",
                          num_threads=8)
    clean_aag = {}
    for i in range(len(te)):
        s = te[i]
        clean_aag[s["filename"]] = s["graph"]
    print(f"[data] clean AAGNet graphs: {len(clean_aag)}", flush=True)

    print(f"\n{'family':12s} {'model':12s} {'faces same-pred':>16s} {'solids 100% stable':>20s}")
    print("-" * 64)
    for fam, dd in [("famA", os.path.join(ROOT, "paired_rpA")),
                    ("famB", os.path.join(ROOT, "paired_rpB"))]:
        files = sorted(glob.glob(os.path.join(dd, "*.pkl")))[:args.limit]
        agree_a = tot_a = 0; solid_a = n_a = 0
        agree_r = tot_r = 0; solid_r = n_r = 0
        for fp in files:
            nm = os.path.basename(fp)[:-4]
            d = pickle.load(open(fp, "rb"))
            par = d.get("parent")
            if par is None:
                continue
            # ---- ours: clean region graph vs re-partitioned region graph ----
            cz = os.path.join(ROOT, "region_graphs", nm + ".npz")
            if os.path.exists(cz):
                z = np.load(cz)
                gc = dgl.graph((z["src"], z["dst"]), num_nodes=z["feats"].shape[0]).to(DEV)
                xc = (torch.from_numpy(z["feats"]).float().to(DEV) - MU) / SD
                pc = rm(gc, xc, torch.from_numpy(z["ef"]).float().to(DEV)).argmax(-1)
                pcf = pc[torch.from_numpy(z["f2r"]).to(DEV)].cpu().numpy()
                gp = dgl.graph((d["src"], d["dst"]), num_nodes=d["feats"].shape[0]).to(DEV)
                xp = (torch.from_numpy(d["feats"]).float().to(DEV) - MU) / SD
                pp = rm(gp, xp, torch.from_numpy(d["ef"]).float().to(DEV)).argmax(-1)
                ppf = pp[torch.from_numpy(d["f2r"]).to(DEV)].cpu().numpy()
                if par.max() < len(pcf):
                    vp = par >= 0   # rejected transfers are -1
                    same = (ppf[vp] == pcf[par[vp]])
                    agree_r += int(same.sum()); tot_r += len(same)
                    solid_r += int(same.all()); n_r += 1
            # ---- AAGNet: clean AAG vs re-partitioned AAG ----
            if nm in clean_aag and d["aag"] is not None:
                gc = clean_aag[nm].to(DEV)
                pcf = am(gc)[0].argmax(-1).cpu().numpy()
                gp = aag_from_dict(d["aag"])
                if gp is not None and gp.num_nodes() == len(par) and par.max() < len(pcf):
                    ppf = am(gp.to(DEV))[0].argmax(-1).cpu().numpy()
                    vp = par >= 0   # rejected transfers are -1
                    same = (ppf[vp] == pcf[par[vp]])
                    agree_a += int(same.sum()); tot_a += len(same)
                    solid_a += int(same.all()); n_a += 1
        print(f"{fam:12s} {'AAGNet':12s} {100*agree_a/max(tot_a,1):15.2f}% "
              f"{100*solid_a/max(n_a,1):19.2f}%")
        print(f"{fam:12s} {'Region':12s} {100*agree_r/max(tot_r,1):15.2f}% "
              f"{100*solid_r/max(n_r,1):19.2f}%")


if __name__ == "__main__":
    main()
