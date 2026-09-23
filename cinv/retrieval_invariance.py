"""Shape RETRIEVAL under re-partitioning.

Tests whether the construction-dependence seen in per-face segmentation is
specific to dense per-face prediction. Retrieval is a different problem
setting and needs no labels at all:

  gallery = clean B-reps of N solids
  query   = the SAME N solids, re-partitioned (geometry identical to 1e-9)

A representation that describes the SOLID retrieves each query's own clean
version at rank 1.  A representation that describes the DRAWING does not.
Metrics: rank-1 accuracy, mean reciprocal rank, and the self-similarity margin.

Embeddings are pooled from each model's own node features -- no retrieval
training, so this measures the representation, not a learned metric. Inputs
are the pkl_clean and pkl_rpA / pkl_rpB directories of the Fusion 360 root.
Learning environment (torch/dgl), with the AAGNet checkout on the path.

Usage: python retrieval_invariance.py [--region <ckpt>] [--n 2000]
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


def aag_from_dict(d):
    g = dgl.graph(tuple(d["graph"]["edges"]), num_nodes=d["graph"]["num_nodes"])
    g.ndata["x"] = torch.tensor(np.array(d["graph_face_attr"]), dtype=torch.float32)
    g.ndata["grid"] = torch.tensor(np.array(d["graph_face_grid"]), dtype=torch.float32)
    g.edata["x"] = torch.tensor(np.array(d["graph_edge_attr"]), dtype=torch.float32)
    if g.edata["x"].size(0) == 0:
        return None
    g.ndata["x"] = ((g.ndata["x"] - STAT["mean_face_attr"]) / STAT["std_face_attr"]).float()
    g.edata["x"] = ((g.edata["x"] - STAT["mean_edge_attr"]) / STAT["std_edge_attr"]).float()
    return g


@torch.no_grad()
def aag_embed(am, g):
    """Mean+max pool of AAGNet's own per-face embeddings -> shape descriptor."""
    g = g.to(DEV)
    na = am.node_attr_encoder(g.ndata["x"])
    ng = am.node_grid_encoder(g.ndata["grid"])
    nf = torch.cat([na, ng], dim=1)
    ef = am.edge_attr_encoder(g.edata["x"])
    node_emb, _ = am.graph_encoder(g, nf, ef)
    z = torch.cat([node_emb.mean(0), node_emb.max(0).values])
    return torch.nn.functional.normalize(z, dim=0).cpu().numpy()


@torch.no_grad()
def region_embed(rm, MU, SD, feats, src, dst, ef):
    g = dgl.graph((src, dst), num_nodes=feats.shape[0]).to(DEV)
    x = (torch.from_numpy(feats).float().to(DEV) - MU) / SD
    h = rm.trunk(g, x, torch.from_numpy(ef).float().to(DEV))
    z = torch.cat([h.mean(0), h.max(0).values])
    return torch.nn.functional.normalize(z, dim=0).cpu().numpy()


def metrics(Q, G):
    """Q[i] should retrieve G[i]. -> rank1, MRR, mean margin."""
    S = Q @ G.T
    n = len(Q)
    order = np.argsort(-S, axis=1)
    ranks = np.array([int(np.where(order[i] == i)[0][0]) + 1 for i in range(n)])
    self_sim = S[np.arange(n), np.arange(n)]
    S2 = S.copy(); S2[np.arange(n), np.arange(n)] = -1e9
    best_other = S2.max(1)
    return (float((ranks == 1).mean()), float((1.0 / ranks).mean()),
            float((self_sim - best_other).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default=paths.ckpt("region_big"))
    ap.add_argument("--n", type=int, default=2000)
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
    am.load_state_dict(sd.get("model", sd.get("state_dict", sd)), strict=False); am.eval()

    ck = torch.load(args.region, map_location=DEV)
    cfg = ck.get("cfg", {})
    rm = RegionNet(ck["in_dim"], cfg.get("d", 384), cfg.get("layers", 6),
                   cfg.get("heads", 6)).to(DEV)
    rm.load_state_dict(ck["model"]); rm.eval()
    MU, SD = ck["mu"].to(DEV), ck["sd"].to(DEV)

    F = paths.dataset("f360")
    print(f"\n{'family':10s} {'model':10s} {'rank-1':>9s} {'MRR':>8s} {'margin':>9s} {'N':>7s}")
    print("-" * 60)
    print("gallery = clean F360 B-reps;  query = SAME solids re-partitioned")
    for fam in ("rpA", "rpB"):
        dd = os.path.join(F, f"pkl_{fam}")
        files = sorted(glob.glob(os.path.join(dd, "*.pkl")))[:args.n]
        qa, ga_, qr, gr = [], [], [], []
        for fp in files:
            nm = os.path.basename(fp)[:-4]
            cz = os.path.join(F, "pkl_clean", nm + ".pkl")
            if not os.path.exists(cz):
                continue
            d = pickle.load(open(fp, "rb")); c = pickle.load(open(cz, "rb"))
            gr.append(region_embed(rm, MU, SD, c["feats"], c["src"], c["dst"], c["ef"]))
            qr.append(region_embed(rm, MU, SD, d["feats"], d["src"], d["dst"], d["ef"]))
            if d["aag"] is not None and c["aag"] is not None:
                gc = aag_from_dict(c["aag"]); gp = aag_from_dict(d["aag"])
                if gc is not None and gp is not None:
                    ga_.append(aag_embed(am, gc)); qa.append(aag_embed(am, gp))
        if qa:
            r1, mrr, mg = metrics(np.stack(qa), np.stack(ga_))
            print(f"{fam:10s} {'AAGNet':10s} {100*r1:8.2f}% {mrr:8.4f} {mg:9.4f} {len(qa):7d}")
        if qr:
            r1, mrr, mg = metrics(np.stack(qr), np.stack(gr))
            print(f"{fam:10s} {'Region':10s} {100*r1:8.2f}% {mrr:8.4f} {mg:9.4f} {len(qr):7d}")


if __name__ == "__main__":
    main()
