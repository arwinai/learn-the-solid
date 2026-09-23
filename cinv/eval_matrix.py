"""The full matrix: every dataset x every model x {clean, famA, famB}.

Prints one row per (dataset, model) with clean / famA / famB mIoU, all evaluated
on the pkl graphs so all models see identical solids and identical labels.
Missing cells are printed as '--' rather than silently omitted. Checkpoints are
looked up under $CINV_CKPT (region_<dataset>.pt, aagnet_<dataset>.pth,
uvnet_<dataset>.pth); AAGNet on MFInstSeg uses the published weights.

Learning environment (torch/dgl). Usage: python eval_matrix.py
"""
from __future__ import annotations
import os, sys, glob, pickle, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch, dgl

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import paths

AAG = paths.AAGNET
UVN = paths.UVNET
sys.path.insert(0, AAG); sys.path.insert(0, UVN)
from region_train import RegionNet, macro_miou
from train_baseline import make, fwd, aag_graph, Pkl, coll

DATA = paths.DATA
DEV = torch.device("cuda")

DATASETS = [("MFInstSeg", "mfinstseg"), ("MFCAD++", "mfcadpp"),
            ("F360seg", "f360seg"), ("CADSynth", "cadsynth")]


def region_ckpt(slug):
    cands = [f"region_{slug}.pt"]
    if slug == "mfinstseg":
        cands.append("region_big.pt")
    if slug == "cadsynth":
        cands.append("region_cadsynth_pre.pt")
    for c in cands:
        if c and os.path.exists(os.path.join(paths.CKPT, c)):
            return os.path.join(paths.CKPT, c)
    return None


@torch.no_grad()
def eval_region(ck, dirs, test_names=None):
    c = torch.load(ck, map_location=DEV); cfg = c.get("cfg", {})
    # edge-feature width depends on the feature flags used at extraction
    # (3 without EHIST, 10 with); the checkpoint is the authority
    edim = c["model"]["convs.0.fc_edge.weight"].shape[1]
    m = RegionNet(c["in_dim"], cfg.get("d", 384), cfg.get("layers", 6),
                  cfg.get("heads", 6), edge_dim=edim).to(DEV)
    m.load_state_dict(c["model"]); m.eval()
    MU, SD = c["mu"].to(DEV), c["sd"].to(DEV)
    out = []
    for i, d in enumerate(dirs):
        if d is None or not os.path.isdir(d):
            out.append(None); continue
        fps = sorted(glob.glob(os.path.join(d, "*.pkl")))
        if i == 0 and test_names is not None:
            # clean dir contains ALL splits; restrict to test or the cell is
            # train-contaminated
            tn = set(test_names)
            fps = [f for f in fps if os.path.basename(f)[:-4] in tn]
        ys, ps = [], []
        for fp in fps:
            p = pickle.load(open(fp, "rb"))
            g = dgl.graph((p["src"], p["dst"]), num_nodes=p["feats"].shape[0]).to(DEV)
            x = (torch.from_numpy(p["feats"]).float().to(DEV) - MU) / SD
            e = torch.from_numpy(p["ef"]).float().to(DEV)
            pf = m(g, x, e).argmax(-1)[torch.from_numpy(p["f2r"]).to(DEV)]
            ys.append(p["y"]); ps.append(pf.cpu().numpy())
        out.append(macro_miou(np.concatenate(ys), np.concatenate(ps)) if ys else None)
    return out


@torch.no_grad()
def eval_base(arch, ckpt, dirs, names=None):
    m = make(arch)
    sd = torch.load(ckpt, map_location=DEV)
    m.load_state_dict(sd.get("model", sd), strict=False); m.eval()
    out = []
    for i, d in enumerate(dirs):
        if d is None or not os.path.isdir(d):
            out.append(None); continue
        ds = Pkl(d, names if i == 0 else None)
        ys, ps = [], []
        dl = torch.utils.data.DataLoader(ds, 64, num_workers=6, collate_fn=coll)
        for g in dl:
            if g is None:
                continue
            g = g.to(DEV)
            o = fwd(m, arch, g)
            ys.append(g.ndata["seg_y"].cpu().numpy()); ps.append(o.argmax(-1).cpu().numpy())
        out.append(macro_miou(np.concatenate(ys), np.concatenate(ps)) if ys else None)
    return out


def fmt(v):
    return "  --  " if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"


def main():
    print(f"\n{'dataset':11s} {'model':10s} {'clean':>7s} {'famA':>7s} {'famB':>7s}")
    print("-" * 48)
    for label, slug in DATASETS:
        root = os.path.join(DATA, slug)
        dirs = [os.path.join(root, "eval_clean"),
                os.path.join(root, "eval_rpA") if os.path.isdir(os.path.join(root, "eval_rpA"))
                else os.path.join(root, "repart_eval_v2"),
                os.path.join(root, "eval_rpB")]
        te = [l.strip() for l in open(os.path.join(root, "test.txt"))] \
            if os.path.exists(os.path.join(root, "test.txt")) else None
        for arch, tags in (("aagnet", [f"aagnet_{slug}", "aagnet_mfcadpp"]),
                           ("uvnet", [f"uvnet_{slug}"])):
            ck = next((os.path.join(paths.CKPT, f"{t}.pth") for t in tags
                       if os.path.exists(os.path.join(paths.CKPT, f"{t}.pth"))), None)
            if slug == "mfinstseg" and arch == "aagnet":
                ck = os.path.join(AAG, "weights/weight_on_MFInstseg.pth")
            if slug == "mfinstseg" and arch == "uvnet":
                ck = os.path.join(paths.CKPT, "uvnet_mfi.pth")
            if ck and os.path.exists(ck):
                r = eval_base(arch, ck, dirs, te)
            else:
                r = [None, None, None]
            print(f"{label:11s} {arch:10s} {fmt(r[0]):>7s} {fmt(r[1]):>7s} {fmt(r[2]):>7s}",
                  flush=True)
        rck = region_ckpt(slug)
        r = eval_region(rck, dirs, te) if rck else [None, None, None]
        print(f"{label:11s} {'ours':10s} {fmt(r[0]):>7s} {fmt(r[1]):>7s} {fmt(r[2]):>7s}",
              flush=True)
        print()


if __name__ == "__main__":
    main()
