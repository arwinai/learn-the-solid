"""Train AAGNet or UV-Net on any dataset, from the pkl graphs.

One entry point so the baselines get identical treatment on every dataset --
same optimiser, same schedule, same epoch budget, same data pipeline; baselines
trained under different setups per dataset are not comparable. Reads the
AAGNet-format graphs in <root>/eval_clean/*.pkl for the names in <root>/train.txt
and <root>/test.txt, saves the weights to <CKPT>/<tag>.pth and prints test
accuracy and macro mIoU. Also exports make / fwd / aag_graph / coll for the
evaluation scripts. Learning environment (torch/dgl), with the AAGNet and
UV-Net checkouts on the path.

Usage: python train_baseline.py --root <dataset_root> --arch {aagnet,uvnet}
       --tag <name> [--epochs 60] [--bs 64] [--lr 1e-3]
"""
from __future__ import annotations
import os, sys, glob, time, pickle, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch, torch.nn as nn
import dgl


HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import paths
AAG = paths.AAGNET
UVN = paths.UVNET
sys.path.insert(0, AAG); sys.path.insert(0, UVN)
from models.inst_segmentors import AAGNetSegmentor
from utils.data_utils import load_statistics
from region_train import macro_miou


DEV = torch.device("cuda")
N_CLASSES = 25
STAT = load_statistics(os.path.join(AAG, "weights", "attr_stat.json"))
KEEP = ("x", "grid", "seg_y")


def strip(g):
    if g is None:
        return None
    for k in list(g.ndata.keys()):
        if k not in KEEP:
            del g.ndata[k]
    for k in list(g.edata.keys()):
        if k != "x":
            del g.edata[k]
    return g


def aag_graph(d, y):
    g = dgl.graph(tuple(d["graph"]["edges"]), num_nodes=d["graph"]["num_nodes"])
    g.ndata["x"] = torch.tensor(np.array(d["graph_face_attr"]), dtype=torch.float32)
    g.ndata["grid"] = torch.tensor(np.array(d["graph_face_grid"]), dtype=torch.float32)
    g.edata["x"] = torch.tensor(np.array(d["graph_edge_attr"]), dtype=torch.float32)
    if g.edata["x"].size(0) == 0 or g.num_nodes() != len(y):
        return None
    g.ndata["x"] = ((g.ndata["x"] - STAT["mean_face_attr"]) / STAT["std_face_attr"]).float()
    g.edata["x"] = ((g.edata["x"] - STAT["mean_edge_attr"]) / STAT["std_edge_attr"]).float()
    g.ndata["seg_y"] = torch.from_numpy(np.asarray(y)).long()
    return strip(g)


class Pkl(torch.utils.data.Dataset):
    def __init__(self, d, names=None):
        fs = sorted(glob.glob(os.path.join(d, "*.pkl")))
        if names is not None:
            s = set(names)
            fs = [f for f in fs if os.path.basename(f)[:-4] in s]
        self.files = fs

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        p = pickle.load(open(self.files[i], "rb"))
        return aag_graph(p["aag"], p["y"]) if p.get("aag") is not None else None


def coll(b):
    b = [g for g in b if g is not None and g.num_nodes() > 0]
    return dgl.batch(b) if b else None


def make(arch):
    if arch == "aagnet":
        return AAGNetSegmentor(
            num_classes=N_CLASSES, arch="AAGNetGraphEncoder",
            edge_attr_dim=12, node_attr_dim=10, edge_attr_emb=64, node_attr_emb=64,
            edge_grid_dim=0, node_grid_dim=7, edge_grid_emb=0, node_grid_emb=64,
            num_layers=3, delta=2, mlp_ratio=2, drop=0.25, drop_path=0.25,
            head_hidden_dim=64, conv_on_edge=False, use_uv_gird=True,
            use_edge_attr=True, use_face_attr=True).to(DEV)
    from uvnet_families import UVNetSeg
    return UVNetSeg().to(DEV)


def fwd(m, arch, g):
    return m(g)[0] if arch == "aagnet" else m(g)


@torch.no_grad()
def ev(m, arch, ds):
    m.eval(); ys, ps = [], []
    dl = torch.utils.data.DataLoader(ds, 64, num_workers=6, collate_fn=coll)
    for g in dl:
        if g is None:
            continue
        g = g.to(DEV)
        o = fwd(m, arch, g)
        ys.append(g.ndata["seg_y"].cpu().numpy()); ps.append(o.argmax(-1).cpu().numpy())
    if not ys:
        return float("nan"), float("nan")
    y, p = np.concatenate(ys), np.concatenate(ps)
    return float((y == p).mean()), macro_miou(y, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--arch", choices=["aagnet", "uvnet"], required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    dst = os.path.join(paths.CKPT, f"{a.tag}.pth")
    if os.path.exists(dst):
        print(f"[skip] {dst} exists", flush=True); return
    names = lambda s: [l.strip() for l in open(os.path.join(a.root, f"{s}.txt")) if l.strip()]
    clean = os.path.join(a.root, "eval_clean")
    tr = Pkl(clean, names("train")); te = Pkl(clean, names("test"))
    print(f"[data] {a.tag} train={len(tr)} test={len(te)}", flush=True)
    if len(tr) == 0:
        print(f"[abort] no training graphs in {clean}", flush=True); return

    m = make(a.arch)
    dl = torch.utils.data.DataLoader(tr, a.bs, shuffle=True, num_workers=8,
                                     collate_fn=coll, drop_last=True)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    lf = nn.CrossEntropyLoss()
    t0 = time.time()
    for ep in range(a.epochs):
        m.train(); ls = []
        for g in dl:
            if g is None:
                continue
            g = g.to(DEV)
            loss = lf(fwd(m, a.arch, g), g.ndata["seg_y"])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            ls.append(loss.item())
        sch.step()
        if (ep + 1) % 20 == 0:
            print(f"[ep {ep+1}] loss={np.mean(ls):.4f} ({(time.time()-t0)/60:.1f}m)", flush=True)
    torch.save({"model": m.state_dict()}, dst)
    acc, miou = ev(m, a.arch, te)
    print(f"SUMMARY tag={a.tag}  test acc={acc:.4f}  mIoU={miou:.4f}", flush=True)


if __name__ == "__main__":
    main()
