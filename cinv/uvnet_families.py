"""UV-Net baseline: training on clean graphs and evaluation under re-partitioning.

UV-Net is the architecture that originated the UV-grid face-graph paradigm, so
it is the second baseline alongside AAGNet. `UVNetSeg` is taken verbatim from
the AAGNet repository's own run_uvnet_mfi.py so both baselines share one data
pipeline; the UV-Net encoders come from the UV-Net checkout (CINV_UVNET).

Writes $CINV_CKPT/uvnet_mfi.pth. Learning environment (torch/dgl).
Usage: python uvnet_families.py [--epochs 40 --bs 64 --lr 2e-3]
"""
from __future__ import annotations
import os, sys, glob, time, pickle, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch, torch.nn as nn
import dgl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths

AAG = paths.AAGNET
UVN = paths.UVNET
sys.path.insert(0, AAG); sys.path.insert(0, UVN)
from uvnet.encoders import UVNetSurfaceEncoder, UVNetGraphEncoder
from dataloader.mfinstseg import MFInstSegDataset
from utils.data_utils import load_statistics
from region_train import macro_miou

ROOT = os.environ.get("DS_ROOT", paths.dataset("mfinstseg"))
DEV = torch.device("cuda")
N_CLASSES = 25
STAT = load_statistics(os.path.join(AAG, "weights/attr_stat.json"))


class UVNetSeg(nn.Module):
    """Verbatim from AAGNet's run_uvnet_mfi.py."""
    def __init__(self, srf_emb=64, crv_emb=64, graph_emb=128, edge_in=12, drop=0.3):
        super().__init__()
        self.surf_encoder = UVNetSurfaceEncoder(in_channels=7, output_dims=srf_emb)
        self.edge_enc = nn.Sequential(
            nn.Linear(edge_in, 64), nn.BatchNorm1d(64), nn.LeakyReLU(),
            nn.Linear(64, crv_emb))
        self.graph_encoder = UVNetGraphEncoder(srf_emb, crv_emb, graph_emb)
        self.seg = nn.Sequential(
            nn.Linear(graph_emb + srf_emb, 512, bias=False), nn.BatchNorm1d(512),
            nn.ReLU(), nn.Dropout(drop),
            nn.Linear(512, 256, bias=False), nn.BatchNorm1d(256),
            nn.ReLU(), nn.Dropout(drop),
            nn.Linear(256, N_CLASSES))

    def forward(self, g):
        srf = self.surf_encoder(g.ndata["grid"])
        crv = self.edge_enc(g.edata["x"])
        node_emb, graph_emb = self.graph_encoder(g, srf, crv)
        n = g.batch_num_nodes().to(graph_emb.device)
        graph_emb = graph_emb.repeat_interleave(n, dim=0)
        return self.seg(torch.cat([node_emb, graph_emb], dim=1))


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


def pkl_graph(d, y):
    g = dgl.graph(tuple(d["graph"]["edges"]), num_nodes=d["graph"]["num_nodes"])
    g.ndata["x"] = torch.tensor(np.array(d["graph_face_attr"]), dtype=torch.float32)
    g.ndata["grid"] = torch.tensor(np.array(d["graph_face_grid"]), dtype=torch.float32)
    g.edata["x"] = torch.tensor(np.array(d["graph_edge_attr"]), dtype=torch.float32)
    if g.edata["x"].size(0) == 0 or g.num_nodes() != len(y):
        return None
    g.ndata["x"] = ((g.ndata["x"] - STAT["mean_face_attr"]) / STAT["std_face_attr"]).float()
    g.edata["x"] = ((g.edata["x"] - STAT["mean_edge_attr"]) / STAT["std_edge_attr"]).float()
    g.ndata["seg_y"] = torch.from_numpy(np.asarray(y)).long()
    return g


class PklDS(torch.utils.data.Dataset):
    def __init__(self, d):
        self.files = sorted(glob.glob(os.path.join(d, "*.pkl")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        p = pickle.load(open(self.files[i], "rb"))
        return strip(pkl_graph(p["aag"], p["y"])) if p["aag"] is not None else None


class CleanDS(torch.utils.data.Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return strip(self.ds[i]["graph"])


def coll(b):
    b = [g for g in b if g is not None and g.num_nodes() > 0]
    return dgl.batch(b) if b else None


@torch.no_grad()
def evaluate(m, ds):
    m.eval(); ys, ps = [], []
    dl = torch.utils.data.DataLoader(ds, 64, num_workers=6, collate_fn=coll)
    for g in dl:
        if g is None:
            continue
        g = g.to(DEV)
        out = m(g)
        ys.append(g.ndata["seg_y"].cpu().numpy()); ps.append(out.argmax(-1).cpu().numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    return float((y == p).mean()), macro_miou(y, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()

    m = UVNetSeg().to(DEV)
    print(f"[model] UV-Net params={sum(p.numel() for p in m.parameters())/1e6:.2f}M", flush=True)
    t0 = time.time()
    tr_raw = MFInstSegDataset(root_dir=ROOT, split="train", center_and_scale=False,
                              normalize=True, random_rotate=False, dataset_type="full",
                              num_threads=8)
    te_raw = MFInstSegDataset(root_dir=ROOT, graphs=tr_raw.graphs(), split="test",
                              center_and_scale=False, normalize=True,
                              dataset_type="full", num_threads=8)
    tr, te = CleanDS(tr_raw), CleanDS(te_raw)
    print(f"[data] train={len(tr)} test={len(te)} ({time.time()-t0:.0f}s)", flush=True)
    dl = torch.utils.data.DataLoader(tr, args.bs, shuffle=True, num_workers=8,
                                     collate_fn=coll, drop_last=True)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lf = nn.CrossEntropyLoss()
    for ep in range(args.epochs):
        m.train(); ls = []
        for g in dl:
            if g is None:
                continue
            g = g.to(DEV)
            loss = lf(m(g), g.ndata["seg_y"])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            ls.append(loss.item())
        sch.step()
        if (ep + 1) % 10 == 0 or ep == args.epochs - 1:
            print(f"[ep {ep+1}] loss={np.mean(ls):.4f} ({(time.time()-t0)/60:.1f}m)", flush=True)
    torch.save({"model": m.state_dict()}, os.path.join(paths.CKPT, "uvnet_mfi.pth"))

    print(f"\n{'UV-Net on MFInstSeg':30s} {'acc':>9s} {'mIoU':>9s}")
    print("-" * 52)
    a, i = evaluate(m, te)
    print(f"{'clean test':30s} {a:9.4f} {i:9.4f}", flush=True)
    for tag, d in [("famA (axis planes)", os.path.join(ROOT, "repart_eval_v2")),
                   ("famB (diagonal, held out)", os.path.join(ROOT, "eval_rpB"))]:
        if os.path.isdir(d):
            a, i = evaluate(m, PklDS(d))
            print(f"{tag:30s} {a:9.4f} {i:9.4f}", flush=True)


if __name__ == "__main__":
    main()
