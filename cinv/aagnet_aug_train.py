"""Train AAGNet WITH re-partition augmentation (family A), then test on family B.

This is the ablation that answers "why not just augment?".  Grid
re-parameterisations form a finite group, so augmenting over them works (UV-Net
2021).  Face partitions do not: there is no finite set to sample.  The
prediction is that famA augmentation largely recovers famA and transfers only
partially to famB, while the region graph needs no augmentation at all and is
unaffected by either.

Fine-tunes the published AAGNet weights on the clean MFInstSeg training set
mixed with the family-A pkls in <root>/<AUG_PKL_DIR> (default eval_rpA_train),
saves <CKPT>/aagnet_<tag>.pth and scores it on the family-A and family-B
evaluation cells. Learning environment (torch/dgl), with the AAGNet checkout
on the path.

Usage: python aagnet_aug_train.py [--epochs 30] [--bs 64] [--lr 1e-3] [--tag aug]
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
sys.path.insert(0, AAG)
from models.inst_segmentors import AAGNetSegmentor
from utils.data_utils import load_statistics
from dataloader.mfinstseg import MFInstSegDataset

ROOT = os.environ.get("DS_ROOT", paths.dataset("mfinstseg"))
DEV = torch.device("cuda")
N_CLASSES = 25
STAT = load_statistics(os.path.join(AAG, "weights", "attr_stat.json"))


def macro_miou(y, p, n=N_CLASSES):
    ious = []
    for c in range(n):
        t, q = (y == c), (p == c)
        u = (t | q).sum()
        if u:
            ious.append((t & q).sum() / u)
    return float(np.mean(ious)) if ious else 0.0


def aag_graph(d, y=None):
    g = dgl.graph(tuple(d["graph"]["edges"]), num_nodes=d["graph"]["num_nodes"])
    g.ndata["x"] = torch.tensor(np.array(d["graph_face_attr"]), dtype=torch.float32)
    g.ndata["grid"] = torch.tensor(np.array(d["graph_face_grid"]), dtype=torch.float32)
    g.edata["x"] = torch.tensor(np.array(d["graph_edge_attr"]), dtype=torch.float32)
    if g.edata["x"].size(0) == 0 or (y is not None and g.num_nodes() != len(y)):
        return None
    g.ndata["x"] = ((g.ndata["x"] - STAT["mean_face_attr"]) / STAT["std_face_attr"]).float()
    g.edata["x"] = ((g.edata["x"] - STAT["mean_edge_attr"]) / STAT["std_edge_attr"]).float()
    if y is not None:
        g.ndata["seg_y"] = torch.from_numpy(np.asarray(y)).long()
    return g


class PklGraphs(torch.utils.data.Dataset):
    """Re-partitioned graphs saved by repart_extract.py."""
    def __init__(self, d):
        self.files = sorted(glob.glob(os.path.join(d, "*.pkl")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = pickle.load(open(self.files[i], "rb"))
        if d["aag"] is None:
            return None
        return aag_graph(d["aag"], d["y"])


class Mixed(torch.utils.data.Dataset):
    def __init__(self, clean, aug, aug_frac=1.0):
        self.clean, self.aug = clean, aug
        self.n_aug = int(len(aug) * aug_frac)

    def __len__(self):
        return len(self.clean) + self.n_aug

    def __getitem__(self, i):
        g = (self.clean[i]["graph"] if i < len(self.clean)
             else self.aug[i - len(self.clean)])
        return _strip(g)


KEEP_N = ("x", "grid", "seg_y")


def _strip(g):
    """dgl.batch requires identical schemas; the clean loader adds bottom_y and
    the re-partitioned graphs have no instance labels, so keep only the fields
    the segmentation head actually consumes."""
    if g is None:
        return None
    for k in list(g.ndata.keys()):
        if k not in KEEP_N:
            del g.ndata[k]
    for k in list(g.edata.keys()):
        if k != "x":
            del g.edata[k]
    return g


def collate(b):
    b = [g for g in b if g is not None and g.num_nodes() > 0]
    return dgl.batch(b) if b else None


@torch.no_grad()
def evaluate(model, ds):
    model.eval(); ys, ps = [], []
    dl = torch.utils.data.DataLoader(ds, 64, num_workers=4,
                                     collate_fn=lambda b: collate([_strip(g) for g in b]))
    for g in dl:
        if g is None:
            continue
        g = g.to(DEV)
        seg, _, _ = model(g)
        ys.append(g.ndata["seg_y"].cpu().numpy()); ps.append(seg.argmax(-1).cpu().numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    return float((y == p).mean()), macro_miou(y, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tag", default="aug")
    args = ap.parse_args()

    model = AAGNetSegmentor(
        num_classes=N_CLASSES, arch="AAGNetGraphEncoder",
        edge_attr_dim=12, node_attr_dim=10, edge_attr_emb=64, node_attr_emb=64,
        edge_grid_dim=0, node_grid_dim=7, edge_grid_emb=0, node_grid_emb=64,
        num_layers=3, delta=2, mlp_ratio=2, drop=0.25, drop_path=0.25,
        head_hidden_dim=64, conv_on_edge=False, use_uv_gird=True,
        use_edge_attr=True, use_face_attr=True).to(DEV)
    sd = torch.load(os.path.join(paths.AAGNET, "weights", "weight_on_MFInstseg.pth"),
                    map_location=DEV)
    model.load_state_dict(sd.get("model", sd.get("state_dict", sd)), strict=False)
    print("[init] from published weights", flush=True)

    t0 = time.time()
    clean = MFInstSegDataset(root_dir=ROOT, split="train", center_and_scale=False,
                             normalize=True, random_rotate=False, dataset_type="full",
                             num_threads=8)
    print(f"[data] clean train={len(clean)} ({time.time()-t0:.0f}s)", flush=True)
    aug = PklGraphs(os.path.join(ROOT, os.environ.get("AUG_PKL_DIR", "eval_rpA_train")))
    print(f"[data] famA augmented train={len(aug)}", flush=True)
    tr = Mixed(clean, aug)
    dl = torch.utils.data.DataLoader(tr, args.bs, shuffle=True, num_workers=8,
                                     collate_fn=collate, drop_last=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.CrossEntropyLoss()
    for ep in range(args.epochs):
        model.train(); ls = []
        for g in dl:
            if g is None:
                continue
            g = g.to(DEV)
            seg, _, _ = model(g)
            loss = lossf(seg, g.ndata["seg_y"])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            ls.append(loss.item())
        sch.step()
        if (ep + 1) % 5 == 0 or ep == args.epochs - 1:
            print(f"[ep {ep+1}] loss={np.mean(ls):.4f} ({(time.time()-t0)/60:.1f}m)", flush=True)
    dst = os.path.join(paths.CKPT, f"aagnet_{args.tag}.pth")
    torch.save({"model": model.state_dict()}, dst)
    print(f"[saved] {dst}", flush=True)

    for tag, d in [("famA (trained on)", os.path.join(ROOT, "repart_eval")),
                   ("famB (held out)", os.path.join(ROOT, "eval_rpB"))]:
        acc, miou = evaluate(model, PklGraphs(d))
        print(f"[AAGNet+aug] {tag:22s} acc={acc:.4f} mIoU={miou:.4f}", flush=True)


if __name__ == "__main__":
    main()
