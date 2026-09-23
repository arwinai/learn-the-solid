"""Train UV-Net WITH family-A re-partition augmentation.

Same recipe as aagnet_aug_train.py (fine-tune from the clean-trained checkpoint
<CKPT>/uvnet_mfi.pth on a clean + augmented mix, 30 epochs, cosine schedule) so
the two augmentation rows are comparable. Clean training graphs are the pkls in
--clean (default <root>/pkl_train_clean); the augmented side is the family-A
pkl dir <root>/<AUG_PKL_DIR> (default eval_rpA_train), the same one AAGNet's
run uses. Saves <CKPT>/<tag>.pth and scores it on the clean, family-A and
family-B evaluation cells. Learning environment (torch/dgl), with the AAGNet
and UV-Net checkouts on the path.

Usage: python uvnet_aug_train.py [--epochs 30] [--bs 64] [--lr 1e-3]
       [--tag uvnet_aug] [--clean <pkl_dir>]
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths                                              # noqa: E402
from train_baseline import make, fwd, aag_graph, coll     # noqa: E402

ROOT = os.environ.get("DS_ROOT", paths.dataset("mfinstseg"))
DEV = torch.device("cuda")


class PklGraphs(torch.utils.data.Dataset):
    def __init__(self, d):
        self.files = sorted(glob.glob(os.path.join(d, "*.pkl")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = pickle.load(open(self.files[i], "rb"))
        if d.get("aag") is None:
            return None
        return aag_graph(d["aag"], d["y"])


class Mixed(torch.utils.data.Dataset):
    def __init__(self, a, b):
        self.a, self.b = a, b

    def __len__(self):
        return len(self.a) + len(self.b)

    def __getitem__(self, i):
        return self.a[i] if i < len(self.a) else self.b[i - len(self.a)]


@torch.no_grad()
def evaluate(m, ds):
    m.eval()
    ys, ps = [], []
    dl = torch.utils.data.DataLoader(ds, 64, num_workers=6, collate_fn=coll)
    for g in dl:
        if g is None:
            continue
        g = g.to(DEV)
        o = fwd(m, "uvnet", g)
        ys.append(g.ndata["seg_y"].cpu().numpy())
        ps.append(o.argmax(-1).cpu().numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    ious = []
    for c in range(25):
        t, q = (y == c), (p == c)
        u = (t | q).sum()
        if u:
            ious.append((t & q).sum() / u)
    return float((y == p).mean()), float(np.mean(ious))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tag", default="uvnet_aug" + os.environ.get("AUG_TAG", "") + "")
    ap.add_argument("--clean", default=os.path.join(ROOT, "pkl_train_clean"))
    a = ap.parse_args()

    m = make("uvnet")
    sd = torch.load(os.path.join(paths.CKPT, "uvnet_mfi.pth"), map_location=DEV)
    m.load_state_dict(sd.get("model", sd), strict=False)
    print("[init] from uvnet_mfi", flush=True)

    tr = Mixed(PklGraphs(a.clean),
               PklGraphs(os.path.join(ROOT, os.environ.get("AUG_PKL_DIR", "eval_rpA_train"))))
    print(f"[data] mixed train = {len(tr)}", flush=True)
    dl = torch.utils.data.DataLoader(tr, a.bs, shuffle=True, num_workers=8,
                                     collate_fn=coll, drop_last=True)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    lossf = nn.CrossEntropyLoss()
    t0 = time.time()
    for ep in range(a.epochs):
        m.train()
        ls = []
        for g in dl:
            if g is None:
                continue
            g = g.to(DEV)
            loss = lossf(fwd(m, "uvnet", g), g.ndata["seg_y"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ls.append(loss.item())
        sch.step()
        print(f"[ep {ep + 1}] loss={np.mean(ls):.4f} "
              f"({(time.time() - t0) / 60:.1f}m)", flush=True)
    dst = os.path.join(paths.CKPT, f"{a.tag}.pth")
    torch.save({"model": m.state_dict()}, dst)
    print(f"[saved] {dst}", flush=True)

    for tag, d in (("clean", os.path.join(ROOT, "eval_clean")),
                   ("famA (trained on)", os.path.join(ROOT, "repart_eval")),
                   ("famB (held out)", os.path.join(ROOT, "eval_rpB"))):
        acc, miou = evaluate(m, PklGraphs(d))
        print(f"SUMMARY uvnet+aug {tag}: acc={acc:.4f} mIoU={miou:.4f}",
              flush=True)


if __name__ == "__main__":
    main()
