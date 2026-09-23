"""Train the face-segmentation network on canonical region graphs.

Reads <root>/{train,val,test}.txt and the per-solid region_graphs/*.npz files
(feats/src/dst/ef/f2r/y), trains a RegionNet (EdgeGAT trunk with EMA weights,
label smoothing, optional class-balanced loss and optional initialisation from
another checkpoint), keeps the best-validation EMA weights, and reports test
accuracy and macro mIoU. The checkpoint is written to paths.ckpt("region_<tag>")
together with the feature normalisation statistics and the run configuration.
Also exports RegionNet / RegionDataset / collate / macro_miou for the
evaluation scripts. Learning environment (torch/dgl).

Usage: python region_train.py --root <dataset_root> --tag <name> [--epochs 150]
       [--d 384 --layers 6 --heads 6] [--cb 0.5] [--init <ckpt> --trunk_only]
       [--frac 0.1] [--drop_groups normal,curv] [--no_edge]
"""
from __future__ import annotations
import os, sys, time, argparse, copy, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import dgl
from dgl.nn import EdgeGATConv

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths

ROOT = os.environ.get("DS_ROOT", paths.dataset("mfinstseg"))
RG = os.path.join(ROOT, "region_graphs")
DEV = torch.device("cuda")
N_CLASSES = 25
EDGE_DIM = 3   # fallback edge-feature width; inferred from the data at train time


def macro_miou(y_true, y_pred, n_classes=N_CLASSES):
    ious = []
    for c in range(n_classes):
        t, p = (y_true == c), (y_pred == c)
        u = (t | p).sum()
        if u == 0:
            continue
        ious.append((t & p).sum() / u)
    return float(np.mean(ious)) if ious else 0.0


class RegionDataset(torch.utils.data.Dataset):
    def __init__(self, split, root=None):
        root = root or ROOT
        names = [l.strip() for l in open(os.path.join(root, f"{split}.txt")) if l.strip()]
        self.files = [os.path.join(root, "region_graphs", n + ".npz") for n in names]
        self.files = [f for f in self.files if os.path.exists(f)]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        z = np.load(self.files[i])
        f2r, feats, src, dst, ef, y = (z["f2r"], z["feats"], z["src"], z["dst"],
                                       z["ef"], z["y"])
        R = feats.shape[0]
        g = dgl.graph((src, dst), num_nodes=R)
        g.ndata["x"] = torch.from_numpy(feats).float()
        g.edata["e"] = torch.from_numpy(ef).float()
        yr = np.zeros(R, dtype=np.int64)
        for r in range(R):
            v = y[f2r == r]
            yr[r] = np.bincount(v).argmax() if len(v) else 0
        g.ndata["y"] = torch.from_numpy(yr)
        return g, torch.from_numpy(f2r), torch.from_numpy(y)


def collate(batch):
    gs, f2rs, ys = zip(*batch)
    off, out = 0, []
    for g, f2r in zip(gs, f2rs):
        out.append(f2r + off); off += g.num_nodes()
    return dgl.batch(gs), torch.cat(out), torch.cat(ys)


class RegionNet(nn.Module):
    def __init__(self, in_dim=38, d=384, layers=6, heads=6, drop=0.1, edge_dim=None):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(in_dim, d), nn.LayerNorm(d), nn.GELU(),
                                  nn.Linear(d, d), nn.LayerNorm(d), nn.GELU())
        ed = EDGE_DIM if edge_dim is None else edge_dim
        self.convs = nn.ModuleList([
            EdgeGATConv(d, ed, d // heads, heads, allow_zero_in_degree=True)
            for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.ffs = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 3 * d), nn.GELU(),
                          nn.Dropout(drop), nn.Linear(3 * d, d)) for _ in range(layers)])
        self.out_norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(drop),
                                  nn.Linear(d, N_CLASSES))
        self.d = d

    def trunk(self, g, x, e):
        h = self.stem(x)
        for conv, nrm, ff in zip(self.convs, self.norms, self.ffs):
            h = h + conv(g, nrm(h), e).flatten(1)
            h = h + ff(h)
        return self.out_norm(h)

    def forward(self, g, x, e):
        return self.head(self.trunk(g, x, e))


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, m in zip(self.shadow.state_dict().values(), model.state_dict().values()):
            if s.dtype.is_floating_point:
                s.mul_(self.decay).add_(m, alpha=1 - self.decay)
            else:
                s.copy_(m)


@torch.no_grad()
def run_eval(model, dl, MU, SD):
    model.eval(); ys, ps = [], []
    for g, f2r, y in dl:
        g = g.to(DEV)
        x = (g.ndata["x"] - MU) / SD
        pf = model(g, x, g.edata["e"]).argmax(-1)[f2r.to(DEV)]
        ys.append(y.numpy()); ps.append(pf.cpu().numpy())
    yy, pp = np.concatenate(ys), np.concatenate(ps)
    return float((yy == pp).mean()), macro_miou(yy, pp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--smooth", type=float, default=0.05)
    ap.add_argument("--cb", type=float, default=0.0, help="class-balance exponent")
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--init", default="", help="checkpoint to init from")
    ap.add_argument("--trunk_only", action="store_true",
                    help="load trunk but NOT the head -- required when transferring "
                         "across datasets whose label taxonomies differ (CADSynth's "
                         "dominant class is 0, MFInstSeg's is 24, so a shared head "
                         "would learn conflicting targets)")
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--root", default=ROOT, help="dataset root (mfinstseg / mfcadpp)")
    ap.add_argument("--frac", type=float, default=1.0, help="fraction of TRAIN labels used")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drop_groups", default="", help="comma list of feature groups to zero")
    ap.add_argument("--no_edge", action="store_true", help="zero the edge features")
    args = ap.parse_args()
    print(f"[cfg] {vars(args)}", flush=True)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if "," in args.root:                       # MULTIROOT pretraining corpus
        roots = args.root.split(",")
        trs = [RegionDataset("train", r) for r in roots]
        tr = torch.utils.data.ConcatDataset(trs)
        tr.files = [f for t in trs for f in t.files]   # for the frac-subsample path
        va, te = RegionDataset("val", roots[0]), RegionDataset("test", roots[0])
    else:
        tr = RegionDataset("train", args.root)
        va, te = RegionDataset("val", args.root), RegionDataset("test", args.root)
    if args.frac < 1.0:
        rng = np.random.default_rng(args.seed)
        k = max(args.bs * 2, int(round(len(tr.files) * args.frac)))
        tr.files = [tr.files[i] for i in rng.permutation(len(tr.files))[:k]]
    print(f"[data] train={len(tr)} (frac={args.frac}) val={len(va)} test={len(te)}",
          flush=True)
    dl_tr = torch.utils.data.DataLoader(tr, args.bs, shuffle=True, num_workers=10,
                                        collate_fn=collate, drop_last=True,
                                        persistent_workers=True)
    dl_va = torch.utils.data.DataLoader(va, 256, num_workers=6, collate_fn=collate)
    dl_te = torch.utils.data.DataLoader(te, 256, num_workers=6, collate_fn=collate)

    xs, ys_all = [], []
    for i in range(0, len(tr), max(1, len(tr) // 3000)):
        g, _, _ = tr[i]
        xs.append(g.ndata["x"]); ys_all.append(g.ndata["y"])
    xs = torch.cat(xs)
    mu, sd = xs.mean(0), xs.std(0).clamp_min(1e-6)
    in_dim = xs.shape[1]

    w = None
    if args.cb > 0:
        cnt = torch.bincount(torch.cat(ys_all), minlength=N_CLASSES).float()
        w = (cnt.clamp_min(1).sum() / cnt.clamp_min(1)) ** args.cb
        w = (w / w.mean()).to(DEV)
        print(f"[cb] weight range {w.min():.2f}..{w.max():.2f}", flush=True)

    GROUPS = {"type_params": (0, 16), "pmoments": (16, 20), "normal": (20, 23),
              "orient": (23, 29), "curv": (29, 32), "boundary": (32, 35),
              "moments2": (35, 41)}
    drop_idx = []
    for gname in [x for x in args.drop_groups.split(",") if x]:
        lo, hi = GROUPS[gname]; drop_idx += list(range(lo, min(hi, in_dim)))
    if drop_idx:
        print(f"[ablate] zeroing feature cols {drop_idx}", flush=True)
    DROP = torch.tensor(drop_idx, dtype=torch.long, device=DEV) if drop_idx else None

    edim = tr[0][0].edata["e"].shape[1]
    print(f"[edges] dim={edim}", flush=True)
    model = RegionNet(in_dim, args.d, args.layers, args.heads, edge_dim=edim).to(DEV)
    if args.init and os.path.exists(args.init):
        ck = torch.load(args.init, map_location=DEV)
        sd_in = ck["model"]
        if args.trunk_only:
            sd_in = {k: v for k, v in sd_in.items() if not k.startswith("head.")}
        missing, unexp = model.load_state_dict(sd_in, strict=False)
        print(f"[init] from {args.init}  missing={len(missing)} unexpected={len(unexp)}",
              flush=True)
    print(f"[model] in_dim={in_dim} params={sum(p.numel() for p in model.parameters())/1e6:.2f}M",
          flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr,
                                              total_steps=args.epochs * len(dl_tr),
                                              pct_start=0.05)
    ema = EMA(model, args.ema)
    MU, SD = mu.to(DEV), sd.to(DEV)
    best, best_ep = -1.0, -1
    ckpt = paths.ckpt(f"region_{args.tag}")
    t0 = time.time()
    for ep in range(args.epochs):
        model.train(); losses = []
        for g, f2r, y in dl_tr:
            g = g.to(DEV)
            x = (g.ndata["x"] - MU) / SD
            if DROP is not None:
                x = x.index_fill(1, DROP, 0.0)
            ee = torch.zeros_like(g.edata["e"]) if args.no_edge else g.edata["e"]
            logit = model(g, x, ee)
            loss = F.cross_entropy(logit, g.ndata["y"], weight=w,
                                   label_smoothing=args.smooth)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); ema.update(model); losses.append(loss.item())
        if (ep + 1) % 10 == 0 or ep == args.epochs - 1:
            vacc, vmiou = run_eval(ema.shadow, dl_va, MU, SD)
            if vmiou > best:
                best, best_ep = vmiou, ep
                import subprocess as _sp
                try:
                    _sha = _sp.run(["git", "rev-parse", "--short", "HEAD"],
                                   cwd=os.path.dirname(os.path.abspath(__file__)),
                                   capture_output=True, text=True).stdout.strip()
                    _dirty = bool(_sp.run(["git", "status", "--porcelain", "-uno"],
                                          cwd=os.path.dirname(os.path.abspath(__file__)),
                                          capture_output=True, text=True).stdout.strip())
                except Exception:
                    _sha, _dirty = "unknown", False
                torch.save({"model": ema.shadow.state_dict(), "mu": mu, "sd": sd,
                            "cfg": vars(args), "in_dim": in_dim,
                            # provenance
                            "code_sha": _sha + ("+dirty" if _dirty else ""),
                            "data_root": os.path.abspath(args.root)}, ckpt)
            print(f"[ep {ep+1}] loss={np.mean(losses):.4f} val_acc={vacc:.4f} "
                  f"val_mIoU={vmiou:.4f} best={best:.4f} ({(time.time()-t0)/60:.1f}m)",
                  flush=True)

    ck = torch.load(ckpt, map_location=DEV)
    best_model = RegionNet(in_dim, args.d, args.layers, args.heads, edge_dim=edim).to(DEV)
    best_model.load_state_dict(ck["model"])
    # eval() matters: the head and the per-layer MLPs carry Dropout, so without
    # it this final number would be computed with dropout active and differ
    # from eval_region_ckpt.py on the same checkpoint (up to 0.006 macro mIoU
    # on Fusion 360, whose smallest class has 75 faces). run_eval() above
    # already switches to eval mode for the validation passes.
    best_model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for g, f2r, y in dl_te:
            g = g.to(DEV)
            x = (g.ndata["x"] - MU) / SD
            if DROP is not None:
                x = x.index_fill(1, DROP, 0.0)
            ee = torch.zeros_like(g.edata["e"]) if args.no_edge else g.edata["e"]
            pf = best_model(g, x, ee).argmax(-1)[f2r.to(DEV)]
            ys.append(y.numpy()); ps.append(pf.cpu().numpy())
    yy, pp = np.concatenate(ys), np.concatenate(ps)
    tacc, tmiou = float((yy == pp).mean()), macro_miou(yy, pp)
    print("=" * 64, flush=True)
    print(f"SUMMARY tag={args.tag}  test acc={tacc:.4f}  mIoU={tmiou:.4f}  "
          f"(best val {best:.4f} @ep{best_ep+1})", flush=True)
    print(f"  AAGNet clean 0.9953 / 0.9851", flush=True)


if __name__ == "__main__":
    main()
