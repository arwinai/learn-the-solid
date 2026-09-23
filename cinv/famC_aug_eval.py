"""Score augmentation-retrained checkpoints on a re-partition cell (family C).

eval_cells.py covers our model and the published baselines; this adds the
augmentation-retrained AAGNet / UV-Net checkpoints, reusing the trainers' own
dataset and evaluate code so the metric convention is identical. Prints
accuracy and macro mIoU per checkpoint. Learning environment (torch/dgl), with
the AAGNet and UV-Net checkouts on the path.

Usage: python famC_aug_eval.py <pkl-cell-dir> <ckpt> [<ckpt> ...]
Each ckpt is aagnet_*.pth or uvnet_*.pth; the architecture is inferred from
the file name.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch

CELL = sys.argv[1]
CKPTS = sys.argv[2:]

import aagnet_aug_train as A          # noqa: E402  (imports AAGNet stack)
import uvnet_aug_train as U           # noqa: E402

ds_a = A.PklGraphs(CELL)
ds_u = U.PklGraphs(CELL) if hasattr(U, "PklGraphs") else ds_a

for ck in CKPTS:
    nm = os.path.basename(ck)
    if not os.path.exists(ck):
        print(f"SUMMARY famC {nm}: MISSING", flush=True)
        continue
    if nm.startswith("aagnet"):
        m = A.AAGNetSegmentor(
            num_classes=A.N_CLASSES, arch="AAGNetGraphEncoder",
            edge_attr_dim=12, node_attr_dim=10, edge_attr_emb=64,
            node_attr_emb=64, edge_grid_dim=0, node_grid_dim=7,
            edge_grid_emb=0, node_grid_emb=64, num_layers=3, delta=2,
            mlp_ratio=2, drop=0.25, drop_path=0.25, head_hidden_dim=64,
            conv_on_edge=False, use_uv_gird=True, use_edge_attr=True,
            use_face_attr=True).to(A.DEV)
        sd = torch.load(ck, map_location=A.DEV)
        m.load_state_dict(sd.get("model", sd), strict=False)
        acc, miou = A.evaluate(m, ds_a)
    else:
        m = U.make("uvnet")
        sd = torch.load(ck, map_location=U.DEV)
        m.load_state_dict(sd.get("model", sd), strict=False)
        acc, miou = U.evaluate(m, ds_u)
    print(f"SUMMARY famC {nm}: acc={acc:.4f} mIoU={miou:.4f}", flush=True)
