"""Prediction churn for the graph-path models on every perturbation cell.

Measures, for AAGNet, UV-Net and our region model, the fraction of faces whose
predicted label changes between the clean B-rep and each perturbed version of
the same solid (re-partition families A/B, rotation, NURBS re-expression and
their composition) on the 3,000-solid MFInstSeg protocol. Perturbed side: the
cell pkls (which carry the child->parent face mapping and an AAGNet-format
graph). Clean side: eval_clean pkls for the baselines and the region_graphs
cache for ours. Rejected transfers (parent = -1) are masked identically for
all models. The checkpoint, the clean cache and every cell must carry the same
generation suffix (override with ALLOW_MIXED_GEN=1); cells can be overridden
with CHURN_FAMS (JSON), the checkpoint with OURS_CK and the clean cache with
CLEAN_RG. Learning environment (torch/dgl), with the AAGNet and UV-Net
checkouts on the path.

Usage: python paired_churn_all.py [--limit 3000]
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import re
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch
import dgl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths                                           # noqa: E402
from train_baseline import make, fwd, aag_graph          # noqa: E402
from region_train import RegionNet                     # noqa: E402

ROOT = os.environ.get("DS_ROOT", paths.dataset("mfinstseg"))
DEV = torch.device("cuda")

import json as _json
FAMS = (_json.loads(os.environ["CHURN_FAMS"]) if "CHURN_FAMS" in os.environ
        else {"rpA": "eval_rpA_pfr_iso4", "rpB": "eval_rpB_pfr_iso4",
              "rot": "eval_rot_pfr_iso4", "nc": "eval_nc_pfr_iso5",
              "combo": "eval_combo2_pfr_iso5"})


def face_preds_base(m, arch, pkl_dict):
    y = pkl_dict.get("y", np.zeros(1))
    aag = pkl_dict.get("aag")
    if aag is None:
        return None
    ny = aag["graph"]["num_nodes"]
    g = aag_graph(aag, np.zeros(ny, np.int64))
    if g is None:
        return None
    g = g.to(DEV)
    with torch.no_grad():
        o = fwd(m, arch, g)
    return o.argmax(-1).cpu().numpy()


@torch.no_grad()
def face_preds_ours(rm, MU, SD, d):
    g = dgl.graph((d["src"], d["dst"]), num_nodes=d["feats"].shape[0]).to(DEV)
    x = (torch.from_numpy(np.asarray(d["feats"])).float().to(DEV) - MU) / SD
    o = rm(g, x, torch.from_numpy(np.asarray(d["ef"])).float().to(DEV))
    p = o.argmax(-1)
    return p[torch.from_numpy(np.asarray(d["f2r"])).to(DEV)].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=3000)
    a = ap.parse_args()

    models = {}
    for arch, ckp in (("aagnet", os.path.join(paths.AAGNET, "weights",
                                              "weight_on_MFInstseg.pth")),
                      ("uvnet", os.path.join(paths.CKPT, "uvnet_mfi.pth"))):
        m = make(arch)
        sd = torch.load(ckp, map_location=DEV)
        m.load_state_dict(sd.get("model", sd.get("state_dict", sd)),
                          strict=False)
        m.eval()
        models[arch] = m

    ours_ck = os.environ.get("OURS_CK", paths.ckpt("region_mfinstseg_pfr_iso4"))
    clean_rg = os.environ.get("CLEAN_RG", "region_graphs_pfr_iso4")
    # Refuse cross-generation comparisons: the checkpoint, the clean cache and
    # every cell must carry the SAME generation suffix, otherwise the "churn"
    # measures a difference between extractor versions rather than the
    # perturbation. Resolved paths are printed for provenance.
    def gen_of(name):
        m = re.search(r"pfr_iso\w+?(?=_s\d+\b|\.|$)", name)
        # seed suffixes (_s1/_s2) belong to the same generation
        return m.group(0) if m else None
    gens = {gen_of(ours_ck), gen_of(clean_rg), *(gen_of(v) for v in FAMS.values())}
    print(f"[churn] ckpt={ours_ck} clean={clean_rg} fams={FAMS}", flush=True)
    if (len(gens) != 1 or None in gens) and not os.environ.get("ALLOW_MIXED_GEN"):
        # None = untagged input: reject rather than silently matching
        raise SystemExit(f"generation mismatch or untagged input: {gens} "
                         f"(set ALLOW_MIXED_GEN=1 to override)")
    ck = torch.load(os.path.join(HERE, ours_ck), map_location=DEV)
    cfg = ck.get("cfg", {})
    edim = ck["model"]["convs.0.fc_edge.weight"].shape[1]
    rm = RegionNet(ck["in_dim"], cfg.get("d", 384), cfg.get("layers", 6),
                   cfg.get("heads", 6), edge_dim=edim).to(DEV)
    rm.load_state_dict(ck["model"])
    rm.eval()
    MU, SD = ck["mu"].to(DEV), ck["sd"].to(DEV)

    for fam, cell in FAMS.items():
        files = sorted(glob.glob(os.path.join(ROOT, cell, "*.pkl")))[:a.limit]
        st = {k: [0, 0, 0, 0] for k in ("aagnet", "uvnet", "ours")}
        for fp in files:
            nm = os.path.basename(fp)[:-4]
            d = pickle.load(open(fp, "rb"))
            par = d.get("parent")
            if par is None:
                continue
            par = np.asarray(par)
            cf = os.path.join(ROOT, "eval_clean", nm + ".pkl")
            cz = os.path.join(ROOT, clean_rg, nm + ".npz")
            # baselines
            if os.path.exists(cf):
                c = pickle.load(open(cf, "rb"))
                for arch, m in models.items():
                    pc = face_preds_base(m, arch, c)
                    pp = face_preds_base(m, arch, d)
                    if pc is None or pp is None or par.max() >= len(pc) \
                            or len(pp) != len(par):
                        continue
                    vp = par >= 0   # rejected transfers are -1;
                                    # identical support for all models
                    same = int((pp[vp] == pc[par[vp]]).sum())
                    t = st[arch]
                    t[0] += same; t[1] += int(vp.sum())
                    t[2] += int(same == int(vp.sum())); t[3] += 1
            # ours, primary model
            if os.path.exists(cz):
                with np.load(cz) as z:
                    cdict = {k: z[k] for k in
                             ("feats", "src", "dst", "ef", "f2r")}
                pc = face_preds_ours(rm, MU, SD, cdict)
                pp = face_preds_ours(rm, MU, SD, d)
                if par.max() < len(pc) and len(pp) == len(par):
                    vp = par >= 0   # rejected transfers are -1;
                                    # identical support for all models
                    same = int((pp[vp] == pc[par[vp]]).sum())
                    t = st["ours"]
                    t[0] += same; t[1] += int(vp.sum())
                    t[2] += int(same == int(vp.sum())); t[3] += 1
        for k, (ag, tot, stab, n) in st.items():
            churn_f = 100 * (1 - ag / max(1, tot))
            churn_s = 100 * (1 - stab / max(1, n))
            print(f"SUMMARY churn {k} {fam} faces_flipped={churn_f:.2f}% "
                  f"solids_affected={churn_s:.2f}% n={n}", flush=True)


if __name__ == "__main__":
    main()
