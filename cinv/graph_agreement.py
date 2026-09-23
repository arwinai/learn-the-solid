"""Canonical-region-graph agreement at benchmark scale (MFInstSeg test).

For each perturbation directory, every perturbed region graph is compared
against its clean counterpart:

  (a) identical region count?
  (b) if identical, residuals in STANDARDIZED units: each channel is divided
      by its training standard deviation (verification_musd.npz, exported
      from the primary checkpoint), so a residual of 1 means the feature
      moved by as much as it typically differs between two different parts.
      Channels constant in training (sd at the 1e-6 clamp floor) are
      excluded -- the model cannot read a constant channel.

Feature families (channel indices in the 115-dim pfr layout):
  EXACT   -- integrals, surface parameters, counts: invariant by
             construction/theory (type one-hot, area, params, principal
             moments, curvature integrals, loop/boundary-length channels).
  SAMPLED -- positions expressed in the blended canonical frame plus
             histogram and spatial-grid channels, all computed by sampling
             the boundary mesh: Lipschitz under mesh perturbation, drifting
             with boundary resampling.

Matching protocol (two failure modes force the lexicographic form):
  matching on ALL channels lets sampled-channel noise swap similar regions
  (fakes exact-channel changes under rotation); matching on EXACT channels
  alone cannot distinguish symmetric twin regions (identical exact features,
  mirrored grids). Regions are therefore paired primarily by exact-feature
  distance, ties broken by all-feature distance.

Structure is compared as well: the matched node bijection is applied to both
graphs' edge sets (adjacency must be identical as a permuted edge set, edge
features are compared per matched edge), and when the Hungarian assignment
fails the adjacency test a colored graph isomorphism search (VF2) decides
whether any structure-preserving bijection exists.

Inputs: clean region graphs (<dataset>/region_graphs_<CLEAN_SFX>/*.npz),
perturbed region graphs (<dataset>/eval_<cell>_<PERT_SFX>/*.pkl) and the
training channel statistics verification_musd.npz next to this file.

Output per directory: identical-structure %, then p50/p90/p99 over pairs of
each pair's largest channel residual, per family, plus adjacency agreement,
edge-feature residuals and the raw drift of train-constant channels.

Usage: python graph_agreement.py [max_parts]
"""
import os
import pickle
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths

DATA = os.environ.get("DS_ROOT", paths.dataset("mfinstseg"))
# CLEAN_SFX names the clean graph dir, PERT_SFX the perturbation dirs, so a
# feature-pipeline generation that only affects some cells can be measured
# against the unchanged clean graphs. Cells that do not exist under the
# requested suffix report MISSING and are skipped.
CLEAN_SFX = os.environ.get("CLEAN_SFX", "pfr_iso4")
PERT_SFX = os.environ.get("PERT_SFX", "pfr_iso4")
CLEAN = os.path.join(DATA, f"region_graphs_{CLEAN_SFX}")
PERTURB_DIRS = [f"eval_{c}_{PERT_SFX}"
                for c in ("rpB", "nc", "rot", "combo2")]
DEFAULT_MAX_PARTS = 3000

_musd = np.load(os.path.join(HERE, "verification_musd.npz"))
MU, SD = _musd["mu"], _musd["sd"]
LIVE = SD > 2e-6                       # drop train-constant channels
IDX = np.where(LIVE)[0]
# EXACT = analytically accumulated, no mesh anywhere on their path, and not
# expressed in the canonical frame: surface type one-hot + area (0-6), surface
# parameters (13-19), loop count (32), boundary length (33).
# Deliberately NOT exact:
#   7-12   centroid/axis      -- expressed in the blended canonical frame
#   20-28  int n dA, int nxn  -- triangle quadrature (Lemma 4, Lipschitz)
#   29-31  mean H, K, H^2     -- triangle quadrature (Lemma 4, Lipschitz)
#   34-36  sharpness shares   -- hard dihedral threshold (Lemma 4's non-Lipschitz case)
#   37-42  second-moment tens -- exact integral, but frame-expressed
EXACT_CH = list(range(0, 7)) + list(range(13, 20)) + [32, 33]
EXACT = np.isin(IDX, EXACT_CH)


def standardize(feats):
    return (np.asarray(feats, np.float64)[:, LIVE] - MU[LIVE]) / SD[LIVE]


def compare(clean_feats, pert_feats, clean_edges=None, pert_edges=None):
    """(counts_equal, exact max residual, sampled max residual,
    adjacency_ok, edge-feature max residual, const-channel max drift).

    Nodes are paired by lexicographic Hungarian matching on standardized
    features. The resulting bijection is applied to both graphs' edge sets:
    adjacency must be identical as a permuted edge set, and edge features are
    compared per matched edge. Train-constant channels are excluded from the
    standardized residuals and their raw drift is reported separately (a
    channel constant in training can still move at test time).
    clean_edges/pert_edges: (src, dst, ef) triples or None (feature-only)."""
    if clean_feats.shape[0] != pert_feats.shape[0]:
        return False, None, None, None, None, None
    a, b = standardize(clean_feats), standardize(pert_feats)
    d2 = (a[:, None, :] - b[None, :, :]) ** 2
    cost = d2[:, :, EXACT].sum(-1) * 1e6 + d2.sum(-1)   # lexicographic
    ri, ci = linear_sum_assignment(cost)
    d = np.abs(a[ri] - b[ci])
    # constant-channel drift, raw units (SD too small to standardize by)
    rawc = np.abs(np.asarray(clean_feats, np.float64)[ri][:, ~LIVE]
                  - np.asarray(pert_feats, np.float64)[ci][:, ~LIVE])
    const_drift = float(rawc.max()) if rawc.size else 0.0
    adj_ok, e_res = None, None
    if clean_edges is not None and pert_edges is not None:
        perm = np.empty(len(ri), np.int64)
        perm[ri] = ci                      # clean node -> pert node
        cs, cd, cef = clean_edges
        ps, pd, pef = pert_edges
        E1 = {}
        for k in range(len(cs)):
            E1[(int(perm[cs[k]]), int(perm[cd[k]]))] = k
        E2 = {(int(ps[k]), int(pd[k])): k for k in range(len(ps))}
        adj_ok = set(E1.keys()) == set(E2.keys())
        if adj_ok and len(E1):
            cef = np.asarray(cef, np.float64)
            pef = np.asarray(pef, np.float64)
            res = [np.abs(cef[E1[e]] - pef[E2[e]]).max() for e in E1]
            e_res = float(max(res))
        elif adj_ok:
            e_res = 0.0
    return (True, float(d[:, EXACT].max()), float(d[:, ~EXACT].max()),
            adj_ok, e_res, const_drift)


def iso_bijection_exists(clean_feats, pert_feats, clean_edges, pert_edges,
                         round_sigma=0.01):
    """Search for an adjacency-preserving bijection that also matches node
    features (rounded to round_sigma in standardized units). The Hungarian
    match can swap feature-identical symmetric nodes and then fail the
    adjacency test spuriously; colored graph isomorphism (VF2) is the
    criterion for 'identical structure'.

    Returns the clean->pert node mapping (dict) if one exists, else None. The
    mapping is returned rather than a boolean so the caller can recompute every
    node and edge residual under one common bijection."""
    import networkx as nx
    a, b = standardize(clean_feats), standardize(pert_feats)
    if a.shape != b.shape:
        return False

    def build(feats, edges):
        g = nx.Graph()
        for i in range(feats.shape[0]):
            color = tuple(np.round(feats[i, EXACT] / round_sigma).astype(int))
            g.add_node(i, c=color)
        src, dst, _ = edges
        for k in range(len(src)):
            g.add_edge(int(src[k]), int(dst[k]))
        return g
    g1 = build(a, clean_edges)
    g2 = build(b, pert_edges)
    gm = nx.algorithms.isomorphism.GraphMatcher(
        g1, g2, node_match=lambda x, y: x["c"] == y["c"])
    if not gm.is_isomorphic():
        return None
    return {int(k): int(v) for k, v in gm.mapping.items()}


def residuals_under_mapping(clean_feats, pert_feats, clean_edges, pert_edges,
                            mapping):
    """All of compare()'s statistics recomputed under an explicit clean->pert
    bijection, so that every reported statistic describes one common mapping.
    Directed edge keys, mirroring compare().
    Returns (exact_res, sampled_res, const_drift, edge_res, adj_ok)."""
    a, b = standardize(clean_feats), standardize(pert_feats)
    n = a.shape[0]
    perm = np.array([mapping[i] for i in range(n)], np.int64)
    d = np.abs(a - b[perm])
    rawc = np.abs(np.asarray(clean_feats, np.float64)[:, ~LIVE]
                  - np.asarray(pert_feats, np.float64)[perm][:, ~LIVE])
    const_drift = float(rawc.max()) if rawc.size else 0.0
    cs, cd, cef = clean_edges
    ps, pd, pef = pert_edges
    E1 = {(int(perm[cs[k]]), int(perm[cd[k]])): k for k in range(len(cs))}
    E2 = {(int(ps[k]), int(pd[k])): k for k in range(len(ps))}
    adj_ok = set(E1.keys()) == set(E2.keys())
    e_res = None
    if adj_ok and len(E1):
        cef = np.asarray(cef, np.float64)
        pef = np.asarray(pef, np.float64)
        e_res = float(max(np.abs(cef[E1[e]] - pef[E2[e]]).max() for e in E1))
    elif adj_ok:
        e_res = 0.0
    return (float(d[:, EXACT].max()), float(d[:, ~EXACT].max()),
            const_drift, e_res, adj_ok)


def main(max_parts=DEFAULT_MAX_PARTS):
    for pdir in PERTURB_DIRS:
        full = os.path.join(DATA, pdir)
        if not os.path.isdir(full):
            print(f"{pdir}: MISSING, skipped")
            continue
        names = sorted(f[:-4] for f in os.listdir(full) if f.endswith(".pkl"))
        names = [n for n in names
                 if os.path.exists(os.path.join(CLEAN, n + ".npz"))][:max_parts]
        n_cmp = n_equal = 0
        ex, sa, matched = [], [], []
        adj_oks, edge_res, cdrifts = [], [], []
        for n in names:
            z = np.load(os.path.join(CLEAN, n + ".npz"))
            with open(os.path.join(full, n + ".pkl"), "rb") as fh:
                p = pickle.load(fh)
            eq, e, s, adj_ok, e_res, cdrift = compare(
                np.asarray(z["feats"]), np.asarray(p["feats"]),
                clean_edges=(np.asarray(z["src"]), np.asarray(z["dst"]),
                             np.asarray(z["ef"])),
                pert_edges=(np.asarray(p["src"]), np.asarray(p["dst"]),
                            np.asarray(p["ef"])))
            n_cmp += 1
            if eq:
                n_equal += 1
                matched.append(n)
                if not adj_ok:
                    # The Hungarian match may have swapped symmetric twins;
                    # the question is whether ANY structure-preserving
                    # bijection exists. When one does, every statistic
                    # appended below (exact, sampled, const-drift, edge
                    # residual) is recomputed under that single mapping,
                    # never mixed with the rejected Hungarian assignment.
                    ce = (np.asarray(z["src"]), np.asarray(z["dst"]),
                          np.asarray(z["ef"]))
                    pe_ = (np.asarray(p["src"]), np.asarray(p["dst"]),
                           np.asarray(p["ef"]))
                    mp = iso_bijection_exists(
                        np.asarray(z["feats"]), np.asarray(p["feats"]),
                        ce, pe_)
                    if mp is not None:
                        e, s, cdrift, e_res, adj_ok = residuals_under_mapping(
                            np.asarray(z["feats"]), np.asarray(p["feats"]),
                            ce, pe_, mp)
                    else:
                        adj_ok = False
                # append only after the mapping is final
                ex.append(e)
                sa.append(s)
                adj_oks.append(bool(adj_ok))
                if e_res is not None:
                    edge_res.append(e_res)
                cdrifts.append(cdrift)
        pct = 100.0 * n_equal / max(n_cmp, 1)

        def pcts(x):
            x = np.array(x)
            if not len(x):
                return "nan/nan/nan/nan"
            return "/".join([f"{np.percentile(x, q):.3g}" for q in (50, 90, 99)]
                            + [f"{x.max():.3g}"])

        print(f"{pdir}: N={n_cmp} identical_counts={n_equal} ({pct:.1f}%) "
              f"exact p50/p90/p99/max={pcts(ex)} | "
              f"sampled p50/p90/p99/max={pcts(sa)}")
        if adj_oks:
            print(f"    adjacency identical under bijection: "
                  f"{100.0 * sum(adj_oks) / len(adj_oks):.1f}% | "
                  f"edge-feature p50/p90/p99/max={pcts(edge_res)} | "
                  f"const-channel raw drift max={max(cdrifts):.3g}")
        if len(ex):
            i = int(np.argmax(ex))
            print(f"    worst-exact part: {matched[i]} = {ex[i]:.3g} sigma")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MAX_PARTS)
