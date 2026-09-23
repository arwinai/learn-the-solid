"""Single place where filesystem locations are resolved.

Every location comes from an environment variable with a repository-relative
default, so the code runs from a checkout without editing.

  CINV_DATA      datasets root (default ./data)
  CINV_CKPT      checkpoints  (default ./ckpt)
  CINV_AAGNET    a checkout of the AAGNet baseline, needed for the baseline
                 comparisons and for its STEP->graph extractor
                 (default ./third_party/AAGNet)
  CINV_UVNET     a checkout of UV-Net, needed only to train/score that
                 baseline (default ./third_party/UV-Net)
  CINV_BREPNET   a checkout of BRepNet, needed only to train/score that
                 baseline (default ./third_party/BRepNet)

Per-dataset roots are <CINV_DATA>/<name>, each holding steps/ plus the
region_graphs*/ and eval_*/ directories the scripts create.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

DATA = os.environ.get("CINV_DATA", os.path.join(ROOT, "data"))
CKPT = os.environ.get("CINV_CKPT", os.path.join(ROOT, "ckpt"))
AAGNET = os.environ.get("CINV_AAGNET", os.path.join(ROOT, "third_party", "AAGNet"))
UVNET = os.environ.get("CINV_UVNET", os.path.join(ROOT, "third_party", "UV-Net"))
BREPNET = os.environ.get("CINV_BREPNET", os.path.join(ROOT, "third_party", "BRepNet"))


def dataset(name):
    """Root of one dataset, e.g. dataset('mfinstseg')."""
    return os.path.join(DATA, name)


def ckpt(name):
    """Path to a checkpoint file by stem, e.g. ckpt('region_mfinstseg')."""
    return os.path.join(CKPT, name if name.endswith(".pt") else name + ".pt")
