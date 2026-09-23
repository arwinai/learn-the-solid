"""Resumable, fault-isolated driver for BRepNet's STEP -> npz feature extraction.

BRepNet ships `pipeline/extract_brepnet_data_from_step.py`, which maps its
extractor over a directory with `ProcessPoolExecutor.map`; an exception inside
a worker propagates out of the consuming loop and ends the whole run, and
anything not yet extracted is lost. On large benchmarks (MFCAD++, CADSynth) a
small number of parts trigger such exceptions inside the extractor, so this
driver keeps the baseline's own `BRepNetExtractor` untouched and replaces only
the loop around it.

Two kinds of bad part are survived, and they need different machinery:

  * a part that *raises*. Caught in the worker and recorded; costs one part.
  * a part that *kills its worker*. A malformed or pathological STEP can make
    OpenCASCADE abort the process, which no `try/except` in the worker can
    intercept; `ProcessPoolExecutor` only reports `BrokenProcessPool`, without
    saying which part was responsible. Rebuilding the pool alone is not enough:
    the killer is still in the queue, so it kills the next pool too. Each
    worker therefore records the part it is holding in a claim file, so after
    a death the parts that were in flight are known; they are then retried one
    at a time, where a death names exactly one part and it can be blacklisted.

Additional safeguards:

  * stall detection -- if no part completes within `--stall-timeout` seconds the
    run aborts rather than hanging indefinitely;
  * resume -- parts with a readable npz are skipped;
  * corruption repair -- an npz left truncated by a killed worker is deleted so
    it gets re-extracted, instead of entering training as a short file;
  * a disk floor checked *during* extraction, not only between stages;
  * no orphans -- pool workers are killed by pid, since killing the parent
    leaves them alive and still writing;
  * an intake-failure manifest (intake_failures.json), recording how many
    parts the baseline's extractor cannot read -- the same disclosure made for
    the files AAGNet's extractor rejects.

Exit codes:
  0  every part accounted for, failures (if any) within `--max-fail-frac`
  2  too many parts failed: the resulting subset would be biased, do not train
  3  stopped early -- disk floor, stall, or unresolvable pool deaths; state on
     disk is consistent, rerun to resume

Requires a BRepNet checkout (paths.BREPNET or CINV_BREPNET) and its
environment (pythonocc).

Usage:
  python brepnet_extract_robust.py --step_path DIR --output DIR \
      [--feature_list JSON] [--num_workers N] [--floor-gb G] \
      [--stall-timeout S] [--max-fail-frac F] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths

BN = paths.BREPNET
sys.path.insert(0, BN)

# A required key: every npz the extractor writes has per-face features. Reading
# it is enough to prove the file was closed properly.
REQUIRED_KEY = "face_features"
CLAIMS = ".claims"
HARD = "WorkerDied: extractor aborted the process (uncatchable in-worker)"


def _worker(task):
    """Extract one part. Returns (stem, ok, error-string).

    Every exception is caught here so that no single part can reach the parent
    as a raise. The claim file is what lets the parent attribute a hard crash,
    which bypasses this handler entirely.
    """
    # pythonocc emits a DeprecationWarning per face per part, which the parent's
    # PYTHONWARNINGS does not reach and which would flood the log. Silence it
    # in the worker, where the calls actually happen.
    import warnings
    warnings.filterwarnings("ignore")

    path, out_dir, schema = task
    stem = Path(path).stem
    claim = Path(out_dir) / CLAIMS / f"{os.getpid()}.claim"
    try:
        claim.write_text(stem)
    except OSError:
        pass
    try:
        from pipeline.extract_brepnet_data_from_step import BRepNetExtractor
        BRepNetExtractor(Path(path), Path(out_dir), schema).process()
    except BaseException as e:                      # noqa: BLE001 -- deliberate
        return stem, False, f"{type(e).__name__}: {e}"
    finally:
        try:
            claim.unlink(missing_ok=True)
        except OSError:
            pass
    # The extractor returns quietly on bodies it declines to handle (for
    # instance a coedge used by several loops), so absence of the file is a
    # failure even when nothing was raised.
    if not (Path(out_dir) / f"{stem}.npz").exists():
        return stem, False, "Declined: no npz written (unsupported topology)"
    return stem, True, ""


def free_gb(path):
    return shutil.disk_usage(path).free / 2**30


def validate_existing(out_dir, names):
    """Names already extracted, with unreadable npz deleted so they re-run."""
    import numpy as np
    done, repaired = set(), 0
    for stem in names:
        f = out_dir / f"{stem}.npz"
        if not f.exists():
            continue
        try:
            with np.load(f) as d:
                _ = d[REQUIRED_KEY].shape
            done.add(stem)
        except Exception:
            f.unlink(missing_ok=True)
            repaired += 1
    return done, repaired


def _snapshot_pids(ex, pids):
    """Record worker pids while the pool is alive.

    `ProcessPoolExecutor` clears `_processes` during shutdown, so the pids have
    to be collected beforehand or there is nothing left to kill.
    """
    for p in (getattr(ex, "_processes", None) or {}).values():
        pids.add(p.pid)
    return pids


def _kill(pids):
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def _read_claims(out_dir):
    """Stems that were in flight, from the claim files left behind."""
    d, out = out_dir / CLAIMS, set()
    for f in d.glob("*.claim"):
        try:
            out.add(f.read_text().strip())
        except OSError:
            pass
        f.unlink(missing_ok=True)
    return {s for s in out if s and not (out_dir / f"{s}.npz").exists()}


def _isolate(path, out_dir, schema):
    """Run one part in a pool of its own: a death here names that part.

    Returns (ok, error-string). This is the only way to tell which part aborted
    a shared pool, so it is worth the extra process spawn.
    """
    ex = ProcessPoolExecutor(max_workers=1)
    pids = set()
    try:
        fu = ex.submit(_worker, (str(path), str(out_dir), schema))
        _snapshot_pids(ex, pids)
        _, ok, err = fu.result()
        return ok, err
    except BrokenProcessPool:
        return False, HARD
    except BaseException as e:                      # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"
    finally:
        _snapshot_pids(ex, pids)
        ex.shutdown(wait=False, cancel_futures=True)
        _kill(pids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step_path", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--feature_list", default=f"{BN}/feature_lists/all.json")
    ap.add_argument("--num_workers", type=int, default=24)
    ap.add_argument("--floor-gb", type=float, default=8.0,
                    help="stop cleanly when free space drops below this")
    ap.add_argument("--stall-timeout", type=float, default=900.0,
                    help="abort if no part completes within this many seconds")
    ap.add_argument("--max-fail-frac", type=float, default=0.02,
                    help="above this share of failures, exit 2 (biased subset)")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N parts")
    a = ap.parse_args()

    step_path, out_dir = Path(a.step_path), Path(a.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / CLAIMS).mkdir(exist_ok=True)
    for f in (out_dir / CLAIMS).glob("*.claim"):
        f.unlink(missing_ok=True)
    schema = json.load(open(a.feature_list))

    files = sorted(list(step_path.glob("**/*.stp"))
                   + list(step_path.glob("**/*.step")))
    if a.limit:
        files = files[:a.limit]
    if not files:
        print(f"FATAL: no STEP files under {step_path}", flush=True)
        return 2

    done, repaired = validate_existing(out_dir, [f.stem for f in files])
    todo = [f for f in files if f.stem not in done]
    print(f"total {len(files)} | already done {len(done)} | corrupt removed "
          f"{repaired} | to extract {len(todo)} | free {free_gb(out_dir):.1f}G",
          flush=True)

    manifest = out_dir / "intake_failures.json"
    failures = {}
    if manifest.exists():
        try:
            failures = json.load(open(manifest)).get("failures", {})
        except Exception:
            failures = {}

    def save(status):
        json.dump({"status": status, "n_total": len(files),
                   "n_extracted": len(done), "n_failed": len(failures),
                   "failures": failures},
                  open(manifest, "w"), indent=1)

    t0, rc, deaths = time.time(), 0, 0
    remaining, next_report = list(todo), 500

    while remaining and rc == 0:
        ex = ProcessPoolExecutor(max_workers=a.num_workers)
        futs = {ex.submit(_worker, (str(f), str(out_dir), schema)): f
                for f in remaining}
        pending, stopped, pids = set(futs), None, set()
        try:
            while pending:
                fin, pending = wait(pending, timeout=a.stall_timeout,
                                    return_when=FIRST_COMPLETED)
                _snapshot_pids(ex, pids)
                if not fin:
                    stopped = ("stall",
                               f"no part finished in {a.stall_timeout:.0f}s")
                    break
                for fu in fin:
                    stem, ok, err = fu.result()
                    if ok:
                        done.add(stem)
                        failures.pop(stem, None)
                    else:
                        failures[stem] = err
                    del futs[fu]
                if len(done) + len(failures) >= next_report:
                    next_report = len(done) + len(failures) + 500
                    el = time.time() - t0
                    print(f"  {len(done)}/{len(files)} ok, {len(failures)} "
                          f"failed, {el:.0f}s elapsed, free "
                          f"{free_gb(out_dir):.1f}G", flush=True)
                    save("running")
                if free_gb(out_dir) < a.floor_gb:
                    stopped = ("disk", f"free below {a.floor_gb}G floor")
                    break
            remaining = [futs[f] for f in pending if f in futs] if stopped else []
        except BrokenProcessPool as e:
            # A worker died hard. Resolve the in-flight parts individually so
            # the killer is identified and cannot poison the next pool.
            deaths += 1
            suspects = _read_claims(out_dir)
            print(f"  pool died ({e}); death {deaths}; "
                  f"{len(suspects)} part(s) in flight, isolating", flush=True)
            by_stem = {f.stem: f for f in remaining}
            for stem in sorted(suspects):
                f = by_stem.get(stem)
                if f is None:
                    continue
                ok, err = _isolate(f, out_dir, schema)
                if ok:
                    done.add(stem)
                    failures.pop(stem, None)
                else:
                    failures[stem] = err
                    print(f"    {stem}: {err}", flush=True)
            got, _ = validate_existing(out_dir, [f.stem for f in remaining])
            done |= got
            remaining = [f for f in remaining
                         if f.stem not in done and f.stem not in failures]
            save("running")
            if deaths > max(20, 0.001 * len(files)):
                print("FATAL: pool deaths keep recurring", flush=True)
                save("aborted: repeated pool deaths")
                rc = 3
            continue
        finally:
            _snapshot_pids(ex, pids)
            ex.shutdown(wait=False, cancel_futures=True)
            _kill(pids)

        if stopped:
            why, msg = stopped
            print(f"STOPPED ({why}): {msg}", flush=True)
            save(f"stopped: {why}")
            rc = 3

    frac = len(failures) / max(1, len(files))
    print(f"\nextracted {len(done)}/{len(files)} parts, {len(failures)} intake "
          f"failures ({100 * frac:.3f}%), {deaths} pool death(s), "
          f"{time.time() - t0:.0f}s", flush=True)
    if failures:
        kinds = {}
        for e in failures.values():
            k = str(e).split(":")[0]
            kinds[k] = kinds.get(k, 0) + 1
        for k, v in sorted(kinds.items(), key=lambda x: -x[1]):
            print(f"  {v:6d}  {k}", flush=True)
    if rc == 3:
        save("stopped")
        return 3
    if frac > a.max_fail_frac:
        print(f"FATAL: failure share {100 * frac:.2f}% exceeds the "
              f"{100 * a.max_fail_frac:.2f}% gate; the surviving subset is not "
              f"a fair sample of the benchmark and must not be trained on.",
              flush=True)
        save("failed gate")
        return 2
    save("complete")
    print("EXTRACTION COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
