#!/usr/bin/env python3
"""
Watcher that:
- loads a trained Autoencoder model
- tails inputs/live_features.csv in near-real-time
- computes reconstruction error as the anomaly score
- writes rows to outputs/scores_live.csv
- updates its threshold live by reading control/threshold.json written by the dashboard
- on exit, archives the run under artifacts/run_<id>/ and runs eval scripts if available
"""

import os
import sys
import time
import json
import shutil
import atexit
import argparse
import datetime
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

# ----- Optional: TensorFlow / Keras for the autoencoder -----
try:
    from tensorflow.keras.models import load_model
except Exception:
    load_model = None


# ---------- Resolve PROJECT_ROOT robustly ----------
_THIS = Path(__file__).resolve()
if _THIS.parent.name in ("scripts", "tools"):
    PROJECT_ROOT = _THIS.parents[1]
else:
    PROJECT_ROOT = _THIS.parent

MODELS_DIR   = PROJECT_ROOT / "models"
INPUTS_DIR   = PROJECT_ROOT / "inputs"
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"
CONTROL_DIR  = PROJECT_ROOT / "control"
TOOLS_DIR    = PROJECT_ROOT / "tools"
ARTIFACTS_DIR= PROJECT_ROOT / "artifacts"

INPUT_CSV    = INPUTS_DIR  / "live_features.csv"
OUTPUT_CSV   = OUTPUTS_DIR / "scores_live.csv"
CONTROL_PATH = CONTROL_DIR / "threshold.json"

# Ensure dirs exist
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
CONTROL_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Will be updated at runtime; used in manifest
_current_threshold = None


# ---------- Control-file threshold ----------
_last_thr_mtime = 0.0
def load_control_threshold(default_thr: float) -> float:
    """Read threshold from control/threshold.json if it changed."""
    global _last_thr_mtime
    try:
        mtime = CONTROL_PATH.stat().st_mtime
        if mtime != _last_thr_mtime:
            data = json.loads(CONTROL_PATH.read_text())
            thr = float(data.get("thr", default_thr))
            _last_thr_mtime = mtime
            print(f"[THR] Updated from control file ({CONTROL_PATH}): {thr:.4f}")
            return thr
    except FileNotFoundError:
        if _last_thr_mtime == 0.0:
            print(f"[THR] Control file not found yet: {CONTROL_PATH}")
    except Exception as e:
        print(f"[THR] Control read error: {e}")
    return default_thr


# ---------- Model / scoring ----------
def load_autoencoder(model_path: str):
    if load_model is None:
        raise RuntimeError("TensorFlow/Keras not available. Install tensorflow.")
    return load_model(model_path)

def reconstruction_error(model, x: np.ndarray) -> float:
    """Compute reconstruction error for one row (MSE)."""
    x = x.reshape(1, -1)
    recon = model.predict(x, verbose=0)
    return float(np.mean((x - recon) ** 2))


# ---- Optional: use a saved feature order (if present) ----
def _load_feature_columns(path=MODELS_DIR / "feature_columns.json"):
    try:
        cols = json.loads(Path(path).read_text())
        if isinstance(cols, list) and all(isinstance(c, str) for c in cols):
            print(f"[BOOT] Loaded feature order ({len(cols)} cols) from {path}")
            return cols
    except Exception:
        pass
    return None

FEATURE_COLUMNS = _load_feature_columns()

# ---- Series-safe conversion to model input ----
def features_from_row(row: pd.Series) -> np.ndarray:
    """
    Convert one CSV row (pandas Series) into a numeric vector for the model.
    - If FEATURE_COLUMNS exists, reindex to that exact order.
    - Otherwise, drop known non-feature columns and coerce remaining to numeric.
    """
    drop_these = {"ts", "index", "flag", "threshold", "label", "y", "anomaly"}
    s = row.copy()

    if FEATURE_COLUMNS:
        s = s.reindex(FEATURE_COLUMNS)
    else:
        s = s[[c for c in s.index if c not in drop_these]]

    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    return s.to_numpy(dtype=np.float32)


# ---------- Main tailing logic ----------
def tail_csv(path: Path, start_at_end: bool = True):
    """
    Generator that yields new rows appended to a CSV.
    It reloads the CSV periodically and yields rows after a maintained cursor.
    """
    last_len = 0
    while True:
        if (not path.exists()) or (path.stat().st_size == 0):
            time.sleep(0.5)
            continue

        try:
            df = pd.read_csv(path)
        except Exception:
            time.sleep(0.1)
            continue

        if start_at_end and last_len == 0:
            last_len = len(df)

        if len(df) > last_len:
            new = df.iloc[last_len:].copy()
            last_len = len(df)
            for _, r in new.iterrows():
                yield r
        else:
            time.sleep(0.2)


def append_score_row(out_path: Path, row_dict: dict, header_if_new: bool = True):
    df = pd.DataFrame([row_dict])
    write_header = header_if_new and ((not out_path.exists()) or (out_path.stat().st_size == 0))
    df.to_csv(out_path, mode="a", header=write_header, index=False)


# ---------- Archive and evaluation ----------
def _try_run(cmd):
    """Run a subprocess; return True if it exits 0, else False (never raises)."""
    try:
        res = subprocess.run(cmd, check=False)
        return res.returncode == 0
    except Exception:
        return False

def archive_and_eval():
    """
    Snapshot outputs/scores_live.csv into artifacts/run_<id>/,
    then run tools/metrics.py and tools/eval_and_plot.py on that snapshot.
    Compatible with your current CLIs:
      - metrics.py: [--infile FILE]  (no --outfile)
      - eval_and_plot.py: --infile FILE --outdir_outputs DIR --outdir_plots DIR
    """
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = ARTIFACTS_DIR / f"run_{run_id}"
    plots_dir = run_dir / "plots"
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not OUTPUT_CSV.exists():
        print("[ARCHIVE] No scores_live.csv to archive.")
        return

    # 1) copy snapshot
    snap_scores = run_dir / "scores_live.csv"
    shutil.copy2(OUTPUT_CSV, snap_scores)

    # convenient copy with timestamp in outputs/
    live_out_copy = OUTPUTS_DIR / f"live_out_{run_id}.csv"
    shutil.copy2(OUTPUT_CSV, live_out_copy)

    # 2) write manifest
    manifest = {
        "run_id": run_id,
        "snapshot": str(snap_scores),
        "threshold_at_shutdown": _current_threshold,
        "created": datetime.datetime.now().isoformat(),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"[ARCHIVE] Snapshot → {snap_scores}")

    # 3) evaluate on the snapshot
    metrics_py = TOOLS_DIR / "metrics.py"
    eval_py    = TOOLS_DIR / "eval_and_plot.py"

    # metrics: only --infile is supported; capture stdout to metrics.txt
    if metrics_py.exists():
        print("[*] Running metrics.py …")
        res = subprocess.run(
            ["python3", str(metrics_py), "--infile", str(snap_scores)],
            capture_output=True, text=True, check=False
        )
        (run_dir / "metrics.txt").write_text(res.stdout + ("\n" + res.stderr if res.stderr else ""))

    # plots: use the explicit outdir flags your script exposes
    if eval_py.exists():
        print("[*] Running eval_and_plot.py …")
        subprocess.run(
            ["python3", str(eval_py),
             "--infile", str(snap_scores),
             "--outdir_outputs", str(run_dir),
             "--outdir_plots",   str(plots_dir)],
            check=False
        )

    print(f"[DONE] Archived run  → {run_dir}")
    print(f"[DONE] Plots (if any) → {plots_dir}")

def stop_other_processes():
    """Best-effort gentle stop for feeder/dashboard if they were launched by this shell."""
    try:
        os.system("pkill -f 'feeder.py' 2>/dev/null || true")
        os.system("pkill -f 'streamlit run tools/dashboard.py' 2>/dev/null || true")
    except Exception:
        pass


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=str(MODELS_DIR / "autoencoder_cicids17_cleaned.h5"),
                    help="Path to trained Keras autoencoder model (.h5)")
    ap.add_argument("--thr", type=float, default=0.02,
                    help="Initial threshold (overridden by control file if present)")
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="Optional sleep (seconds) between rows to simulate processing")
    args = ap.parse_args()

    print(f"[BOOT] Model  : {args.model}")
    print(f"[BOOT] Input  : {INPUT_CSV}")
    print(f"[BOOT] Output : {OUTPUT_CSV}")
    print(f"[BOOT] Control: {CONTROL_PATH}")
    print(f"[BOOT] Start threshold: {args.thr:.4f}")

    model = load_autoencoder(args.model)

    last_thr_check = 0.0
    idx = 0

    global _current_threshold
    _current_threshold = float(args.thr)

    for row in tail_csv(INPUT_CSV, start_at_end=False):
        # Check the control file every ~0.5s
        now = time.time()
        if now - last_thr_check > 0.5:
            args.thr = load_control_threshold(args.thr)
            _current_threshold = float(args.thr)
            last_thr_check = now

        try:
            feats = features_from_row(row)
            if feats.size == 0:
                continue
            score = reconstruction_error(model, feats)
        except Exception as e:
            print(f"[ERR] Scoring failed: {e}")
            continue

        flag = 1 if score > args.thr else 0
        out = {
            "ts": int(now),
            "index": int(idx),
            "score": float(score),
            "flag": int(flag),
            "threshold": float(args.thr),
        }
        append_score_row(OUTPUT_CSV, out)
        if flag:
            print(f"[ALERT] index={idx} score={score:.6f} > thr={args.thr:.4f}")
        idx += 1

        if args.sleep > 0:
            time.sleep(args.sleep)


# ---------- Register clean shutdown ----------
def _on_exit():
    print("\n[EXIT] watcher stopped by user")
    # optional: stop feeder/UI if you start them from same shell
    stop_other_processes()
    print("[*] Archiving & evaluating run…")
    archive_and_eval()
    print("[✓] Done.\n")


if __name__ == "__main__":
    try:
        atexit.register(_on_exit)
        main()
    except KeyboardInterrupt:
        # allow atexit(_on_exit) to run
        pass

