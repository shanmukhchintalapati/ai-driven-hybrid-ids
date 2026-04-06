#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

def slug(x: float) -> str:
    return str(x).replace(".", "p")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="outputs/scores_live.csv",
                    help="Input CSV with columns: ts|timestamp (or none), index (optional), score")
    ap.add_argument("--threshold", type=float, default=0.02,
                    help="Flag rows where score > threshold")
    ap.add_argument("--outdir_outputs", default="outputs",
                    help="Directory to save recomputed CSV")
    ap.add_argument("--outdir_plots", default="plots",
                    help="Directory to save time-series plot")
    args = ap.parse_args()

    os.makedirs(args.outdir_outputs, exist_ok=True)
    os.makedirs(args.outdir_plots, exist_ok=True)

    # --- Load ---
    df = pd.read_csv(args.infile)

    # pick a time axis
    if "ts" in df.columns:
        ts_col = "ts"
    elif "timestamp" in df.columns:
        ts_col = "timestamp"
    else:
        df["row"] = range(len(df))
        ts_col = "row"

    if "score" not in df.columns:
        raise ValueError("Expected a 'score' column in the input CSV")

    thr = args.threshold

    # --- Recompute flags ---
    df["flag"] = (df["score"] > thr).astype(int)
    alerts = int(df["flag"].sum())

    # --- Build output to match metrics.py schema: ts,index,err,flag,threshold ---
    if "index" not in df.columns:
        df["index"] = range(len(df))

    out = pd.DataFrame({
        "ts": df[ts_col],
        "index": df["index"],
        "err": df["score"],
        "flag": df["flag"].astype(int),
        "threshold": [thr] * len(df)
    })

    tag = slug(thr)
    out_csv = os.path.join(args.outdir_outputs, f"live_out_thr{tag}.csv")
    out.to_csv(out_csv, index=False, na_rep="nan")

    # --- Plot error series + threshold ---
    plt.figure()
    plt.plot(df[ts_col], df["score"], label="reconstruction error")
    plt.axhline(y=thr, linestyle="--", label=f"threshold={thr}")
    plt.xlabel(ts_col)
    plt.ylabel("error")
    plt.title(f"Error series (thr={thr}) | alerts={alerts}")
    plt.legend()
    out_png = os.path.join(args.outdir_plots, f"error_timeseries_thr{tag}.png")
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

    print(f"Loaded {len(df)} rows from {args.infile}")
    print(f"Recomputed flags with threshold={thr}")
    print(f"Alerts raised: {alerts}")
    print(f"Saved CSV -> {out_csv}")
    print(f"Saved plot -> {out_png}")

if __name__ == "__main__":
    main()


