#!/usr/bin/env python3
# HIDS Live Dashboard (compact table sans index/row_id + derived columns)
import os, json, time
from pathlib import Path
import pandas as pd
import streamlit as st
import altair as alt

# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "outputs" / "scores_live.csv"
CONTROL_DIR = PROJECT_ROOT / "control"
CONTROL_PATH = CONTROL_DIR / "threshold.json"
CONTROL_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Page ----------
st.set_page_config(layout="wide", page_title="HIDS – Live Anomaly Monitor")
st.title("HIDS – Live Anomaly Monitor")

# ---------- Session defaults ----------
for k, v in {
    "thr": 0.02,
    "prev_rows": 0,
    "prev_alerts": 0,
    "last_thr_written": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------- Helpers ----------
def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=["ts", "index", "score", "flag", "threshold"])
    try:
        return pd.read_csv(path)
    except Exception:
        time.sleep(0.1)
        return pd.read_csv(path)

def write_threshold_atomic(thr: float):
    payload = {"thr": float(thr), "ts": time.time()}
    tmp = CONTROL_PATH.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, CONTROL_PATH)

def read_existing_threshold() -> float:
    try:
        data = json.loads(CONTROL_PATH.read_text())
        return float(data.get("thr", st.session_state.thr))
    except Exception:
        return st.session_state.thr

# Initialize slider from control file once
if st.session_state.last_thr_written is None and CONTROL_PATH.exists():
    st.session_state.thr = read_existing_threshold()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Controls")
    st.session_state.thr = st.slider(
        "Anomaly threshold", 0.0, 1.0, float(st.session_state.thr), 0.001
    )
    refresh_ms = st.slider("Refresh (ms)", 200, 5000, 1200, 50)

    st.divider()
    st.subheader("Compact table")
    compact_rows = st.slider("Rows (tail, compact only)", 10, 5000, 20, 1)
    compact_cols = st.slider("Columns (compact only)", 5, 30, 10, 1)

    st.divider()
    pause = st.checkbox("Pause refresh", value=False)

    st.caption(f"Watcher must write to: {CSV_PATH}")
    try:
        if CONTROL_PATH.exists():
            data = json.loads(CONTROL_PATH.read_text())
            st.caption(f"Control file: thr={float(data.get('thr', 0.0)):.3f}")
    except Exception:
        pass

# Write control if changed
if st.session_state.last_thr_written != st.session_state.thr:
    write_threshold_atomic(st.session_state.thr)
    st.session_state.last_thr_written = st.session_state.thr

# ---------- Rendering ----------
frame = st.empty()

def threshold_rule(y_value: float):
    return alt.Chart(pd.DataFrame({"y": [y_value]})).mark_rule(strokeDash=[6,6]).encode(y="y:Q")

def pick_compact_columns(df: pd.DataFrame, k: int) -> list:
    # Always hide raw position helpers
    hidden = {"index", "row_id"}
    # Prioritize the most useful columns
    priority = ["time", "score", "margin", "flag", "above_thr", "threshold", "roll_mean_200", "roll_std_200", "ts"]
    cols = [c for c in priority if c in df.columns and c not in hidden]
    # Fill with remaining (stable order), skipping hidden
    for c in df.columns:
        if c not in cols and c not in hidden:
            cols.append(c)
        if len(cols) >= k:
            break
    return cols[:k]

def enrich_for_display(df: pd.DataFrame, thr: float) -> pd.DataFrame:
    if df.empty:
        return df
    # add a display-friendly row id for charts (not shown in tables)
    if "index" in df.columns:
        df["row_id"] = df["index"]
    else:
        df["row_id"] = range(len(df))
    # time, live above-threshold, margin, rolling stats
    try:
        df["time"] = pd.to_datetime(df["ts"], unit="s")
    except Exception:
        df["time"] = pd.NaT
    df["above_thr"] = (df["score"] > thr).astype(int)
    df["margin"] = (df["score"] - thr).astype(float)
    df["roll_mean_200"] = df["score"].rolling(window=200, min_periods=20).mean()
    df["roll_std_200"] = df["score"].rolling(window=200, min_periods=20).std()
    return df

def render_once():
    df = safe_read_csv(CSV_PATH)
    ui_thr = float(st.session_state.thr)
    rows = len(df)

    if rows:
        df = enrich_for_display(df, ui_thr)

    # alert count (prefer model flag if present)
    alerts = int(df["flag"].sum()) if ("flag" in df.columns) else int(df.get("above_thr", pd.Series(dtype=int)).sum())

    # Chart data (last 500 pts)
    chart_df = pd.DataFrame(columns=["row_id", "score"])
    if rows:
        chart_df = df[["row_id", "score"]].tail(500).copy()

    with frame.container():
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows seen", f"{rows:,}", delta=rows - st.session_state.prev_rows)
        c2.metric("Alerts raised", f"{alerts:,}", delta=alerts - st.session_state.prev_alerts)
        c3.metric("Current threshold", f"{ui_thr:.3f}")

        # Chart + dotted threshold line
        if not chart_df.empty:
            base = alt.Chart(chart_df).mark_line().encode(
                x=alt.X("row_id:Q", title="row"),
                y=alt.Y("score:Q", title="reconstruction error")
            )
            st.altair_chart((base + threshold_rule(ui_thr)).properties(height=260), use_container_width=True)
        else:
            st.info("Waiting for scores… Start feeder & watcher.")

        if rows:
            # Hide the raw helper columns in both views
            hide = ["index", "row_id"]

            # Compact table: limited rows & columns, no index/row_id
            comp_cols = pick_compact_columns(df, compact_cols)
            compact_df = df.drop(columns=hide, errors="ignore").tail(compact_rows)[comp_cols]
            st.caption(f"Compact view (showing {len(comp_cols)} columns): {', '.join(comp_cols)}")
            st.dataframe(compact_df, use_container_width=True, height=260)

            # Expanded table: ALL rows & ALL columns EXCEPT index/row_id
            with st.expander("Expand: full table (ALL rows & columns, except helpers)"):
                full_df = df.drop(columns=hide, errors="ignore")
                st.dataframe(full_df, use_container_width=True, height=520)

    st.session_state.prev_rows = rows
    st.session_state.prev_alerts = alerts

# First render
render_once()
while not pause:
    time.sleep(refresh_ms / 1000.0)
    frame.empty()
    render_once()
