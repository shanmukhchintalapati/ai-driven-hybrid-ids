#!/usr/bin/env python3
"""
Aggressive high-variability feeder for HIDS demos.
Produces wide-ranging traffic:
- Baseline Gaussian noise
- Random anomalies (sparse/many/band/alternating)
- Frequent bursts
- Ramps (sustained, gradual attacks)
- Noisy windows (variance spikes)
- Concept drift (slow mean shift)
- Extreme single-feature spikes
- 'Attack windows' (heavy, long-lived offsets across many features)

Defaults are intentionally strong so you see alerts at moderate thresholds (e.g., 0.02–0.06).
"""

import os, csv, time, argparse
from pathlib import Path
import numpy as np

# ----- Resolve project root -----
_THIS = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parents[1] if _THIS.parent.name in ("scripts","tools") else _THIS.parent
INPUTS_DIR = PROJECT_ROOT / "inputs"
CSV_PATH   = INPUTS_DIR / "live_features.csv"
INPUTS_DIR.mkdir(parents=True, exist_ok=True)

def parse_args():
    p = argparse.ArgumentParser("Aggressive HIDS feeder")
    # stream
    p.add_argument("--rows", type=int, default=-1, help="Total rows (-1 = infinite)")
    p.add_argument("--rps", type=float, default=8.0, help="Rows/sec (approx)")
    p.add_argument("--sleep", type=float, default=None, help="Seconds between rows (overrides --rps)")
    # features
    p.add_argument("--dim", type=int, default=78, help="Feature dimension")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    # baseline
    p.add_argument("--mu", type=float, default=0.0)
    p.add_argument("--sigma", type=float, default=0.003)
    # random anomalies
    p.add_argument("--anomaly-pct", type=float, default=0.40, help="Per-row anomaly probability")
    p.add_argument("--anomaly-low", type=float, default=0.05)
    p.add_argument("--anomaly-high", type=float, default=0.25)
    # bursts (consecutive anomalies)
    p.add_argument("--burst-every", type=int, default=80)
    p.add_argument("--burst-len", type=int, default=30)
    p.add_argument("--burst-low", type=float, default=0.08)
    p.add_argument("--burst-high", type=float, default=0.30)
    # ramps (sustained)
    p.add_argument("--ramp-every", type=int, default=280)
    p.add_argument("--ramp-len", type=int, default=70)
    p.add_argument("--ramp-peak", type=float, default=0.22)
    # noisy windows (variance spikes)
    p.add_argument("--noisy-every", type=int, default=220)
    p.add_argument("--noisy-len", type=int, default=45)
    p.add_argument("--sigma-mult", type=float, default=5.0)
    # concept drift
    p.add_argument("--drift-every", type=int, default=900)
    p.add_argument("--drift-step", type=float, default=0.0012)
    # attack windows (heavy, long-lived offsets across many features)
    p.add_argument("--attack-every", type=int, default=700)
    p.add_argument("--attack-len", type=int, default=120)
    p.add_argument("--attack-mag", type=float, default=0.10, help="Base offset for attack windows")
    p.add_argument("--attack-mult", type=float, default=2.0, help="Random scale for each row in window")
    # header
    p.add_argument("--write-header", action="store_true", help="Write header if file empty")
    return p.parse_args()

def write_header_if_needed(fh, dim:int):
    if fh.tell() == 0:
        w = csv.writer(fh)
        w.writerow([f"f{i}" for i in range(dim)])
        fh.flush(); os.fsync(fh.fileno())

def gen_row(rng, dim, mu, sigma, anomaly=False, mag=0.0, pattern=None):
    x = rng.normal(loc=mu, scale=sigma, size=dim)
    if not anomaly:
        return x
    if pattern is None:
        pattern = rng.choice(["many","sparse","band","alternating"])
    if pattern == "many":
        k = rng.integers(int(0.4*dim), dim)  # 40–100% of features
        idx = rng.choice(dim, size=k, replace=False)
        x[idx] += rng.choice([-1.0,1.0]) * mag * rng.uniform(0.9,1.3)
    elif pattern == "sparse":
        k = rng.integers(max(2,int(0.08*dim)), max(3,int(0.20*dim)))
        idx = rng.choice(dim, size=k, replace=False)
        x[idx] += rng.choice([-1.0,1.0]) * mag * rng.uniform(0.8,1.4)
    elif pattern == "band":
        width = rng.integers(max(3,int(0.15*dim)), max(5,int(0.45*dim)))
        start = rng.integers(0, dim-width+1)
        idx = np.arange(start, start+width)
        x[idx] += rng.choice([-1.0,1.0]) * mag * rng.uniform(0.9,1.3)
    else:  # alternating
        for i in range(dim):
            if (i % 2)==0 and rng.random()<0.7:
                x[i] +=  mag * rng.uniform(0.6,1.2)
            elif (i % 2)==1 and rng.random()<0.4:
                x[i] += -mag * rng.uniform(0.4,1.0)
    return x

def main():
    a = parse_args()
    rng = np.random.default_rng(a.seed)
    sleep_s = a.sleep if a.sleep is not None else (1.0/max(a.rps,1e-6))

    in_burst = 0
    in_ramp  = 0; ramp_step = 0
    in_noisy = 0
    in_attack = 0
    base_mu = a.mu

    print(f"[BOOT] Feeder -> {CSV_PATH}")
    print(f"[BOOT] rows={'∞' if a.rows<0 else a.rows}, rps~{(1.0/sleep_s):.2f}, dim={a.dim}")
    print(f"[BOOT] anomaly_pct={a.anomaly_pct}, burst_every={a.burst_every} len={a.burst_len}")
    print(f"[BOOT] ramp_every={a.ramp_every} len={a.ramp_len} peak={a.ramp_peak}")
    print(f"[BOOT] noisy_every={a.noisy_every} len={a.noisy_len} xσ={a.sigma_mult}")
    print(f"[BOOT] drift_every={a.drift_every} step={a.drift_step}")
    print(f"[BOOT] attack_every={a.attack_every} len={a.attack_len} mag={a.attack_mag} mult={a.attack_mult}")

    with open(CSV_PATH, "a", newline="") as fh:
        if a.write_header: write_header_if_needed(fh, a.dim)
        w = csv.writer(fh)

        i = 0
        try:
            while a.rows<0 or i<a.rows:
                # scheduled regimes
                if a.drift_every>0 and i>0 and (i % a.drift_every)==0:
                    base_mu += a.drift_step
                    print(f"[DRIFT] row={i} new_base_mu={base_mu:.6f}")

                if a.burst_every>0 and (i % a.burst_every)==0:  # start burst
                    in_burst = a.burst_len

                if a.ramp_every>0 and (i % a.ramp_every)==0:    # start ramp
                    in_ramp = a.ramp_len; ramp_step = 0

                if a.noisy_every>0 and (i % a.noisy_every)==0:  # start noisy window
                    in_noisy = a.noisy_len

                if a.attack_every>0 and (i % a.attack_every)==0: # start attack window
                    in_attack = a.attack_len
                    print(f"[ATTACK] start window at row={i} len={a.attack_len}")

                # effective sigma
                cur_sigma = a.sigma * (a.sigma_mult if in_noisy>0 else 1.0)

                # choose anomaly mode
                is_anom, mag, pattern = False, 0.0, None

                if in_attack>0:
                    # heavy sustained offset across many features
                    is_anom = True
                    mag = a.attack_mag * rng.uniform(1.0, a.attack_mult)
                    pattern = "many"
                    in_attack -= 1

                elif in_burst>0:
                    is_anom = True
                    mag = rng.uniform(a.burst_low, a.burst_high)
                    pattern = rng.choice(["many","band","alternating"])
                    in_burst -= 1

                elif in_ramp>0:
                    in_ramp -= 1; ramp_step += 1
                    half = max(1, a.ramp_len//2)
                    frac = (ramp_step/half) if ramp_step<=half else max(0.0,(a.ramp_len-ramp_step)/half)
                    mag = frac * a.ramp_peak
                    if mag>0.002:
                        is_anom = True; pattern = "many"

                elif rng.random() < a.anomaly_pct:
                    is_anom = True
                    mag = rng.uniform(a.anomaly_low, a.anomaly_high)
                    pattern = rng.choice(["sparse","many","band","alternating"])

                # occasional extreme spike (guaranteed large single-feature)
                if (i % 180)==0 and rng.random()<0.8:
                    vec = rng.normal(loc=base_mu, scale=cur_sigma, size=a.dim)
                    j = int(rng.integers(0, a.dim))
                    vec[j] += rng.choice([-1.0,1.0]) * rng.uniform(0.30, 0.80)
                    w.writerow([f"{v:.6f}" for v in vec.tolist()])
                    fh.flush(); os.fsync(fh.fileno())
                    if (i % 360)==0:
                        print(f"[SPIKE] extreme row={i} idx={j}")
                    i += 1
                    time.sleep(sleep_s)
                    continue

                # normal/anomalous row
                row = gen_row(rng, a.dim, base_mu, cur_sigma, anomaly=is_anom, mag=mag, pattern=pattern)
                w.writerow([f"{v:.6f}" for v in row.tolist()])
                fh.flush(); os.fsync(fh.fileno())

                # status log
                if (i % 200)==0 and i>0:
                    print(f"[FEED] row={i} (anom={int(is_anom)}, mag={mag:.4f}, attack={int(in_attack>0)}, noisy={int(in_noisy>0)})")

                # decrement timers
                if in_noisy>0: in_noisy -= 1

                i += 1
                time.sleep(sleep_s)

        except KeyboardInterrupt:
            print("\n[EXIT] feeder stopped by user")

if __name__ == "__main__":
    main()

