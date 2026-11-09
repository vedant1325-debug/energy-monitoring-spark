"""Generate one year of synthetic 15-minute smart-meter data for every site.

The data mimics what a building-management system exports: one CSV per site per month,
timestamps in UTC, and the usual real-world mess (nulls, negative values, duplicate rows).
Three operational problems are planted so the Spark pipeline has something real to find:

  * Dublin Office  - BMS schedule override from 2026-03-01: HVAC runs 24/7.
  * Frankfurt DC   - cooling efficiency degrades through the year (PUE creeps up).
  * Pune Plant     - production meter reports zeros for 48 hours (meter fault).
  * Everywhere     - rare demand spikes (3-6x normal) on random meters.

Usage:  python src/generate_data.py [--days 365] [--start 2025-11-01] [--seed 42]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW_DIR, load_sites  # noqa: E402

CATEGORY_MIX = {
    "Office":        {"HVAC": 0.45, "Lighting": 0.20, "IT": 0.25, "Other": 0.10},
    "Data Center":   {"IT": 0.62, "HVAC": 0.30, "Lighting": 0.03, "Other": 0.05},
    "Manufacturing": {"Production": 0.60, "HVAC": 0.20, "Lighting": 0.10, "Other": 0.10},
    "R&D Lab":       {"HVAC": 0.40, "Production": 0.25, "IT": 0.20, "Lighting": 0.15},
    "Warehouse":     {"HVAC": 0.35, "Lighting": 0.35, "Production": 0.20, "Other": 0.10},
}

BMS_FAULT_SITE, BMS_FAULT_START = "S09", pd.Timestamp("2026-03-01", tz="UTC")
DEGRADE_SITE = "S08"
METER_FAULT = ("S03-PRODUCTION", pd.Timestamp("2026-07-14 06:00", tz="UTC"),
               pd.Timestamp("2026-07-16 06:00", tz="UTC"))


def occupancy(site_type, hour, dow):
    """0..1 activity level by local hour and weekday (0=Mon .. 6=Sun)."""
    wk, sat = dow < 5, dow == 5
    if site_type == "Data Center":
        return 0.97 + 0.03 * np.sin(2 * np.pi * (hour - 9) / 24)
    if site_type == "Manufacturing":
        shift = (hour >= 6) & (hour < 22)
        return np.where(wk & shift, 1.0, np.where(sat & (hour >= 6) & (hour < 14), 0.9, 0.25))
    if site_type == "Warehouse":
        shift = (hour >= 6) & (hour < 20)
        return np.where(wk & shift, 1.0, np.where(sat & (hour >= 8) & (hour < 14), 0.7, 0.25))
    open_h, close_h = (8, 20) if site_type == "R&D Lab" else (8, 19)
    floor = 0.35 if site_type == "R&D Lab" else 0.2
    day = (hour >= open_h) & (hour < close_h)
    shoulder = ((hour >= open_h - 1) & (hour < open_h)) | ((hour >= close_h) & (hour < close_h + 2))
    occ = np.where(day, 1.0, np.where(shoulder, 0.55, floor))
    return np.where(wk, occ, floor * 0.9)


def site_frame(site, ts_utc, rng):
    local = ts_utc.tz_convert(site["timezone"])
    hour = (local.hour + local.minute / 60).to_numpy(dtype=float)
    dow = local.day_of_week.to_numpy()
    doy = local.day_of_year.to_numpy()
    n = len(ts_utc)

    seasonal = site["hemisphere"] * site["temp_amplitude_c"] * np.cos(2 * np.pi * (doy - 200) / 365)
    daily = 3.0 * np.cos(2 * np.pi * (hour - 15) / 24)
    temp = site["temp_mean_c"] + seasonal + daily + rng.normal(0, 1.2, n)

    stype = site["site_type"]
    occ = occupancy(stype, hour, dow)
    progress = np.linspace(0, 1, n)
    frames = []
    for category, share in CATEGORY_MIX[stype].items():
        meter_id = f"{site['site_id']}-{category.upper()}"
        occ_c = occ
        if category == "HVAC" and site["site_id"] == BMS_FAULT_SITE:
            occ_c = np.where(ts_utc >= BMS_FAULT_START, 1.0, occ)  # schedule override
        if stype == "Data Center":
            factor = {
                "IT": 0.92 + 0.05 * np.sin(2 * np.pi * (hour - 9) / 24),
                "HVAC": 0.70 + 0.02 * np.clip(temp - 10, 0, None),
                "Lighting": np.ones(n),
                "Other": 0.9 * np.ones(n),
            }[category]
            if category == "HVAC" and site["site_id"] == DEGRADE_SITE:
                factor = factor * (1 + 0.40 * progress)
        elif category == "HVAC":
            factor = (0.30 + 0.70 * occ_c) * np.clip(0.55 + 0.045 * np.abs(temp - 21), 0.55, 1.6)
        elif category == "Lighting":
            factor = 0.08 + 0.92 * occ_c
        elif category == "IT":
            factor = 0.45 + 0.55 * occ_c
        elif category == "Production":
            factor = np.clip(occ_c, 0.1, None) ** 1.2
        else:
            factor = 0.40 + 0.60 * occ_c

        kw = site["peak_load_kw"] * share * factor * rng.lognormal(0, 0.05, n)
        spikes = rng.random(n) < 0.0005
        kw = np.where(spikes, kw * rng.uniform(3, 6, n), kw)
        if meter_id == METER_FAULT[0]:
            kw = np.where((ts_utc >= METER_FAULT[1]) & (ts_utc < METER_FAULT[2]), 0.0, kw)

        frames.append(pd.DataFrame({
            "reading_ts": ts_utc.tz_localize(None),
            "site_id": site["site_id"],
            "meter_id": meter_id,
            "category": category,
            "energy_kwh": np.round(kw * 0.25, 4),
            "voltage_v": np.round(site["nominal_voltage"] * rng.normal(1, 0.01, n), 1),
            "ambient_temp_c": np.round(temp, 1),
        }))
    df = pd.concat(frames, ignore_index=True)

    # Real exports are messy: nulls, sign errors and duplicated rows.
    m = len(df)
    df.loc[rng.random(m) < 0.002, "energy_kwh"] = np.nan
    neg = rng.random(m) < 0.0005
    df.loc[neg, "energy_kwh"] = -df.loc[neg, "energy_kwh"]
    dupes = df.sample(frac=0.003, random_state=int(rng.integers(1e9)))
    return pd.concat([df, dupes], ignore_index=True).sort_values(["reading_ts", "meter_id"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--start", default="2025-11-01")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    ts = pd.date_range(args.start, periods=args.days * 96, freq="15min", tz="UTC")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for old in RAW_DIR.glob("*.csv"):
        old.unlink()

    t0, total, files = time.time(), 0, 0
    for site in load_sites():
        df = site_frame(site, ts, rng)
        for month, part in df.groupby(df["reading_ts"].dt.strftime("%Y%m")):
            part.to_csv(RAW_DIR / f"{site['site_id']}_{month}.csv", index=False,
                        date_format="%Y-%m-%d %H:%M:%S")
            files += 1
        total += len(df)
        print(f"  {site['site_id']} {site['site_name']:<24} {len(df):>9,} rows")
    print(f"Generated {total:,} readings in {files} CSV files -> {RAW_DIR} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
