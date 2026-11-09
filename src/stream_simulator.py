"""Simulate live smart meters: every few seconds drop one JSON file holding the latest
15-minute reading from every meter at every site (like a Kafka topic partition dump).

Run alongside streaming_monitor.py:
    python src/stream_simulator.py --batches 24 --interval 2
"""
import argparse
import json
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import STREAM_IN_DIR, load_sites  # noqa: E402
from generate_data import CATEGORY_MIX  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=24, help="number of 15-min intervals to emit")
    ap.add_argument("--interval", type=float, default=2.0, help="seconds between files")
    ap.add_argument("--start", default="2026-11-09 08:00:00")
    ap.add_argument("--spike-prob", type=float, default=0.02)
    args = ap.parse_args()

    STREAM_IN_DIR.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(11)
    clock = datetime.fromisoformat(args.start)
    for b in range(args.batches):
        records = []
        for site in load_sites():
            for category, share in CATEGORY_MIX[site["site_type"]].items():
                kw = site["peak_load_kw"] * share * rnd.uniform(0.75, 0.95)
                if rnd.random() < args.spike_prob:
                    kw *= rnd.uniform(1.6, 2.4)  # a chiller trips, a furnace surges, ...
                records.append({
                    "reading_ts": clock.strftime("%Y-%m-%d %H:%M:%S"),
                    "site_id": site["site_id"],
                    "meter_id": f"{site['site_id']}-{category.upper()}",
                    "category": category,
                    "energy_kwh": round(kw * 0.25, 3),
                    "voltage_v": float(site["nominal_voltage"]),
                    "ambient_temp_c": round(site["temp_mean_c"] + rnd.uniform(-3, 3), 1),
                })
        tmp = STREAM_IN_DIR / f".batch_{b:04d}.tmp"
        tmp.write_text("\n".join(json.dumps(r) for r in records))
        tmp.rename(STREAM_IN_DIR / f"batch_{b:04d}.json")  # atomic publish
        print(f"[simulator] {clock:%H:%M} UTC -> {len(records)} readings")
        clock += timedelta(minutes=15)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
