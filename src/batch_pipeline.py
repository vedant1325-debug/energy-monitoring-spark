"""Batch ETL + analytics: raw CSV meter dumps -> curated Parquet lake -> KPI reports.

Run:  python src/batch_pipeline.py
On a cluster:  SPARK_MASTER=spark://<host>:7077 spark-submit src/batch_pipeline.py
"""
import csv
import datetime as dt
import json
import sys
import time
from pathlib import Path

from pyspark.sql import functions as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CURATED_DIR, RAW_DIR, READING_SCHEMA, RESULTS_DIR,  # noqa: E402
                    get_spark, sites_df)
import transforms as T  # noqa: E402


def write_report(df, name):
    """Reports are small, so collect them to the driver and write one tidy CSV."""
    rows = df.collect()
    path = RESULTS_DIR / f"{name}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(df.columns)
        for r in rows:
            w.writerow([round(v, 4) if isinstance(v, float) else
                        v.isoformat(sep=" ") if isinstance(v, dt.datetime) else v for v in r])
    print(f"  wrote {path.relative_to(RESULTS_DIR.parent)} ({len(rows)} rows)")
    return rows


def main():
    spark = get_spark("EnergyMonitoring-BatchETL")
    RESULTS_DIR.mkdir(exist_ok=True)
    timings, t = {}, time.time()

    def lap(name):
        nonlocal t
        timings[name] = round(time.time() - t, 2)
        t = time.time()

    # 1. Ingest --------------------------------------------------------------
    raw = (spark.read.option("header", True)
           .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")
           .schema(READING_SCHEMA).csv(str(RAW_DIR)))
    sites = sites_df(spark)
    print(f"Raw input: {len(list(RAW_DIR.glob('*.csv')))} files, "
          f"{raw.rdd.getNumPartitions()} input partitions")

    # 2. Data quality --------------------------------------------------------
    dq = T.quality_report(raw, sites)
    lap("ingest_and_quality")

    # 3. Clean + enrich, persist curated layer partitioned for pruning --------
    enriched = T.enrich(T.clean_readings(raw), sites).cache()
    clean_rows = enriched.count()
    dq["clean_rows"] = clean_rows
    dq["rows_removed"] = dq["rows"] - clean_rows
    (enriched.drop("local_ts")
             .write.mode("overwrite").partitionBy("region", "year_month")
             .parquet(str(CURATED_DIR / "readings")))
    lap("clean_enrich_write_parquet")

    # 4. Analytics -----------------------------------------------------------
    print("Writing reports:")
    sites_rows = write_report(T.site_summary(enriched), "site_summary")
    write_report(T.region_monthly(enriched), "region_monthly")
    write_report(T.category_breakdown(enriched), "category_breakdown")
    write_report(T.hourly_profile(enriched), "hourly_profile")
    write_report(T.after_hours_monthly(enriched), "after_hours_monthly")
    write_report(T.datacenter_pue(enriched), "datacenter_pue_monthly")
    daily = T.site_daily(enriched)
    daily.write.mode("overwrite").parquet(str(CURATED_DIR / "site_daily"))
    lap("analytics")

    # 5. Anomaly detection ---------------------------------------------------
    spikes = write_report(T.detect_spikes(enriched).orderBy(F.desc("z_score")), "anomalies_spikes")
    faults = write_report(T.detect_meter_faults(enriched), "anomalies_meter_faults")
    lap("anomaly_detection")

    totals = enriched.agg(F.sum("energy_kwh").alias("kwh"), F.sum("cost_usd").alias("cost"),
                          F.sum("co2_kg").alias("co2"), F.countDistinct("meter_id").alias("meters"),
                          F.min("local_date").alias("start"), F.max("local_date").alias("end")).first()
    summary = {
        "period": f"{totals['start']} to {totals['end']}",
        "sites": len(sites_rows),
        "meters": totals["meters"],
        "regions": sorted({r["region"] for r in sites_rows}),
        "total_mwh": round(totals["kwh"] / 1000, 1),
        "total_cost_usd": round(totals["cost"], 0),
        "total_co2_tonnes": round(totals["co2"] / 1000, 1),
        "spike_anomalies": len(spikes),
        "spike_excess_cost_usd": round(sum(r["excess_cost_usd"] for r in spikes), 0),
        "meter_faults": len(faults),
        "data_quality": dq,
        "spark": {"version": spark.version, "master": spark.sparkContext.master,
                  "default_parallelism": spark.sparkContext.defaultParallelism,
                  "shuffle_partitions": int(spark.conf.get("spark.sql.shuffle.partitions"))},
        "timings_sec": timings,
    }
    timings["total"] = round(sum(timings.values()), 2)
    (RESULTS_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    spark.stop()


if __name__ == "__main__":
    main()
