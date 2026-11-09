"""Spark Structured Streaming: near-real-time demand monitoring and alerting.

Source  : JSON files landing in data/stream/incoming (swap for Kafka in production:
          spark.readStream.format("kafka").option("subscribe", "meter-readings")...)
Process : stream-static join with site metadata, 15-minute tumbling windows with a
          watermark for late data, demand compared with each site's contracted capacity.
Sinks   : console (live view) and Parquet (alert history), with checkpointing for
          exactly-once recovery.

Run:  python src/streaming_monitor.py --duration 70
"""
import argparse
import shutil
import sys
from pathlib import Path

from pyspark.sql import functions as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CHECKPOINT_DIR, INTERVAL_HOURS, READING_SCHEMA,  # noqa: E402
                    STREAM_IN_DIR, STREAM_OUT_DIR, get_spark, sites_df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=int, default=70, help="seconds to run before stopping")
    ap.add_argument("--fresh", action="store_true", help="delete old checkpoints and output")
    args = ap.parse_args()
    if args.fresh:
        shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)
        shutil.rmtree(STREAM_OUT_DIR, ignore_errors=True)
    STREAM_IN_DIR.mkdir(parents=True, exist_ok=True)

    spark = get_spark("EnergyMonitoring-Streaming")
    sites = F.broadcast(sites_df(spark))

    readings = (spark.readStream.schema(READING_SCHEMA)
                .option("maxFilesPerTrigger", 1)
                .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")
                .json(str(STREAM_IN_DIR)))

    demand = (readings
              .filter(F.col("energy_kwh").isNotNull() & (F.col("energy_kwh") >= 0))
              .join(sites, "site_id")  # stream-static join
              .withColumn("power_kw", F.col("energy_kwh") / F.lit(INTERVAL_HOURS))
              .withWatermark("reading_ts", "30 minutes")
              .groupBy(F.window("reading_ts", "15 minutes"), "site_id", "site_name", "region",
                       "contract_demand_kw")
              .agg(F.sum("power_kw").alias("demand_kw"), F.count(F.lit(1)).alias("meters_reporting"))
              .withColumn("load_pct", F.round(100 * F.col("demand_kw") / F.col("contract_demand_kw"), 1))
              .withColumn("status", F.when(F.col("load_pct") > 100, "ALERT")
                                     .when(F.col("load_pct") > 90, "WARN").otherwise("OK")))

    def handle_batch(batch_df, batch_id):
        # A micro-batch holds at most one row per site per window, so take ONE snapshot
        # of it; re-running a stateful batch for each action can give different answers.
        view = batch_df.select(F.col("window.start").alias("interval_utc"), "site_id", "site_name",
                               "region", F.round("demand_kw", 0).alias("demand_kw"),
                               "contract_demand_kw", "load_pct", "status")
        rows = view.collect()
        if not rows:
            return
        snapshot = spark.createDataFrame(rows, view.schema)
        alerts = [r for r in rows if r["status"] != "OK"]
        if alerts:
            alerts_df = spark.createDataFrame(alerts, view.schema)
            alerts_df.write.mode("append").parquet(str(STREAM_OUT_DIR / "alerts"))
            print(f"\n=== micro-batch {batch_id}: {len(alerts)} site(s) over 90% of contracted demand ===")
            alerts_df.orderBy(F.desc("load_pct")).show(truncate=False)
        (snapshot.groupBy("interval_utc", "region")
                 .agg(F.round(F.sum("demand_kw") / 1000, 2).alias("demand_mw"))
                 .write.mode("append").parquet(str(STREAM_OUT_DIR / "region_demand")))

    query = (demand.writeStream.outputMode("update")
             .foreachBatch(handle_batch)
             .option("checkpointLocation", str(CHECKPOINT_DIR / "demand"))
             .trigger(processingTime="2 seconds")
             .start())
    query.awaitTermination(args.duration)
    print(f"\nProcessed {query.lastProgress['batchId'] + 1 if query.lastProgress else 0} micro-batches")
    query.stop()

    alerts_path = STREAM_OUT_DIR / "alerts"
    if alerts_path.exists():
        hist = spark.read.parquet(str(alerts_path))
        print("Alert history by site:")
        hist.groupBy("site_id", "site_name", "status").count().orderBy(F.desc("count")).show(truncate=False)
    spark.stop()


if __name__ == "__main__":
    main()
