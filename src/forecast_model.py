"""Spark MLlib: forecast next-day energy use per site with a Gradient-Boosted Trees model.

Reads the daily aggregates written by batch_pipeline.py, builds lag features with
window functions, trains on the first ten months and tests on the last two.

Sites differ ~40x in size, so the model learns a scale-free target: today's kWh divided
by the site's trailing 7-day average. The prediction is multiplied back to kWh.
"""
import csv
import json
import sys
from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.sql import Window
from pyspark.sql import functions as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CURATED_DIR, RESULTS_DIR, get_spark  # noqa: E402

TEST_START = "2026-09-01"


def build_features(daily):
    # Drop partial days at the edges of the export (local midnight != UTC midnight).
    full = Window.partitionBy("site_id")
    daily = (daily.withColumn("max_readings", F.max("readings").over(full))
                  .filter(F.col("readings") >= 0.95 * F.col("max_readings")))
    w = Window.partitionBy("site_id").orderBy("local_date")
    base = F.col("avg_prev_7")
    return (daily
            .withColumn("lag_1", F.lag("kwh", 1).over(w))
            .withColumn("lag_7", F.lag("kwh", 7).over(w))
            .withColumn("avg_prev_7", F.avg("kwh").over(w.rowsBetween(-7, -1)))
            .withColumn("temp_change", F.col("avg_temp_c") - F.lag("avg_temp_c", 1).over(w))
            .dropna(subset=["lag_7", "temp_change"])
            .withColumn("lag_1_ratio", F.col("lag_1") / base)
            .withColumn("lag_7_ratio", F.col("lag_7") / base)
            .withColumn("target_ratio", F.col("kwh") / base)
            .withColumn("day_of_week", F.dayofweek("local_date"))
            .withColumn("month", F.month("local_date"))
            .withColumn("is_weekend", F.col("day_of_week").isin(1, 7).cast("int")))


def main():
    spark = get_spark("EnergyMonitoring-Forecast")
    daily = spark.read.parquet(str(CURATED_DIR / "site_daily"))
    data = build_features(daily).cache()
    train = data.filter(F.col("local_date") < TEST_START)
    test = data.filter(F.col("local_date") >= TEST_START)

    numeric = ["lag_1_ratio", "lag_7_ratio", "avg_temp_c", "temp_change", "day_of_week", "is_weekend"]
    pipeline = Pipeline(stages=[
        StringIndexer(inputCols=["site_id", "site_type"], outputCols=["site_idx", "type_idx"]),
        OneHotEncoder(inputCols=["site_idx", "type_idx"], outputCols=["site_vec", "type_vec"]),
        VectorAssembler(inputCols=numeric + ["site_vec", "type_vec"], outputCol="features"),
        GBTRegressor(featuresCol="features", labelCol="target_ratio", predictionCol="pred_ratio",
                     maxIter=80, maxDepth=5, seed=7),
    ])
    model = pipeline.fit(train)
    pred = (model.transform(test)
                 .withColumn("prediction", F.col("pred_ratio") * F.col("avg_prev_7")).cache())

    ev = RegressionEvaluator(labelCol="kwh", predictionCol="prediction")
    ape = F.abs(F.col("prediction") - F.col("kwh")) / F.col("kwh")
    naive_ape = F.abs(F.col("lag_7") - F.col("kwh")) / F.col("kwh")
    mape, naive_mape = pred.agg(F.avg(ape), F.avg(naive_ape)).first()

    attrs = pred.schema["features"].metadata["ml_attr"]["attrs"]
    names = {a["idx"]: a["name"] for group in attrs.values() for a in group}
    importances = model.stages[-1].featureImportances.toArray()
    top = sorted(((names[i], float(v)) for i, v in enumerate(importances)), key=lambda x: -x[1])[:8]

    metrics = {
        "model": "GBTRegressor (80 trees, depth 5)",
        "train_rows": train.count(), "test_rows": test.count(),
        "test_period": f"{TEST_START} onward",
        "rmse_kwh": round(ev.evaluate(pred, {ev.metricName: "rmse"}), 1),
        "r2": round(ev.evaluate(pred, {ev.metricName: "r2"}), 4),
        "mape_pct": round(100 * mape, 2),
        "naive_same_day_last_week_mape_pct": round(100 * naive_mape, 2),
        "top_features": [{"feature": n, "importance": round(v, 4)} for n, v in top],
    }
    (RESULTS_DIR / "forecast_metrics.json").write_text(json.dumps(metrics, indent=2))
    with open(RESULTS_DIR / "forecast_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["site_id", "local_date", "actual_kwh", "predicted_kwh"])
        for r in pred.select("site_id", "local_date", "kwh", "prediction").orderBy("site_id", "local_date").collect():
            w.writerow([r[0], r[1], round(r[2], 1), round(r[3], 1)])
    print(json.dumps(metrics, indent=2))
    spark.stop()


if __name__ == "__main__":
    main()
