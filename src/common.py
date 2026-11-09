"""Shared settings: paths, schemas, site metadata and the SparkSession factory."""
import json
import os
import time
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.types import (DoubleType, IntegerType, StringType, StructField,
                               StructType, TimestampType)

# Keep the Python driver in UTC too, so collected timestamps are not shifted to local time.
os.environ["TZ"] = "UTC"
time.tzset()

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("EM_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw" / "readings"
CURATED_DIR = DATA_DIR / "curated"
STREAM_IN_DIR = DATA_DIR / "stream" / "incoming"
STREAM_OUT_DIR = DATA_DIR / "stream" / "output"
CHECKPOINT_DIR = DATA_DIR / "stream" / "checkpoints"
RESULTS_DIR = ROOT / "results"
SITES_FILE = ROOT / "config" / "sites.json"

# One row = one smart-meter reading for a 15-minute interval (timestamps are UTC).
READING_SCHEMA = StructType([
    StructField("reading_ts", TimestampType(), True),
    StructField("site_id", StringType(), True),
    StructField("meter_id", StringType(), True),
    StructField("category", StringType(), True),
    StructField("energy_kwh", DoubleType(), True),
    StructField("voltage_v", DoubleType(), True),
    StructField("ambient_temp_c", DoubleType(), True),
])

SITE_SCHEMA = StructType([
    StructField("site_id", StringType()),
    StructField("site_name", StringType()),
    StructField("city", StringType()),
    StructField("country", StringType()),
    StructField("region", StringType()),
    StructField("site_type", StringType()),
    StructField("timezone", StringType()),
    StructField("floor_area_sqm", IntegerType()),
    StructField("peak_load_kw", IntegerType()),
    StructField("contract_demand_kw", IntegerType()),
    StructField("tariff_usd_per_kwh", DoubleType()),
    StructField("grid_kgco2_per_kwh", DoubleType()),
    StructField("open_hour", IntegerType()),
    StructField("close_hour", IntegerType()),
])

INTERVAL_HOURS = 0.25  # 15-minute meter interval


def load_sites():
    with open(SITES_FILE) as f:
        return json.load(f)


def sites_df(spark):
    rows = [{f.name: s[f.name] for f in SITE_SCHEMA.fields} for s in load_sites()]
    return spark.createDataFrame(rows, schema=SITE_SCHEMA)


def get_spark(app_name):
    """Local mode by default; set SPARK_MASTER=spark://host:7077 to run on a cluster."""
    master = os.environ.get("SPARK_MASTER", "local[*]")
    spark = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", os.environ.get("SHUFFLE_PARTITIONS", "32"))
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.driver.memory", os.environ.get("DRIVER_MEMORY", "4g"))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark
