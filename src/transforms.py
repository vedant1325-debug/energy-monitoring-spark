"""Pure DataFrame transformations used by the batch and streaming jobs (unit-tested)."""
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from common import INTERVAL_HOURS

KEYS = ["site_id", "meter_id", "reading_ts"]
MAX_KWH_PER_INTERVAL = 10_000.0  # sanity cap: anything above is a corrupt reading


# ---------------------------------------------------------------- data quality
def quality_report(raw: DataFrame, sites: DataFrame) -> dict:
    """Count the problems in the raw feed before cleaning."""
    stats = raw.agg(
        F.count(F.lit(1)).alias("rows"),
        F.sum(F.col("energy_kwh").isNull().cast("int")).alias("null_energy"),
        F.sum((F.col("energy_kwh") < 0).cast("int")).alias("negative_energy"),
        F.sum((F.col("energy_kwh") > MAX_KWH_PER_INTERVAL).cast("int")).alias("over_cap"),
        F.sum(F.col("reading_ts").isNull().cast("int")).alias("null_timestamp"),
    ).first().asDict()
    dupes = (raw.groupBy(*KEYS).count().filter("count > 1")
             .agg(F.sum(F.col("count") - 1).alias("d")).first()["d"])
    orphans = raw.join(F.broadcast(sites), "site_id", "left_anti").count()
    stats.update(duplicate_rows=int(dupes or 0), unknown_site_rows=orphans)
    return {k: int(v or 0) for k, v in stats.items()}


def clean_readings(raw: DataFrame) -> DataFrame:
    return (raw.dropna(subset=KEYS + ["energy_kwh"])
               .filter((F.col("energy_kwh") >= 0) & (F.col("energy_kwh") <= MAX_KWH_PER_INTERVAL))
               .dropDuplicates(KEYS))


# ---------------------------------------------------------------- enrichment
def enrich(readings: DataFrame, sites: DataFrame) -> DataFrame:
    """Broadcast-join site metadata, convert to local time, derive cost and carbon."""
    df = readings.join(F.broadcast(sites), "site_id", "inner")
    local_ts = F.from_utc_timestamp("reading_ts", F.col("timezone"))
    hour = F.hour("local_ts")
    weekend = F.dayofweek("local_ts").isin(1, 7)  # 1 = Sunday, 7 = Saturday
    return (df.withColumn("local_ts", local_ts)
              .withColumn("local_date", F.to_date("local_ts"))
              .withColumn("year_month", F.date_format("local_ts", "yyyy-MM"))
              .withColumn("local_hour", hour)
              .withColumn("is_weekend", weekend)
              .withColumn("is_business_hours",
                          (~weekend) & (hour >= F.col("open_hour")) & (hour < F.col("close_hour")))
              .withColumn("power_kw", F.col("energy_kwh") / F.lit(INTERVAL_HOURS))
              .withColumn("cost_usd", F.col("energy_kwh") * F.col("tariff_usd_per_kwh"))
              .withColumn("co2_kg", F.col("energy_kwh") * F.col("grid_kgco2_per_kwh")))


# ---------------------------------------------------------------- analytics
def _totals():
    return [F.sum("energy_kwh").alias("kwh"), F.sum("cost_usd").alias("cost_usd"),
            F.sum("co2_kg").alias("co2_kg")]


def site_interval_demand(df: DataFrame) -> DataFrame:
    """Site-level demand per 15-min interval (sum of all meters)."""
    return df.groupBy("site_id", "reading_ts").agg(F.sum("power_kw").alias("site_kw"))


def site_summary(df: DataFrame) -> DataFrame:
    peaks = (site_interval_demand(df).groupBy("site_id")
             .agg(F.max("site_kw").alias("peak_kw"),
                  F.percentile_approx("site_kw", 0.99).alias("p99_demand_kw")))
    summary = (df.groupBy("site_id", "site_name", "city", "country", "region", "site_type",
                          "floor_area_sqm")
                 .agg(*_totals(),
                      F.sum(F.when(~F.col("is_business_hours"), F.col("energy_kwh"))).alias("after_hours_kwh"))
                 .join(peaks, "site_id")
                 .withColumn("eui_kwh_per_m2", F.col("kwh") / F.col("floor_area_sqm"))
                 .withColumn("after_hours_share", F.col("after_hours_kwh") / F.col("kwh")))
    rank_w = Window.partitionBy("region").orderBy(F.desc("kwh"))
    return summary.withColumn("rank_in_region", F.rank().over(rank_w)).orderBy(F.desc("kwh"))


def region_monthly(df: DataFrame) -> DataFrame:
    return df.groupBy("region", "year_month").agg(*_totals()).orderBy("year_month", "region")


def category_breakdown(df: DataFrame) -> DataFrame:
    w = Window.partitionBy("site_type")
    return (df.groupBy("site_type", "category").agg(F.sum("energy_kwh").alias("kwh"))
              .withColumn("share", F.col("kwh") / F.sum("kwh").over(w))
              .orderBy("site_type", F.desc("kwh")))


def hourly_profile(df: DataFrame) -> DataFrame:
    """Average site demand (kW) by local hour, weekday vs weekend, per site type."""
    per_site = (df.groupBy("site_id", "site_type", "reading_ts", "local_hour", "is_weekend")
                  .agg(F.sum("power_kw").alias("site_kw")))
    return (per_site.groupBy("site_type", "is_weekend", "local_hour")
                    .agg(F.avg("site_kw").alias("avg_kw"))
                    .orderBy("site_type", "is_weekend", "local_hour"))


def after_hours_monthly(df: DataFrame) -> DataFrame:
    """Share of energy used outside business hours, per occupied site per month."""
    occupied = df.filter(F.col("site_type").isin("Office", "R&D Lab"))
    return (occupied.groupBy("site_id", "site_name", "year_month")
                    .agg(F.sum("energy_kwh").alias("kwh"),
                         F.sum(F.when(~F.col("is_business_hours"), F.col("energy_kwh"))).alias("after_hours_kwh"),
                         F.sum(F.when(~F.col("is_business_hours"), F.col("cost_usd"))).alias("after_hours_cost_usd"))
                    .withColumn("after_hours_share", F.col("after_hours_kwh") / F.col("kwh"))
                    .orderBy("site_id", "year_month"))


def datacenter_pue(df: DataFrame) -> DataFrame:
    """Power Usage Effectiveness = total facility energy / IT energy."""
    dc = df.filter(F.col("site_type") == "Data Center")
    return (dc.groupBy("site_id", "site_name", "year_month")
              .agg(F.sum("energy_kwh").alias("total_kwh"),
                   F.sum(F.when(F.col("category") == "IT", F.col("energy_kwh"))).alias("it_kwh"))
              .withColumn("pue", F.col("total_kwh") / F.col("it_kwh"))
              .orderBy("site_id", "year_month"))


def site_daily(df: DataFrame) -> DataFrame:
    return (df.groupBy("site_id", "site_type", "local_date")
              .agg(F.sum("energy_kwh").alias("kwh"), F.avg("ambient_temp_c").alias("avg_temp_c"),
                   F.count(F.lit(1)).alias("readings")))


# ---------------------------------------------------------------- anomalies
def detect_spikes(df: DataFrame, z_threshold: float = 4.0) -> DataFrame:
    """Flag readings far above that meter's normal level for the same hour and day type.

    The baseline is a window over (meter, local hour, weekday/weekend), so every meter
    is compared only with itself at comparable times - no global threshold needed.
    """
    w = Window.partitionBy("meter_id", "local_hour", "is_weekend")
    scored = (df.withColumn("baseline_kwh", F.avg("energy_kwh").over(w))
                .withColumn("baseline_std", F.stddev("energy_kwh").over(w))
                .withColumn("z_score", (F.col("energy_kwh") - F.col("baseline_kwh")) / F.col("baseline_std")))
    return (scored.filter((F.col("z_score") > z_threshold) &
                          (F.col("energy_kwh") > 2 * F.col("baseline_kwh")))
                  .select("site_id", "site_name", "meter_id", "category", "reading_ts", "local_ts",
                          "energy_kwh", "baseline_kwh", "z_score",
                          ((F.col("energy_kwh") - F.col("baseline_kwh")) * F.col("tariff_usd_per_kwh"))
                          .alias("excess_cost_usd")))


def detect_meter_faults(df: DataFrame, min_intervals: int = 8) -> DataFrame:
    """Find runs of >= min_intervals consecutive zero readings (gaps-and-islands)."""
    by_meter = Window.partitionBy("meter_id").orderBy("reading_ts")
    by_meter_state = Window.partitionBy("meter_id", "is_zero").orderBy("reading_ts")
    flagged = (df.select("site_id", "site_name", "meter_id", "reading_ts", "energy_kwh")
                 .withColumn("is_zero", F.col("energy_kwh") <= 1e-6)
                 .withColumn("run_id", F.row_number().over(by_meter) - F.row_number().over(by_meter_state)))
    return (flagged.filter("is_zero")
                   .groupBy("site_id", "site_name", "meter_id", "run_id")
                   .agg(F.min("reading_ts").alias("start_utc"), F.max("reading_ts").alias("end_utc"),
                        F.count(F.lit(1)).alias("zero_intervals"))
                   .filter(F.col("zero_intervals") >= min_intervals)
                   .withColumn("hours_lost", F.col("zero_intervals") * F.lit(INTERVAL_HOURS))
                   .drop("run_id"))
