"""Unit tests for the Spark transformations, on tiny hand-built DataFrames."""
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import READING_SCHEMA, get_spark, sites_df  # noqa: E402
import transforms as T  # noqa: E402

T0 = datetime(2026, 1, 5, 10, 0)  # Monday 10:00 UTC


@pytest.fixture(scope="session")
def spark():
    import os
    os.environ["SPARK_MASTER"] = "local[2]"
    os.environ["SHUFFLE_PARTITIONS"] = "2"
    s = get_spark("tests")
    yield s
    s.stop()


def readings(spark, rows):
    return spark.createDataFrame(rows, READING_SCHEMA)


def r(ts, kwh, meter="S07-HVAC", site="S07", cat="HVAC"):
    return (ts, site, meter, cat, kwh, 230.0, 10.0)


def test_clean_removes_nulls_negatives_and_duplicates(spark):
    df = readings(spark, [r(T0, 5.0), r(T0, 5.0), r(T0 + timedelta(minutes=15), None),
                          r(T0 + timedelta(minutes=30), -3.0), r(T0 + timedelta(minutes=45), 4.0)])
    assert T.clean_readings(df).count() == 2


def test_quality_report_counts_problems(spark):
    df = readings(spark, [r(T0, 5.0), r(T0, 5.0), r(T0 + timedelta(minutes=15), None),
                          r(T0 + timedelta(minutes=30), -3.0), r(T0, 1.0, site="XX", meter="XX-1")])
    dq = T.quality_report(df, sites_df(spark))
    assert dq["rows"] == 5
    assert dq["duplicate_rows"] == 1
    assert dq["null_energy"] == 1
    assert dq["negative_energy"] == 1
    assert dq["unknown_site_rows"] == 1


def test_enrich_converts_time_and_computes_cost(spark):
    # 10:00 UTC on Monday is 15:30 in Mumbai (S01): business hours.
    df = T.enrich(readings(spark, [r(T0, 10.0, site="S01", meter="S01-HVAC")]), sites_df(spark)).first()
    assert df["local_hour"] == 15
    assert df["is_business_hours"] is True
    assert df["power_kw"] == pytest.approx(40.0)
    assert df["cost_usd"] == pytest.approx(10.0 * 0.10)
    assert df["co2_kg"] == pytest.approx(10.0 * 0.71)


def test_spike_detection_flags_only_the_outlier(spark):
    # Same meter, same hour, 30 normal days + one 5x spike.
    rows = [r(T0 + timedelta(days=7 * i), 10.0 + (i % 3) * 0.1) for i in range(30)]
    rows.append(r(T0 + timedelta(days=7 * 30), 50.0))
    enriched = T.enrich(readings(spark, rows), sites_df(spark))
    spikes = T.detect_spikes(enriched).collect()
    assert len(spikes) == 1 and spikes[0]["energy_kwh"] == 50.0


def test_meter_fault_finds_zero_runs(spark):
    vals = [5.0] * 4 + [0.0] * 10 + [5.0] * 4 + [0.0] * 3  # one long run, one short
    rows = [r(T0 + timedelta(minutes=15 * i), v) for i, v in enumerate(vals)]
    enriched = T.enrich(readings(spark, rows), sites_df(spark))
    faults = T.detect_meter_faults(enriched, min_intervals=8).collect()
    assert len(faults) == 1
    assert faults[0]["zero_intervals"] == 10
    assert faults[0]["hours_lost"] == pytest.approx(2.5)
