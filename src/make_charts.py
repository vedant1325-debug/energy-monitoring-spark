"""Turn the Spark result CSVs into charts (docs/images) and headline insights (results/insights.json).

Runs on the driver only: the inputs are small aggregated reports, not raw data.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_DIR, ROOT, load_sites  # noqa: E402

IMG = ROOT / "docs" / "images"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8b2"
REGION_COLOR = {"APAC": BLUE, "EMEA": ORANGE, "AMER": AQUA}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "lines.linewidth": 2,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(IMG / f"{name}.png", dpi=180)
    plt.close(fig)
    print(f"  docs/images/{name}.png")


def label_end(ax, x, y, text, color, dy=0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", va="center",
                color=INK, fontsize=10, fontweight="bold")
    ax.plot([x], [y], "o", color=color, ms=5)


def monthly_by_region():
    df = pd.read_csv(RESULTS_DIR / "region_monthly.csv")
    df = df[df.year_month.between("2025-11", "2026-10")]  # drop partial edge months
    fig, ax = plt.subplots(figsize=(8, 4.2))
    nudge = {"APAC": -8, "EMEA": 0, "AMER": 8}
    for region, color in REGION_COLOR.items():
        d = df[df.region == region]
        ax.plot(d.year_month, d.kwh / 1e6, color=color)
        label_end(ax, len(d) - 1, d.kwh.iloc[-1] / 1e6, region, color, nudge[region])
    ax.set_title("Monthly electricity use by region (GWh)")
    ax.set_ylim(bottom=0)
    ax.set_xticks(range(len(d)), [m[2:] for m in d.year_month], rotation=0, fontsize=9)
    ax.set_xlim(-0.3, len(d) + 0.8)
    save(fig, "monthly_by_region")


def site_cost():
    df = pd.read_csv(RESULTS_DIR / "site_summary.csv").sort_values("cost_usd")
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [REGION_COLOR[r] for r in df.region]
    ax.barh(df.site_name, df.cost_usd / 1e6, color=colors, height=0.7)
    for y, v in enumerate(df.cost_usd / 1e6):
        ax.text(v + 0.08, y, f"${v:.1f}M", va="center", fontsize=9, color=INK2)
    ax.set_title("Annual electricity cost by site (USD millions)")
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in REGION_COLOR.values()]
    ax.legend(handles, REGION_COLOR.keys(), loc="lower right", frameon=False)
    save(fig, "site_cost")


def after_hours():
    df = pd.read_csv(RESULTS_DIR / "after_hours_monthly.csv")
    df = df[df.year_month.between("2025-11", "2026-10")]
    offices = [s["site_id"] for s in load_sites() if s["site_type"] == "Office" and s["site_id"] != "S09"]
    peers = df[df.site_id.isin(offices)].groupby("year_month").after_hours_share.mean()
    dub = df[df.site_id == "S09"].set_index("year_month").after_hours_share
    x = range(len(dub))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(x, peers.values * 100, color=MUTED)
    ax.plot(x, dub.values * 100, color=ORANGE)
    label_end(ax, len(dub) - 1, peers.values[-1] * 100, "Other offices (avg)", MUTED)
    label_end(ax, len(dub) - 1, dub.values[-1] * 100, "Dublin", ORANGE)
    march = list(dub.index).index("2026-03")
    ax.axvline(march, color=INK2, ls="--", lw=1)
    ax.text(march + 0.1, 42.5, "Mar 2026: HVAC\nschedule override", fontsize=9, color=INK2)
    ax.set_ylim(40, 66)
    ax.set_xlim(-0.3, len(dub) + 2.2)
    ax.set_xticks(list(x), [m[2:] for m in dub.index], fontsize=9)
    ax.set_title("Share of energy used outside business hours (%)")
    save(fig, "after_hours_dublin")


def pue():
    df = pd.read_csv(RESULTS_DIR / "datacenter_pue_monthly.csv")
    df = df[df.year_month.between("2025-11", "2026-10")]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    names = {"S08": ("Frankfurt", ORANGE), "S02": ("Bengaluru", MUTED), "S13": ("Austin", "#8f8e88")}
    for sid, (name, color) in names.items():
        d = df[df.site_id == sid]
        ax.plot(range(len(d)), d.pue, color=color)
        label_end(ax, len(d) - 1, d.pue.iloc[-1], name, color)
    ax.set_xticks(range(len(d)), [m[2:] for m in d.year_month], fontsize=9)
    ax.set_xlim(-0.3, len(d) + 1.4)
    ax.set_title("Data-center PUE by month (lower is better)")
    save(fig, "datacenter_pue")


def hourly():
    df = pd.read_csv(RESULTS_DIR / "hourly_profile.csv")
    df = df[df.is_weekend.astype(str).str.lower() == "false"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for stype, color in [("Data Center", BLUE), ("Manufacturing", ORANGE), ("Office", AQUA)]:
        d = df[df.site_type == stype].sort_values("local_hour")
        y = 100 * d.avg_kw / d.avg_kw.max()
        ax.plot(d.local_hour, y, color=color)
        label_end(ax, 23, y.iloc[-1], stype, color)
    ax.set_xlim(0, 28)
    ax.set_ylim(0, 108)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel("Local hour (weekdays)")
    ax.set_title("Daily load shape: % of each site type's peak hour")
    save(fig, "hourly_profile")


def forecast():
    df = pd.read_csv(RESULTS_DIR / "forecast_predictions.csv", parse_dates=["local_date"])
    d = df[df.site_id == "S12"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(d.local_date, d.actual_kwh / 1000, color=MUTED, lw=4, label="Actual")
    ax.plot(d.local_date, d.predicted_kwh / 1000, color=BLUE, lw=1.6, label="MLlib forecast")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("New York HQ: daily energy, actual vs forecast (MWh)")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
    save(fig, "forecast_vs_actual")


def insights():
    ah = pd.read_csv(RESULTS_DIR / "after_hours_monthly.csv")
    dub = ah[ah.site_id == "S09"]
    before = dub[dub.year_month.between("2025-11", "2026-02")]
    after = dub[dub.year_month >= "2026-03"]
    base_share = before.after_hours_kwh.sum() / before.kwh.sum()
    business = after.kwh - after.after_hours_kwh
    expected_after = business * base_share / (1 - base_share)
    excess_kwh = float((after.after_hours_kwh - expected_after).sum())
    tariff = next(s["tariff_usd_per_kwh"] for s in load_sites() if s["site_id"] == "S09")

    pue = pd.read_csv(RESULTS_DIR / "datacenter_pue_monthly.csv")
    fra = pue[(pue.site_id == "S08") & pue.year_month.between("2025-11", "2026-10")]
    sites = pd.read_csv(RESULTS_DIR / "site_summary.csv")
    by_type = sites.groupby("site_type").kwh.sum() / sites.kwh.sum()
    out = {
        "dublin_after_hours_share_before_pct": round(100 * base_share, 1),
        "dublin_after_hours_share_after_pct": round(100 * after.after_hours_kwh.sum() / after.kwh.sum(), 1),
        "dublin_excess_mwh": round(excess_kwh / 1000, 1),
        "dublin_excess_cost_usd": round(excess_kwh * tariff),
        "frankfurt_pue_first": round(float(fra.pue.iloc[0]), 2),
        "frankfurt_pue_max": round(float(fra.pue.max()), 2),
        "energy_share_by_site_type_pct": {k: round(100 * v, 1) for k, v in by_type.sort_values(ascending=False).items()},
        "most_expensive_site": sites.sort_values("cost_usd").iloc[-1].site_name,
    }
    (RESULTS_DIR / "insights.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    for fn in (monthly_by_region, site_cost, after_hours, pue, hourly, forecast):
        fn()
    insights()
