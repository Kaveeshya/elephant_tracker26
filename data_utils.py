"""
data_utils.py
Data loading, colour palettes, and geospatial/statistical helper functions
for the Kaudulla Elephant Tracker Streamlit app.

This is a faithful Python port of the data-prep and helper-function section
of the original R Shiny app (app.R).
"""
import numpy as np
import pandas as pd
import streamlit as st
from shapely.geometry import MultiPoint, Polygon
from shapely import concave_hull, convex_hull
from pyproj import Geod, Transformer

# ──────────────────────────────────────────────────────────────────────────
# Colour palettes (ported 1:1 from the R script)
# ──────────────────────────────────────────────────────────────────────────
ELEPHANT_COLOURS = {
    "Talatha": "#E63946",
    "Pazhani": "#457B9D",
    "recollared female": "#2A9D8F",
    "Rahu": "#F4A261",
    "Kasun": "#9B2226",
    "Dona": "#6A0572",
    "Mina": "#0096C7",
    "Illuk": "#52B788",
    "Dewmi": "#F77F00",
    "Gothami": "#CB4335",
    "Wilmini": "#1B4332",
    "female_1": "#B5838D",
    "Tara Devi": "#D4A017",
    "Damien": "#3D405B",
}

TRACKING_PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
    "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000",
]

MCP_PALETTE = [
    "#FF5252", "#2196F3", "#4CAF50", "#FFC107", "#9C27B0",
    "#FF9800", "#00BCD4", "#8BC34A", "#E91E63", "#3F51B5",
    "#795548", "#607D8B", "#CDDC39", "#F44336",
]

# Week-of-month colours (4 fixed, highly distinct) for the migration map —
# only one month is shown at a time, so there are always at most 4 weeks
# on screen (days 1-7, 8-14, 15-21, 22-end), one clear colour each.
WEEK_OF_MONTH_COLORS = {
    "Week 1": "#e6194b",  # red
    "Week 2": "#3cb44b",  # green
    "Week 3": "#4363d8",  # blue
    "Week 4": "#f58231",  # orange
}


def week_of_month(dt_series):
    """Bucket a datetime series into 'Week 1'..'Week 4' (days 1-7, 8-14,
    15-21, 22-end), matching the R app's week_of_month factor exactly."""
    day = pd.to_datetime(dt_series).dt.day
    bucket = ((day - 1) // 7 + 1).clip(upper=4)
    return "Week " + bucket.astype(str)

REF_LINES_LAT = [
    {"val": 8.140, "color": "#4fc3f7", "label": "Kaudulla Tank (~8.140\u00b0N)"},
    {"val": 8.080, "color": "#ef9a9a", "label": "S. Park Boundary (~8.080\u00b0N)"},
    {"val": 8.220, "color": "#ef9a9a", "label": "N. Park Boundary (~8.220\u00b0N)"},
]
REF_LINES_LON = [
    {"val": 80.895, "color": "#4fc3f7", "label": "Kaudulla Tank (~80.895\u00b0E)"},
    {"val": 80.950, "color": "#ef9a9a", "label": "E. Park Boundary (~80.950\u00b0E)"},
    {"val": 80.872, "color": "#ef9a9a", "label": "W. Park Boundary (~80.872\u00b0E)"},
]

PARK_BOUNDARY = {"lat_min": 8.080, "lat_max": 8.220, "lon_min": 80.872, "lon_max": 80.950}

CLIMATE_PLOT_INFO = {
    "Solar Radiation": dict(col="solar_radiation",
                             breaks=[0, 15, 20, 22, 23.5, 25, 26.5, 28],
                             labels=["0-15", "15-20", "20-22", "22-23.5", "23.5-25", "25-26.5", "26.5-28"],
                             title="Daily Solar Radiation Calendar Heatmap"),
    "Rainfall": dict(col="rainfall",
                      breaks=[0, 0.3, 0.6, 1.5, 2.5, 4, 10, 55],
                      labels=["0-0.3", "0.3-0.6", "0.6-1.5", "1.5-2.5", "2.5-4", "4-10", "10-55"],
                      title="Daily Rainfall Calendar Heatmap"),
    "Pressure": dict(col="pressure",
                      breaks=[0, 100.8, 101.0, 101.1, 101.2, 101.3, 101.4, 101.5],
                      labels=["0-100.8", "100.8-101", "101-101.1", "101.1-101.2", "101.2-101.3", "101.3-101.4", "101.4-101.5"],
                      title="Daily Pressure Calendar Heatmap"),
    "Maximum Temperature": dict(col="temp_max",
                                 breaks=[0, 26.7, 27.4, 28.1, 28.8, 29.5, 30.2, 31],
                                 labels=["0-26.7", "26.7-27.4", "27.4-28.1", "28.1-28.8", "28.8-29.5", "29.5-30.2", "30.2-31"],
                                 title="Daily Maximum Temperature Calendar Heatmap"),
    "Earth Skin Temperature": dict(col="temp_skin",
                                    breaks=[0, 27.2, 27.9, 28.6, 29.3, 30, 30.7, 31.5],
                                    labels=["0-27.2", "27.2-27.9", "27.9-28.6", "28.6-29.3", "29.3-30", "30-30.7", "30.7-31.5"],
                                    title="Daily Earth Skin Temperature Calendar Heatmap"),
    "Wind Speed": dict(col="wind_speed",
                        breaks=[0, 2, 4, 5, 6, 7, 8, 10],
                        labels=["0-2", "2-4", "4-5", "5-6", "6-7", "7-8", "8-10"],
                        title="Daily Wind Speed Calendar Heatmap"),
    "Maximum Wind Speed": dict(col="wind_speed_max",
                                breaks=[0, 3, 5, 6, 7, 8, 9, 11],
                                labels=["0-3", "3-5", "5-6", "6-7", "7-8", "8-9", "9-11"],
                                title="Daily Maximum Wind Speed Calendar Heatmap"),
}
CALENDAR_COLORS = ["#FFFF99", "#FFCC66", "#F5B27A", "#FF6F91", "#9966CC", "#330066", "#000000"]
AVAILABILITY_COLORS = ["#D9D9D9", "#FEE08B", "#D9EF8B", "#91CF60", "#4DAC26", "#006400"]
AVAILABILITY_BREAKS = [0, 1, 25, 50, 75, 99.999, 100]
AVAILABILITY_LABELS = ["0 %", "1 - 25 %", "25 - 50 %", "50 - 75 %", "75 - 99 %", "100 %"]


def get_color(name, palette=ELEPHANT_COLOURS, default="#888888"):
    return palette.get(name, default)


def build_elephant_palette(names, base_palette):
    """Assign each name (sorted) a colour from base_palette, cycling if needed."""
    names_sorted = sorted(names)
    return {n: base_palette[i % len(base_palette)] for i, n in enumerate(names_sorted)}


# ──────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_elephants(path="kaudulla_elephants_clean.csv"):
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, format="ISO8601")
    df["datetime_sl"] = df["datetime"].dt.tz_convert("Asia/Colombo")
    df["date_parsed"] = df["datetime_sl"].dt.date
    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    df["year"] = df["datetime_sl"].dt.strftime("%Y")
    df["month"] = df["datetime_sl"].dt.strftime("%m")
    df["year_month"] = df["datetime_sl"].dt.strftime("%Y-%m")
    if "imputed" in df.columns:
        df["imputed"] = df["imputed"].astype(str).str.lower().isin(["true", "1", "t", "yes"])
    else:
        df["imputed"] = False
    return df


@st.cache_data(show_spinner=False)
def load_climate(path="daily_climate.xlsx"):
    df = pd.read_excel(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_climate_hourly(path="POWER_Point_Hourly_kawudulla_new.csv"):
    """Raw NASA POWER hourly climate CSV — has a variable-length metadata
    header ending in '-END HEADER-', then YEAR,MO,DY,HR,T2M,PRECTOTCORR,
    WS2M,WD2M columns. -999 is NASA POWER's missing-value sentinel."""
    with open(path, "r") as f:
        lines = f.readlines()
    header_end = next(i for i, l in enumerate(lines) if l.strip() == "-END HEADER-")
    df = pd.read_csv(path, skiprows=header_end + 1)
    df = df.replace(-999, np.nan)
    df["date"] = pd.to_datetime(dict(year=df["YEAR"], month=df["MO"], day=df["DY"]))
    df["datetime"] = df["date"] + pd.to_timedelta(df["HR"], unit="h")
    return df


def month_choices(df):
    """Return dict {label: key} sorted chronologically, with 'All months' first."""
    m = (
        df.assign(month_key=df["date_parsed"].astype(str).str.slice(0, 7))
        .drop_duplicates("month_key")
        .sort_values("month_key")
    )
    m["month_label"] = pd.to_datetime(m["month_key"] + "-01").dt.strftime("%B %Y")
    choices = {"All months": "all"}
    for _, r in m.iterrows():
        choices[r["month_label"]] = r["month_key"]
    return choices


# ──────────────────────────────────────────────────────────────────────────
# Track helpers: gap detection so lines don't join across long data gaps
# ──────────────────────────────────────────────────────────────────────────
def insert_gaps(df, time_col="datetime_sl", cols=("lat", "lon")):
    """Insert a NaN row in the middle of unusually large time gaps so that
    plotly (with connectgaps=False) leaves a visible break in the line."""
    d = df.sort_values(time_col).reset_index(drop=True)
    if len(d) < 2:
        return d
    t = d[time_col].astype("int64") / 1e9  # seconds
    gaps = t.diff().dropna().values
    pos_gaps = gaps[gaps > 0]
    med = np.median(pos_gaps) if len(pos_gaps) else 3600
    if not np.isfinite(med):
        med = 3600
    thr = max(med * 4, 6 * 3600)
    big_idx = np.where(gaps > thr)[0]  # index i means gap between row i and i+1
    if len(big_idx) == 0:
        return d
    na_rows = d.iloc[big_idx].copy().reset_index(drop=True)
    gap_offsets = pd.to_timedelta(gaps[big_idx] / 2, unit="s")
    na_rows[time_col] = d[time_col].iloc[big_idx].reset_index(drop=True) + gap_offsets
    for c in cols:
        na_rows[c] = np.nan
    out = pd.concat([d, na_rows], ignore_index=True).sort_values(time_col).reset_index(drop=True)
    return out


# ──────────────────────────────────────────────────────────────────────────
# MCP / home-range helpers
# ──────────────────────────────────────────────────────────────────────────
_GEOD = Geod(ellps="WGS84")


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def compute_bearing(lat1, lon1, lat2, lon2):
    lat1r, lon1r, lat2r, lon2r = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2r - lon1r
    x = np.sin(dlon) * np.cos(lat2r)
    y = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360) % 360


def add_movement_metrics(df):
    df = df.sort_values(["name", "datetime"]).copy()
    g = df.groupby("name")
    df["prev_lat"] = g["lat"].shift()
    df["prev_lon"] = g["lon"].shift()
    df["prev_time"] = g["datetime"].shift()
    df["step_km"] = haversine_km(df["prev_lat"], df["prev_lon"], df["lat"], df["lon"])
    df["hours"] = (df["datetime"] - df["prev_time"]).dt.total_seconds() / 3600
    df["speed_kmh"] = np.where(df["hours"] > 0, df["step_km"] / df["hours"], np.nan)
    df["bearing"] = compute_bearing(df["prev_lat"], df["prev_lon"], df["lat"], df["lon"])
    df = df.drop(columns=["prev_lat", "prev_lon", "prev_time"])
    return df


def compute_hull(df, ratio=0.3):
    """Concave hull (ratio=0.3) matching R's st_concave_hull, with a convex
    hull fallback. Returns dict(lons, lats, area_km2) or None."""
    if len(df) < 3:
        return None
    pts = MultiPoint(list(zip(df["lon"].values, df["lat"].values)))
    try:
        hull = concave_hull(pts, ratio=ratio, allow_holes=False)
        if hull.is_empty or hull.geom_type != "Polygon":
            hull = convex_hull(pts)
    except Exception:
        hull = convex_hull(pts)
    if hull.is_empty or hull.geom_type != "Polygon":
        return None
    xs, ys = hull.exterior.coords.xy
    area_m2, _ = _GEOD.geometry_area_perimeter(hull)
    area_km2 = abs(area_m2) / 1e6
    return {"lons": list(xs), "lats": list(ys), "area_km2": area_km2}


def bin_bearings(bearings, n_bins=16):
    bearings = np.asarray(bearings, dtype=float)
    bearings = bearings[np.isfinite(bearings)]
    bin_width = 360 / n_bins
    bin_labels = np.arange(0, 360, bin_width)
    idx = np.clip((bearings // bin_width).astype(int), 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins)
    return pd.DataFrame({"theta": bin_labels, "r": counts})


# ──────────────────────────────────────────────────────────────────────────
# Density & Climate tab helpers
# ──────────────────────────────────────────────────────────────────────────
# UTM zone 44N suits Sri Lanka — used only to build metrically-correct hexagons.
_UTM44N = Transformer.from_crs("EPSG:4326", "EPSG:32644", always_xy=True)
_UTM44N_INV = Transformer.from_crs("EPSG:32644", "EPSG:4326", always_xy=True)


def to_utm(lons, lats):
    return _UTM44N.transform(np.asarray(lons), np.asarray(lats))


def from_utm(xs, ys):
    return _UTM44N_INV.transform(np.asarray(xs), np.asarray(ys))


def suitable_hex_bins(n):
    """Pick a 'suitable' number of hexagon columns from the sample size,
    matching the R app's dc_suitable_bins()."""
    if n <= 1:
        return 5
    return max(6, min(40, round(np.sqrt(n))))


DC_TEMP_LABELS = ["Cool (<24°C)", "Mild (24-27°C)", "Warm (27-30°C)", "Hot (30-33°C)", "Very Hot (>33°C)"]
DC_TEMP_COLORS = dict(zip(DC_TEMP_LABELS, ["#4A6FA5", "#7FA65C", "#E0A83E", "#D9723A", "#B33A3A"]))
DC_TEMP_BREAKS = [-np.inf, 24, 27, 30, 33, np.inf]

DC_RAIN_LABELS = [
    "Dry / Very Light (<2.5 mm/day)", "Light (2.5 - 25 mm/day)", "Moderate (25 - 50 mm/day)",
    "Fairly Heavy (50 - 100 mm/day)", "Heavy (100 - 150 mm/day)", "Very Heavy (>150 mm/day)",
]
DC_RAIN_COLORS = dict(zip(DC_RAIN_LABELS, ["#E8D9A0", "#C2E699", "#78C679", "#31A354", "#006837", "#081D58"]))
DC_RAIN_BREAKS = [-np.inf, 2.5, 25, 50, 100, 150, np.inf]


def build_climate_ym_category(hourly_df, value_col, breaks, labels):
    """Average a climate variable per (year-month x Day/Night period) and
    bucket it into a category, matching the R app's dc_weather_ym_temp /
    dc_weather_ym_rain tables."""
    d = hourly_df.copy()
    d["period"] = np.where((d["HR"] >= 6) & (d["HR"] < 18), "Day", "Night")
    grp = d.groupby(["YEAR", "MO", "period"])[value_col].mean().reset_index(name="avg_value")
    grp["year_month"] = pd.to_datetime(dict(year=grp["YEAR"], month=grp["MO"], day=1))
    grp["category"] = pd.cut(grp["avg_value"], bins=breaks, labels=labels)
    return grp[["year_month", "period", "avg_value", "category"]]


# ──────────────────────────────────────────────────────────────────────────
# Day / Night tab helpers
# ──────────────────────────────────────────────────────────────────────────
DN_KAUD_LAT, DN_KAUD_LON = 8.168, 80.913

DN_PERIOD_5_COLS = {
    "Hot 1 (Jun-Oct 2024)": "#991b1b",
    "Cool 1 (Nov 2024 - Mar 2025)": "#1e40af",
    "Hot 2 (Apr-Oct 2025)": "#ef4444",
    "Cool 2 (Nov 2025 - Mar 2026)": "#93c5fd",
    "Hot 3 (Apr-Jun 2026)": "#fca5a5",
}
DN_RAIN_2_COLS = {"Heavy": "#dc2626", "Low": "#16a34a"}
# NOTE: the original R app's DN_HOURLY_RAIN_COLS constant used mismatched
# label text ("10 - 50 mm/hr" / "> 50 mm/hr") that didn't match the actual
# classification thresholds (10/20, not 10/50) used to generate the
# category values — meaning those two buckets silently fell back to gray
# in the original app. Fixed here to use the thresholds that are actually
# computed, matching the sidebar legend text.
DN_HOURLY_RAIN_COLS = {
    "> 0 - 2 mm/hr": "#93c5fd",
    "2 - 5 mm/hr": "#3b82f6",
    "5 - 10 mm/hr": "#1d4ed8",
    "10 - 20 mm/hr": "#1e3a8a",
    "> 20 mm/hr": "#312e81",
}

DN_TRANSITIONS = ["2024-11-01", "2025-04-01", "2025-11-01", "2026-04-01"]
DN_HC_BANDS = [
    {"x0": "2024-06-01", "x1": "2024-11-01", "fill": "rgba(253,186,116,0.35)"},
    {"x0": "2024-11-01", "x1": "2025-04-01", "fill": "rgba(147,197,253,0.35)"},
    {"x0": "2025-04-01", "x1": "2025-11-01", "fill": "rgba(253,186,116,0.35)"},
    {"x0": "2025-11-01", "x1": "2026-04-01", "fill": "rgba(147,197,253,0.35)"},
    {"x0": "2026-04-01", "x1": "2026-07-05", "fill": "rgba(253,186,116,0.35)"},
]

DN_RAIN_TRANSITIONS = [
    "2024-10-01", "2024-12-01", "2025-01-01", "2025-02-01", "2025-04-01", "2025-06-01",
    "2025-10-01", "2025-12-01", "2026-01-01", "2026-02-01", "2026-04-01", "2026-06-01",
]
DN_RAIN_BANDS = [
    {"x0": "2024-06-01", "x1": "2024-10-01", "fill": "rgba(254,215,170,0.40)"},
    {"x0": "2024-10-01", "x1": "2024-12-01", "fill": "rgba(147,197,253,0.40)"},
    {"x0": "2024-12-01", "x1": "2025-01-01", "fill": "rgba(254,215,170,0.40)"},
    {"x0": "2025-01-01", "x1": "2025-02-01", "fill": "rgba(147,197,253,0.40)"},
    {"x0": "2025-02-01", "x1": "2025-04-01", "fill": "rgba(254,215,170,0.40)"},
    {"x0": "2025-04-01", "x1": "2025-06-01", "fill": "rgba(147,197,253,0.40)"},
    {"x0": "2025-06-01", "x1": "2025-10-01", "fill": "rgba(254,215,170,0.40)"},
    {"x0": "2025-10-01", "x1": "2025-12-01", "fill": "rgba(147,197,253,0.40)"},
    {"x0": "2025-12-01", "x1": "2026-01-01", "fill": "rgba(254,215,170,0.40)"},
    {"x0": "2026-01-01", "x1": "2026-02-01", "fill": "rgba(147,197,253,0.40)"},
    {"x0": "2026-02-01", "x1": "2026-04-01", "fill": "rgba(254,215,170,0.40)"},
    {"x0": "2026-04-01", "x1": "2026-06-01", "fill": "rgba(147,197,253,0.40)"},
    {"x0": "2026-06-01", "x1": "2026-07-05", "fill": "rgba(254,215,170,0.40)"},
]


def dn_classify_hot_cool(dates):
    d = pd.to_datetime(dates)
    conditions = [d < "2024-11-01", d < "2025-04-01", d < "2025-11-01", d < "2026-04-01"]
    return np.select(conditions, ["Hot", "Cool", "Hot", "Cool"], default="Hot")


def dn_classify_period(dates):
    d = pd.to_datetime(dates)
    conditions = [d < "2024-11-01", d < "2025-04-01", d < "2025-11-01", d < "2026-04-01"]
    choices = ["Hot 1 (Jun-Oct 2024)", "Cool 1 (Nov 2024 - Mar 2025)",
               "Hot 2 (Apr-Oct 2025)", "Cool 2 (Nov 2025 - Mar 2026)"]
    return np.select(conditions, choices, default="Hot 3 (Apr-Jun 2026)")


def dn_classify_rain(dates):
    d = pd.to_datetime(dates)
    conditions = [
        d < "2024-10-01", d < "2024-12-01", d < "2025-01-01", d < "2025-02-01",
        d < "2025-04-01", d < "2025-06-01", d < "2025-10-01", d < "2025-12-01",
        d < "2026-01-01", d < "2026-02-01", d < "2026-04-01", d < "2026-06-01",
    ]
    choices = ["Low", "Heavy", "Low", "Heavy", "Low", "Heavy", "Low", "Heavy", "Low", "Heavy", "Low", "Heavy"]
    return np.select(conditions, choices, default="Low")


def build_dn_gps(elephants_df, hourly_df):
    """GPS fixes tagged with day/night, hot/cool, 5-period, rain-period, and
    the matching hourly rainfall rate — matches the R app's dn_gps. Uses the
    RAW (UTC-labelled) datetime, not the Colombo-local one, since that's
    what the original app classifies against."""
    d = elephants_df.copy()
    dt = d["datetime"]
    dt_naive = dt.dt.tz_localize(None) if dt.dt.tz is not None else dt
    d["dt_round"] = dt_naive.dt.floor("h")
    hour = dt_naive.dt.hour
    date_dn = dt_naive.dt.floor("D")
    d["day_night"] = np.where((hour >= 6) & (hour < 18), "Day", "Night")
    d["hot_cool"] = dn_classify_hot_cool(date_dn)
    d["period"] = dn_classify_period(date_dn)
    d["rain"] = dn_classify_rain(date_dn)

    hourly = hourly_df[["datetime", "PRECTOTCORR"]].rename(
        columns={"datetime": "dt_round", "PRECTOTCORR": "rain_mm_hr"})
    d = d.merge(hourly, on="dt_round", how="left")
    return d


def build_dn_daily_temp(hourly_df):
    daily = hourly_df.groupby("date")["T2M"].mean().reset_index(name="T_avg").dropna(subset=["T_avg"])
    daily["hot_cool"] = dn_classify_hot_cool(daily["date"])
    return daily


def build_dn_daily_rain(hourly_df):
    daily = hourly_df.groupby("date")["PRECTOTCORR"].mean().reset_index(name="P_avg").dropna(subset=["P_avg"])
    daily["rain"] = dn_classify_rain(daily["date"])
    return daily


# ──────────────────────────────────────────────────────────────────────────
# Vegetation Tracking tab helpers
# ──────────────────────────────────────────────────────────────────────────
VT_KAUD_LAT, VT_KAUD_LON = 8.175, 80.913
VT_FOCUS_YEARS = (2024, 2025, 2026)

VT_YEAR_COLS = {"2024": "#ea580c", "2025": "#2563eb", "2026": "#16a34a"}
VT_GPS_YEAR_COLS = {"2024": "#f97316", "2025": "#3b82f6", "2026": "#22c55e"}
VT_DENS_COLS = {
    "Non-Vegetation": "#92714a", "Shrubs & Degraded": "#b5960f", "Sparse Vegetation": "#6aab2e",
    "Moderate Canopy": "#2d6e1f", "High Density Forest": "#14400d",
}
VT_HLTH_COLS = {
    "Non-Vegetation": "#92714a", "Unhealthy Plant": "#dc2626",
    "Moderate Healthy": "#d97706", "Very Healthy": "#16a34a",
}
VT_ELE_PALETTE = [
    "#e11d48", "#7c3aed", "#0891b2", "#16a34a", "#d97706",
    "#be185d", "#1d4ed8", "#15803d", "#b45309", "#6d28d9",
    "#0e7490", "#dc2626", "#065f46", "#92400e",
]

VT_DENSITY_RAW = pd.DataFrame({
    "year": [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026],
    "Non-Vegetation": [18.12, 32.52, 33.92, 29.39, 34.25, 23.38, 25.09, 24.02, 34.44],
    "Shrubs & Degraded": [1.01, 1.97, 1.49, 1.90, 2.49, 7.15, 3.34, 4.07, 1.25],
    "Sparse Vegetation": [12.87, 8.32, 6.60, 7.60, 13.46, 20.04, 13.22, 14.84, 4.83],
    "Moderate Canopy": [68.00, 57.20, 57.98, 60.17, 49.80, 49.38, 58.04, 56.72, 59.46],
    "High Density Forest": [0.00, 0.00, 0.00, 0.95, 0.00, 0.05, 0.31, 0.36, 0.00],
})
VT_HEALTH_RAW = pd.DataFrame({
    "year": [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026],
    "Non-Vegetation": [17.88, 30.26, 33.58, 26.80, 30.53, 16.60, 18.84, 21.31, 31.14],
    "Unhealthy Plant": [14.11, 7.32, 8.43, 7.14, 11.07, 21.01, 13.95, 12.23, 5.84],
    "Moderate Healthy": [67.99, 62.42, 57.97, 66.05, 58.38, 62.39, 67.20, 66.46, 63.00],
    "Very Healthy": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
})
VT_MOD_CANOPY = pd.DataFrame({"year": [2024, 2025, 2026], "mod_canopy": [58.04, 56.72, 59.46]})

_MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def build_vt_monthly_range(elephants_df, focus_years=VT_FOCUS_YEARS):
    d = elephants_df.copy()
    d["year_int"] = d["year"].astype(int)
    d = d[d["year_int"].isin(focus_years)]
    d["month"] = d["datetime"].dt.month
    g = d.groupby(["year_int", "month", "name"]).agg(
        lat_sd=("lat", "std"), lon_sd=("lon", "std"), n_pts=("lat", "size"),
    ).reset_index()
    g = g[g["n_pts"] >= 5].copy()
    g["lat_sd"] = g["lat_sd"].fillna(0)
    g["lon_sd"] = g["lon_sd"].fillna(0)
    g["radius_km"] = np.sqrt(g["lat_sd"] ** 2 + g["lon_sd"] ** 2) * 111
    g["month_lbl"] = g["month"].apply(lambda m: _MONTH_ABBR[m - 1])
    return g.rename(columns={"year_int": "year"})


def build_vt_hotspots(elephants_df, focus_years=VT_FOCUS_YEARS):
    d = elephants_df.copy()
    d["year_int"] = d["year"].astype(int)
    d = d[d["year_int"].isin(focus_years)]
    d["lat_g"] = d["lat"].round(3)
    d["lon_g"] = d["lon"].round(3)

    all_hs = (
        d.groupby(["year_int", "lat_g", "lon_g"])["name"]
        .agg(visits="size", n_elephants="nunique", elephants=lambda s: ", ".join(sorted(s.unique())))
        .reset_index()
        .rename(columns={"lat_g": "lat", "lon_g": "lon", "year_int": "year"})
    )
    each_hs = (
        d.groupby(["year_int", "name", "lat_g", "lon_g"])
        .size().reset_index(name="visits")
        .rename(columns={"lat_g": "lat", "lon_g": "lon", "year_int": "year"})
    )
    return all_hs, each_hs


def vt_get_top(hs_df, n_top, name=None):
    df = hs_df if name is None else hs_df[hs_df["name"] == name]
    out = []
    for yr, g in df.groupby("year"):
        top = g.nlargest(n_top, "visits").reset_index(drop=True)
        top["rank"] = range(1, len(top) + 1)
        out.append(top)
    return pd.concat(out, ignore_index=True) if out else df.iloc[0:0].assign(rank=[])


def build_vt_fix_counts(elephants_df, focus_years=VT_FOCUS_YEARS):
    d = elephants_df.copy()
    d["year_int"] = d["year"].astype(int)
    d = d[d["year_int"].isin(focus_years)]
    g = d.groupby(["year_int", "name"]).size().reset_index(name="fixes")
    g["year"] = g["year_int"].astype(str)
    return g[["year", "name", "fixes"]]


# ──────────────────────────────────────────────────────────────────────────
# General Tracking tab helpers
# ──────────────────────────────────────────────────────────────────────────
TK_PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
    "#008080", "#e6beff", "#9a6324",
]


def build_tk_dataset(elephants_df, hourly_df):
    """GPS fixes joined to hourly climate (temp/rainfall/wind speed) at the
    floor-hour level — matches the R app's track_data + climate_data +
    base_dataset inner join. Uses the raw UTC datetime, same as the R app."""
    d = elephants_df.copy()
    dt = d["datetime"]
    dt_naive = dt.dt.tz_localize(None) if dt.dt.tz is not None else dt
    d["dt_round"] = dt_naive.dt.floor("h")
    d["date"] = dt_naive.dt.date

    clim = hourly_df.rename(columns={"datetime": "dt_round", "T2M": "temp", "PRECTOTCORR": "rainfall"})
    wind_col = next((c for c in ["WS10M", "WS2M", "WS50M"] if c in clim.columns), None)
    clim["wind_speed"] = clim[wind_col] if wind_col else np.nan
    clim = clim[["dt_round", "temp", "rainfall", "wind_speed"]]

    merged = d.merge(clim, on="dt_round", how="inner").sort_values("dt_round")
    return merged
