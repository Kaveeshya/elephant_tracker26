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
from pyproj import Geod

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
AVAILABILITY_COLORS = ["#7F0000", "#E34A33", "#FDAE61", "#FFFF99", "#78C679", "#006400"]
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
    return df


@st.cache_data(show_spinner=False)
def load_climate(path="daily_climate.xlsx"):
    df = pd.read_excel(path)
    df["date"] = pd.to_datetime(df["date"])
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
