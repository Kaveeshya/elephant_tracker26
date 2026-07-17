"""
Kaudulla Elephant Tracker — Streamlit port of the original R Shiny app.
Run with:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium
from folium.plugins import Fullscreen
from streamlit_folium import st_folium
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from statsmodels.nonparametric.smoothers_lowess import lowess
import base64

import data_utils as du
from calendar_heatmap import make_calendar_heatmap

# ══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Kaudulla Elephant Tracker",
    page_icon="🐘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════
# CSS — teal / green theme matching the original shinydashboard skin
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
:root { --accent:#0f766e; --accent-light:#ecfeff; }
.stApp { background:#f5f7fb; font-family:'Segoe UI', system-ui, sans-serif; }
[data-testid="stSidebar"] { background:#ffffff; border-right:1px solid #e5e7eb; }
[data-testid="stSidebar"] .stButton button {
    background:var(--accent); color:white; border:none; border-radius:8px;
    font-size:11px; padding:4px 2px;
}
[data-testid="stSidebar"] .stButton button:hover { background:#115e59; color:white; }
h1,h2,h3 { color:#1e293b; }
.section-title { font-size:30px; font-weight:800; color:#1e293b; text-align:center; margin:10px 0 16px; letter-spacing:-0.5px; }
.sub-title { font-size:22px; font-weight:700; color:#334155; text-align:center; margin:18px 0 10px; }
.section-description { font-size:15px; color:#64748b; text-align:center; line-height:1.7; max-width:900px; margin:0 auto 20px auto; }
.card {
    background:white; border-radius:14px; padding:18px 20px; margin-bottom:18px;
    box-shadow:0 2px 12px rgba(0,0,0,.08);
}
.card-title { color:#0f766e; font-size:17px; font-weight:700; margin-bottom:6px; }
.card-note { color:#666; font-size:12px; margin-bottom:10px; line-height:1.6; }
.ref-heading { color:#0f766e; font-size:14px; font-weight:700; margin-bottom:4px;}
.ref-text { color:#475569; font-size:13px; line-height:1.65; }
.vbox {
    border-radius:14px; padding:16px 18px; color:white; box-shadow:0 3px 12px rgba(0,0,0,.10);
}
.vbox .num { font-size:26px; font-weight:800; line-height:1.1; }
.vbox .lab { font-size:12.5px; opacity:.92; margin-top:2px;}
.ref-badge {
    display:inline-flex; align-items:center; justify-content:center; width:20px; height:20px;
    border-radius:50%; font-size:11px; font-weight:700; color:white;
}
.ref-badge.core{ background:#0f766e; } .ref-badge.boundary{ background:#c1440e; }
[data-testid="stMetricValue"] { color:#0f766e; }
hr { border-color:#e5e7eb; }
</style>
""", unsafe_allow_html=True)

VBOX_COLORS = {
    "green": "linear-gradient(135deg,#16a34a,#15803d)",
    "olive": "linear-gradient(135deg,#65a30d,#4d7c0f)",
    "teal": "linear-gradient(135deg,#0f766e,#115e59)",
    "blue": "linear-gradient(135deg,#2563eb,#1e40af)",
}


def _hex_alpha(hex_color, alpha):
    """'#RRGGBB' + alpha (0-1) -> 'rgba(r,g,b,a)'. Plotly rejects 8-digit
    hex (RRGGBBAA) colors outright, so this is needed anywhere a
    semi-transparent color built from a hex string is passed to Plotly."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def vbox(value, label, icon, color="teal"):
    st.markdown(f"""
    <div class="vbox" style="background:{VBOX_COLORS.get(color, VBOX_COLORS['teal'])}">
        <div class="num">{icon} {value}</div>
        <div class="lab">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def card_open(title=None, note=None):
    html = '<div class="card">'
    if title:
        html += f'<div class="card-title">{title}</div>'
    if note:
        html += f'<div class="card-note">{note}</div>'
    st.markdown(html, unsafe_allow_html=True)


def card_close():
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════
elephants_df = du.load_elephants("kaudulla_elephants_clean.csv")
climate_df = du.load_climate("daily_climate.xlsx")
if "MONTH_CHOICES" not in st.session_state:
    st.session_state.MONTH_CHOICES = du.month_choices(elephants_df)
MONTH_CHOICES = st.session_state.MONTH_CHOICES

if "mcp_df" not in st.session_state:
    st.session_state.mcp_df = du.add_movement_metrics(elephants_df.copy())
mcp_df = st.session_state.mcp_df
ALL_NAMES = sorted(elephants_df["name"].unique().tolist())
FEMALE_NAMES = sorted(elephants_df.loc[elephants_df.sex == "Female", "name"].unique().tolist())
MALE_NAMES = sorted(elephants_df.loc[elephants_df.sex == "Male", "name"].unique().tolist())
SEXES = sorted(elephants_df["sex"].unique().tolist())

tracking_colors = du.build_elephant_palette(ALL_NAMES, du.TRACKING_PALETTE)
mcp_colors = du.build_elephant_palette(ALL_NAMES, du.MCP_PALETTE)

DATA_MIN_DATE = elephants_df["date_parsed"].min()
DATA_MAX_DATE = elephants_df["date_parsed"].max()

# ══════════════════════════════════════════════════════════════════════════
# SESSION STATE DEFAULTS
# ══════════════════════════════════════════════════════════════════════════
if "sel_elephants" not in st.session_state:
    st.session_state.sel_elephants = ALL_NAMES.copy()
if "date_range" not in st.session_state:
    st.session_state.date_range = (DATA_MIN_DATE, DATA_MAX_DATE)

# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        "<div style='display:flex;align-items:center;gap:8px;padding:4px 0 2px;'>"
        "<img src='https://upload.wikimedia.org/wikipedia/commons/1/11/Flag_of_Sri_Lanka.svg' height='22'>"
        "<b style='font-size:17px;color:#0f766e;'>Kaudulla Elephant Tracker</b></div>",
        unsafe_allow_html=True,
    )
    st.caption("**Kaudulla National Park**  \nNorth Central Province, Sri Lanka  \n8°08′N 80°54′E")
    st.caption("*GPS Collar Monitoring Programme*  \n[Wildlife Department of Sri Lanka](https://wildlife.gov.lk)")
    st.divider()

    nav = option_menu(
        None,
        ["Latitude vs Time", "Longitude vs Time", "Both Coordinates", "Heat Maps",
         "Elephant Tracking", "Live Elephant Path", "Migration & Climate",
         "Data Table", "Home Range & Speed", "Dona & Recollared", "Density & Climate", "Day / Night",
         "Vegetation Tracking", "Tracking"],
        icons=["graph-up", "graph-up", "layers", "fire", "map", "play-circle",
               "globe-americas", "table", "compass", "person-badge", "grid-3x3", "brightness-high", "tree",
               "sliders"],
        default_index=0,
        styles={
            "container": {"padding": "0", "background-color": "#ffffff"},
            "icon": {"color": "#0f766e", "font-size": "14px"},
            "nav-link": {"font-size": "13px", "font-weight": "600", "color": "#334155",
                         "border-radius": "8px", "margin": "2px 0"},
            "nav-link-selected": {"background-color": "#0f766e", "color": "white"},
        },
    )

    st.divider()

    def _set_elephants(names):
        st.session_state.sel_elephants = names

    st.multiselect("Select Elephants", ALL_NAMES, key="sel_elephants")
    c1, c2, c3, c4 = st.columns(4)
    c1.button("All", use_container_width=True, on_click=_set_elephants, args=(ALL_NAMES.copy(),))
    c2.button("Fem.", use_container_width=True, on_click=_set_elephants, args=(FEMALE_NAMES.copy(),))
    c3.button("Male", use_container_width=True, on_click=_set_elephants, args=(MALE_NAMES.copy(),))
    c4.button("Clear", use_container_width=True, on_click=_set_elephants, args=([],))

    st.divider()
    date_range = st.date_input(
        "Date Range", value=st.session_state.date_range,
        min_value=DATA_MIN_DATE, max_value=DATA_MAX_DATE, key="date_range_input",
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        st.session_state.date_range = date_range
    date_start, date_end = st.session_state.date_range

    sel_month_label = st.selectbox("Month", list(MONTH_CHOICES.keys()))
    sel_month = MONTH_CHOICES[sel_month_label]

    add_smooth = st.checkbox("Add LOESS smoother", value=False)

    st.divider()
    agg_level_label = st.radio(
        "Time Resolution", ["Raw (hourly)", "Daily mean", "Weekly mean"], index=0,
    )
    agg_level = {"Raw (hourly)": "raw", "Daily mean": "day", "Weekly mean": "week"}[agg_level_label]

    st.divider()
    st.caption("**Key Literature**  \nFernando et al. (2008)  \nPastorini et al. (2010)  \nRatnayeke et al. (2023)")

sel_elephants = st.session_state.sel_elephants

# ══════════════════════════════════════════════════════════════════════════
# FILTERED / AGGREGATED DATA
# ══════════════════════════════════════════════════════════════════════════
def get_filtered():
    df = elephants_df[
        (elephants_df["date_parsed"] >= date_start) & (elephants_df["date_parsed"] <= date_end)
    ]
    if sel_elephants:
        df = df[df["name"].isin(sel_elephants)]
    else:
        df = df.iloc[0:0]
    if sel_month != "all":
        df = df[df["year_month"] == sel_month]
    return df


def get_agg(df):
    if agg_level == "raw" or df.empty:
        return df
    d = df.copy()
    if agg_level == "day":
        d["period"] = d["datetime_sl"].dt.floor("D")
    else:
        d["period"] = d["datetime_sl"].dt.tz_localize(None).dt.to_period("W").dt.start_time.dt.tz_localize("Asia/Colombo")
    out = (
        d.groupby(["name", "sex", "period"], as_index=False)
        .agg(lat=("lat", "mean"), lon=("lon", "mean"))
        .rename(columns={"period": "datetime_sl"})
    )
    return out


def get_mcp_filtered():
    df = mcp_df[
        (mcp_df["sex"].isin([s for s in SEXES if s in st.session_state.get("mcp_sex_filter", SEXES)])) &
        (mcp_df["datetime_sl"].dt.date >= date_start) & (mcp_df["datetime_sl"].dt.date <= date_end)
    ]
    if sel_elephants:
        df = df[df["name"].isin(sel_elephants)]
    else:
        df = df.iloc[0:0]
    if sel_month != "all":
        df = df[df["year_month"] == sel_month]
    return df


filtered_df = get_filtered()
agg_df = get_agg(filtered_df)


# ══════════════════════════════════════════════════════════════════════════
# PLOTTING HELPERS
# ══════════════════════════════════════════════════════════════════════════
def add_loess(fig, sub, y_col, color, yaxis="y", dash="dot", name_suffix=" (smooth)"):
    s = sub.dropna(subset=[y_col])
    if len(s) <= 10:
        return
    x_num = s["datetime_sl"].astype("int64") / 1e9
    try:
        sm = lowess(s[y_col].values, x_num.values, frac=0.3, return_sorted=True)
    except Exception:
        return
    x_ts = pd.to_datetime(sm[:, 0], unit="s", utc=True).tz_convert("Asia/Colombo")
    fig.add_trace(go.Scatter(
        x=x_ts, y=sm[:, 1], mode="lines", name=name_suffix, yaxis=yaxis,
        line=dict(color=color, width=2.5, dash=dash), opacity=0.5,
        showlegend=False, hoverinfo="skip",
    ))


def make_plot(df, y_col, y_title, ref_lines=None):
    fig = go.Figure()
    names_in_data = df["name"].unique().tolist()
    for el in sorted(names_in_data):
        sub = df[df["name"] == el].sort_values("datetime_sl")
        sub = du.insert_gaps(sub)
        clr = du.get_color(el)
        fig.add_trace(go.Scatter(
            x=sub["datetime_sl"], y=sub[y_col], mode="lines+markers", name=el,
            connectgaps=False,
            line=dict(color=clr, width=1.5),
            marker=dict(color=clr, size=4, opacity=0.85, line=dict(color=clr, width=1)),
            text=[f"<b>{el}</b><br>Time (SL): {t:%d %b %Y %H:%M}<br>{y_title}: {v:.5f}°<br>Sex: {s}"
                  for t, v, s in zip(sub["datetime_sl"], sub[y_col], sub.get("sex", [""] * len(sub)))],
            hoverinfo="text",
        ))
        if add_smooth:
            add_loess(fig, sub, y_col, clr)
    if ref_lines:
        xmin, xmax = df["datetime_sl"].min(), df["datetime_sl"].max()
        for rl in ref_lines:
            fig.add_trace(go.Scatter(
                x=[xmin, xmax], y=[rl["val"], rl["val"]], mode="lines", name=rl["label"],
                line=dict(color=rl["color"], width=1.5, dash="dash"), hoverinfo="name",
            ))
    fig.update_layout(
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font=dict(color="#333333", family="Segoe UI"),
        xaxis=dict(title="Date / Time (Asia/Colombo)", gridcolor="#e5e5e5", tickformat="%b %Y"),
        yaxis=dict(title=y_title, gridcolor="#e5e5e5"),
        legend=dict(bgcolor="#ffffff", bordercolor="#4caf50", borderwidth=1, font=dict(size=11)),
        margin=dict(t=40, b=60, l=70, r=20),
        height=460,
    )
    return fig


def make_dual_axis_plot(df):
    fig = go.Figure()
    names_in_data = sorted(df["name"].unique().tolist())
    for el in names_in_data:
        sub = df[df["name"] == el].sort_values("datetime_sl")
        sub = du.insert_gaps(sub)
        clr = du.get_color(el)
        fig.add_trace(go.Scatter(
            x=sub["datetime_sl"], y=sub["lat"], mode="lines+markers", name=f"{el} – Lat",
            legendgroup=el, yaxis="y", connectgaps=False,
            line=dict(color=clr, width=1.5, dash="solid"),
            marker=dict(color=clr, size=4, symbol="circle", line=dict(color=clr, width=1)),
            hovertext=[f"<b>{el}</b><br>Latitude: {v:.5f}°N<br>{t:%d %b %Y %H:%M}"
                       for t, v in zip(sub["datetime_sl"], sub["lat"])], hoverinfo="text",
        ))
        fig.add_trace(go.Scatter(
            x=sub["datetime_sl"], y=sub["lon"], mode="lines+markers", name=f"{el} – Lon",
            legendgroup=el, yaxis="y2", connectgaps=False,
            line=dict(color=clr, width=1.5, dash="dot"),
            marker=dict(color=clr, size=4, symbol="triangle-up", line=dict(color=clr, width=1)),
            hovertext=[f"<b>{el}</b><br>Longitude: {v:.5f}°E<br>{t:%d %b %Y %H:%M}"
                       for t, v in zip(sub["datetime_sl"], sub["lon"])], hoverinfo="text",
        ))
        if add_smooth:
            add_loess(fig, sub, "lat", clr, yaxis="y", dash="solid", name_suffix="Lat smooth")
            add_loess(fig, sub, "lon", clr, yaxis="y2", dash="dashdot", name_suffix="Lon smooth")

    xmin, xmax = df["datetime_sl"].min(), df["datetime_sl"].max()
    for rl in du.REF_LINES_LAT:
        fig.add_trace(go.Scatter(x=[xmin, xmax], y=[rl["val"]] * 2, mode="lines", name=rl["label"],
                                  yaxis="y", line=dict(color=rl["color"], width=1, dash="dash"), hoverinfo="name"))
    for rl in du.REF_LINES_LON:
        fig.add_trace(go.Scatter(x=[xmin, xmax], y=[rl["val"]] * 2, mode="lines", name=rl["label"],
                                  yaxis="y2", line=dict(color=rl["color"], width=1, dash="dashdot"), hoverinfo="name"))

    fig.update_layout(
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font=dict(color="#333333", family="Segoe UI"),
        xaxis=dict(title="Date / Time (Asia/Colombo)", gridcolor="#e5e5e5", tickformat="%b %Y"),
        yaxis=dict(title=dict(text="Latitude (°N, WGS84)", font=dict(color="#0277bd")), gridcolor="#e5e5e5",
                    tickfont=dict(color="#0277bd")),
        yaxis2=dict(title=dict(text="Longitude (°E, WGS84)", font=dict(color="#ef6c00")), overlaying="y",
                     side="right", showgrid=False, tickfont=dict(color="#ef6c00")),
        legend=dict(bgcolor="#ffffff", bordercolor="#4caf50", borderwidth=1, font=dict(size=10)),
        margin=dict(t=40, b=60, l=70, r=70),
        height=600,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════
# MAP HELPERS
# ══════════════════════════════════════════════════════════════════════════
def base_folium_map(tiles="CartoDB positron"):
    m = folium.Map(location=[8.15, 80.905], zoom_start=12, tiles=tiles, control_scale=True, prefer_canvas=True)
    folium.TileLayer("CartoDB positron", name="Light").add_to(m)
    folium.TileLayer("OpenStreetMap", name="Street").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satellite",
    ).add_to(m)
    return m


def build_sync_map(df, up_to_time=None):
    """Faint full tracks (context) + bold path/marker up to `up_to_time`."""
    m = base_folium_map()
    names_in_data = sorted(df["name"].unique().tolist())
    bounds = []
    for el in names_in_data:
        sub = df[df["name"] == el].sort_values("datetime_sl").dropna(subset=["lat", "lon"])
        if len(sub) < 2:
            continue
        clr = du.get_color(el)
        pts = list(zip(sub["lat"], sub["lon"]))
        folium.PolyLine(pts, color=clr, weight=1.5, opacity=0.35).add_to(m)
        bounds.extend(pts)

        if up_to_time is not None:
            sub2 = sub[sub["datetime_sl"] <= up_to_time]
            if len(sub2) == 0:
                continue
            pts2 = list(zip(sub2["lat"], sub2["lon"]))
            if len(pts2) >= 2:
                folium.PolyLine(pts2, color=clr, weight=3, opacity=0.95).add_to(m)
            cur = sub2.iloc[-1]
            folium.CircleMarker(
                [cur["lat"], cur["lon"]], radius=7, color="#ffffff", weight=2,
                fill=True, fill_color=clr, fill_opacity=1,
                popup=f"<b>{el}</b><br>{cur['datetime_sl']:%d %b %Y %H:%M}<br>"
                      f"Lat: {cur['lat']:.5f}°N<br>Lon: {cur['lon']:.5f}°E",
                tooltip=f"{el} — {cur['datetime_sl']:%d %b %Y %H:%M}",
            ).add_to(m)

    if bounds:
        m.fit_bounds(bounds)
    folium.LayerControl(collapsed=True).add_to(m)
    _add_categorical_legend(m, "Elephant", {el: du.get_color(el) for el in names_in_data})
    return m


def _add_categorical_legend(m, title, color_map):
    if not color_map:
        return
    rows = "".join(
        f"<div style='display:flex;align-items:center;gap:6px;margin:2px 0;'>"
        f"<span style='width:11px;height:11px;border-radius:50%;background:{c};display:inline-block;'></span>"
        f"<span style='font-size:11px;color:#333;'>{n}</span></div>"
        for n, c in color_map.items()
    )
    html = f"""
    <div style="position: fixed; bottom: 20px; right: 20px; z-index:9999; background:white;
                padding:8px 10px; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,.25); max-height:220px; overflow-y:auto;">
        <div style="font-size:12px;font-weight:700;color:#0f766e;margin-bottom:4px;">{title}</div>
        {rows}
    </div>
    """
    m.get_root().html.add_child(folium.Element(html))


# ══════════════════════════════════════════════════════════════════════════
# MIGRATION MAP — ports R's build_migration_map()
# ══════════════════════════════════════════════════════════════════════════
def build_migration_map(dat, show_numbers=False):
    """Build the single-elephant migration map: one circle marker per GPS
    fix (colored by week-of-month), optional sequence-number labels, and
    start (green play icon) / end (red flag icon) markers. Reused for both
    the live render and the 'open in new tab' export, matching the R app."""
    if dat is None or dat.empty:
        m = folium.Map(location=[7.0, 80.0], zoom_start=7, tiles="OpenStreetMap", prefer_canvas=True)
        folium.Marker(
            [7.0, 80.0], icon=folium.Icon(color="gray"),
            popup="No elephant data available for selected year & month",
        ).add_to(m)
        return m

    m = folium.Map(tiles="OpenStreetMap", prefer_canvas=True)
    bounds = []

    for el, d in dat.groupby("name"):
        d = d.sort_values("datetime_sl").reset_index(drop=True)
        n_pts = len(d)

        for i, r in d.iterrows():
            seq = i + 1
            clr = du.WEEK_OF_MONTH_COLORS.get(r["week_of_month"], "#333333")
            folium.CircleMarker(
                [r["lat"], r["lon"]], radius=5, color=clr, fill=True, fill_color=clr,
                fill_opacity=1, weight=0,
                popup=f"<b>Elephant:</b> {el}<br><b>Sequence:</b> {seq} of {n_pts}<br>"
                      f"<b>Date:</b> {r['datetime']}<br><b>Week:</b> {r['week_of_month']}<br>"
                      f"<b>Year:</b> {r['year']}<br><b>Month:</b> {r['month']}",
            ).add_to(m)
            bounds.append((r["lat"], r["lon"]))

            if show_numbers:
                folium.Marker(
                    [r["lat"], r["lon"]],
                    icon=folium.DivIcon(html=(
                        f"<div style='font-weight:bold; font-size:12px; color:black; "
                        f"transform:translate(-50%, -22px); text-shadow:-1px -1px 0 #fff, "
                        f"1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff;'>{seq}</div>"
                    )),
                ).add_to(m)

        start_pt, end_pt = d.iloc[0], d.iloc[-1]
        folium.Marker(
            [start_pt["lat"], start_pt["lon"]],
            icon=folium.Icon(color="green", icon="play", prefix="fa"),
            popup=f"<b>{el}</b><br>Start: {start_pt['datetime']}",
        ).add_to(m)
        folium.Marker(
            [end_pt["lat"], end_pt["lon"]],
            icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa"),
            popup=f"<b>{el}</b><br>End: {end_pt['datetime']}",
        ).add_to(m)

    if bounds:
        m.fit_bounds(bounds)

    legend_html = "".join(
        f"<div style='display:flex;align-items:center;gap:6px;margin:2px 0;'>"
        f"<span style='width:11px;height:11px;border-radius:50%;background:{c};display:inline-block;'></span>"
        f"<span style='font-size:11px;color:#333;'>{w}</span></div>"
        for w, c in du.WEEK_OF_MONTH_COLORS.items()
    )
    m.get_root().html.add_child(folium.Element(f"""
    <div style="position: fixed; bottom: 20px; right: 20px; z-index:9999; background:white;
                padding:8px 10px; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,.25);">
        <div style="font-size:12px;font-weight:700;color:#0f766e;margin-bottom:4px;">Week of Month</div>
        {legend_html}
    </div>
    """))
    return m


# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — LATITUDE VS TIME
# ══════════════════════════════════════════════════════════════════════════
def render_lat_tab():
    n = len(filtered_df)
    c1, c2, c3, c4 = st.columns(4)
    with c1: vbox(f"{n:,}", "GPS Fixes (filtered)", "📍", "green")
    with c2: vbox(filtered_df["name"].nunique(), "Elephants Selected", "🐘", "olive")
    with c3:
        d = filtered_df["date_parsed"].min()
        vbox(d.strftime("%d %b %Y") if pd.notna(d) else "—", "Data From", "📅", "teal")
    with c4:
        d = filtered_df["date_parsed"].max()
        vbox(d.strftime("%d %b %Y") if pd.notna(d) else "—", "Data To", "📅", "teal")

    st.write("")
    card_open(
        "📍 Latitude vs Time — GPS Collar Data, Kaudulla National Park",
        "Kaudulla elephants range roughly between latitudes 8.10°N and 8.25°N. The park boundary lies "
        "around 8.08°–8.22°N (Fernando et al. 2008). Northward movement often corresponds to the seasonal "
        "arrival at Kaudulla tank when Minneriya dries (Ratnayeke et al. 2023).",
    )
    if agg_df.empty:
        st.info("No data for the selected filters.")
    else:
        fig = make_plot(agg_df, "lat", "Latitude (°N, WGS84)", du.REF_LINES_LAT)
        st.plotly_chart(fig, use_container_width=True, key="lat_plot")
    card_close()

    card_open(
        "🗺️ Live Position — Synced with Latitude Chart",
        "Use the time scrubber to move the map marker along each elephant's track (Streamlit's static-chart "
        "hover can't drive a live map the way Shiny's reactive hover can, so a slider gives the same "
        "point-in-time exploration).",
    )
    render_time_scrubber_map(agg_df, "lat")
    card_close()

    card_open("📚 Literature Context — Latitude & Elephant Ranging in Sri Lanka")
    lc1, lc2, lc3, lc4 = st.columns(4)
    with lc1:
        st.markdown("<div class='ref-heading'>Fernando et al. (2008)</div>"
                     "<div class='ref-text'>Home ranges of Sri Lankan elephants averaged 46–103 km², with "
                     "latitudinal movement of 0.1°–0.3° correlated with seasonal tank water levels in the dry zone.</div>",
                     unsafe_allow_html=True)
    with lc2:
        st.markdown("<div class='ref-heading'>Ratnayeke et al. (2023)</div>"
                     "<div class='ref-text'>Kaudulla–Minneriya corridor study showed elephants shift northward "
                     "into Kaudulla from May–October when Minneriya tank partially dries, peaking Aug–Sept.</div>",
                     unsafe_allow_html=True)
    with lc3:
        st.markdown("<div class='ref-heading'>Wildlife Dept. of Sri Lanka</div>"
                     "<div class='ref-text'>The GPS collar programme in Kaudulla NP (est. 2002, 6,900 ha) monitors "
                     "movement to inform HEC mitigation and corridor management. "
                     "<a href='https://wildlife.gov.lk' target='_blank'>wildlife.gov.lk</a></div>",
                     unsafe_allow_html=True)
    with lc4:
        st.markdown("<div class='ref-heading'>Geographic Reference</div>"
                     "<div class='ref-text'>Latitude 8.10°–8.25°N (WGS84). Kaudulla tank at ~8.14°N is a key "
                     "dry-season water source. The Mahaweli River floodplain at ~8.22°N forms the N. boundary.</div>",
                     unsafe_allow_html=True)
    card_close()


def render_time_scrubber_map(df, coord_col):
    if df.empty:
        st.info("No data for the selected filters.")
        return
    times = sorted(df["datetime_sl"].unique())
    idx = st.slider(
        "Time", 0, len(times) - 1, len(times) - 1,
        format="", key=f"scrub_{coord_col}_{nav}",
    )
    cur_t = pd.Timestamp(times[idx])
    st.caption(f"Showing positions up to: **{cur_t:%d %b %Y %H:%M}** (Asia/Colombo)")
    m = build_sync_map(df, up_to_time=cur_t)
    st_folium(m, height=420, use_container_width=True, key=f"map_{coord_col}_{nav}", returned_objects=[])


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — LONGITUDE VS TIME
# ══════════════════════════════════════════════════════════════════════════
def render_lon_tab():
    card_open(
        "📍 Longitude vs Time — GPS Collar Data, Kaudulla National Park",
        "Longitudes span 80.87°–80.96°E. The Kaudulla tank lies near 80.90°E. Elephants moving eastward "
        "(higher longitude) approach the park's eastern boundary, which borders agricultural land — a key "
        "HEC zone (Pastorini et al. 2010; Wildlife Dept. Sri Lanka 2023 Annual Report).",
    )
    if agg_df.empty:
        st.info("No data for the selected filters.")
    else:
        fig = make_plot(agg_df, "lon", "Longitude (°E, WGS84)", du.REF_LINES_LON)
        st.plotly_chart(fig, use_container_width=True, key="lon_plot")
    card_close()

    card_open("🗺️ Live Position — Synced with Longitude Chart",
               "Drag the time scrubber to move the map marker along each elephant's track.")
    render_time_scrubber_map(agg_df, "lon")
    card_close()

    card_open("📚 Literature Context — Longitude & East–West Ranging")
    lc1, lc2, lc3, lc4 = st.columns(4)
    with lc1:
        st.markdown("<div class='ref-heading'>Pastorini et al. (2010)</div>"
                     "<div class='ref-text'>Genetic analysis confirmed east–west sub-population structure "
                     "partly driven by the Mahaweli River. Kaudulla elephants belong to the eastern dry-zone "
                     "meta-population.</div>", unsafe_allow_html=True)
    with lc2:
        st.markdown("<div class='ref-heading'>Leimgruber et al. (2008)</div>"
                     "<div class='ref-text'>Longitude displacement of >0.05° per day indicates long-range "
                     "foraging excursions beyond the core Kaudulla–Minneriya protected area.</div>",
                     unsafe_allow_html=True)
    with lc3:
        st.markdown("<div class='ref-heading'>HEC Hotspot — Eastern Boundary</div>"
                     "<div class='ref-text'>Longitudes >80.94°E place elephants near the Giritale–Hingurakgoda "
                     "road and paddy fields. Fence lines run along ~80.95°E on the eastern park edge.</div>",
                     unsafe_allow_html=True)
    with lc4:
        st.markdown("<div class='ref-heading'>Geographic Reference</div>"
                     "<div class='ref-text'>Longitude 80.87°–80.96°E (WGS84). Kaudulla tank central axis "
                     "≈80.89°E. Highway A11 crosses the corridor near 80.93°E.</div>", unsafe_allow_html=True)
    card_close()


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — BOTH COORDINATES
# ══════════════════════════════════════════════════════════════════════════
def render_both_tab():
    card_open(
        "📍 Latitude & Longitude vs Time (Overlaid, Dual Y-Axis)",
        "Latitude (solid lines, circles, left axis) and longitude (dotted lines, triangles, right axis) are "
        "plotted on the same chart per elephant, using matching colours. Correlated dips in latitude with "
        "rising longitude typically indicate movement toward agricultural areas on the eastern boundary.",
    )
    if agg_df.empty:
        st.info("No data for the selected filters.")
    else:
        st.plotly_chart(make_dual_axis_plot(agg_df), use_container_width=True, key="both_plot")
    card_close()

    card_open("🗺️ Live Position — Synced with Lat/Lon Chart")
    render_time_scrubber_map(agg_df, "both")
    card_close()

    col1, col2 = st.columns([5, 7])
    with col1:
        card_open(
            "🗺️ Reference Map — Kaudulla Tank & Park Boundary",
            "Approximate park boundary (dashed) and the Kaudulla Tank reference point used throughout the "
            "Latitude/Longitude tabs.",
        )
        render_reference_map()
        card_close()
    with col2:
        card_open(
            "📍 Key Coordinates",
            "Each numbered badge below matches the same badge on the reference map — "
            "● teal numbers are point features, ● orange numbers are boundary lines.",
        )
        ref_rows = [
            ("1", "core", "Kaudulla Tank (core reference)", "8.140°N", "80.895°E",
             "Dry-season water source; latitudinal reference line on the Lat/Lon tabs"),
            ("2", "core", "Park entrance / safari zone", "8.111°N", "80.886°E",
             "Southwestern edge of range; low elephant density"),
            ("3", "core", "Kaudulla Wewa (mapped reservoir)", "8.168°N", "80.926°E",
             "Northeastern shoreline; frequent gathering point in dry months"),
            ("4", "boundary", "Southern park boundary", "8.080°N", "—",
             "Southward range limit shown as a reference line on the Latitude tab"),
            ("5", "boundary", "Northern park boundary", "8.220°N", "—",
             "Northward range limit shown as a reference line on the Latitude tab"),
            ("6", "boundary", "Eastern boundary (HEC zone)", "—", "80.950°E",
             "Agricultural edge; excursions beyond this longitude flag conflict risk"),
            ("7", "boundary", "Western park boundary", "—", "80.872°E",
             "Westward range limit shown as a reference line on the Longitude tab"),
        ]
        rows_html = "".join(
            f"<tr><td><span class='ref-badge {cls}'>{num}</span></td><td><b>{name}</b></td>"
            f"<td>{lat}</td><td>{lon}</td><td style='font-size:12px;color:#555;'>{note}</td></tr>"
            for num, cls, name, lat, lon, note in ref_rows
        )
        st.markdown(
            f"<table style='width:100%;font-size:12px;color:#444;'>"
            f"<thead><tr><th>#</th><th>Location</th><th>Lat</th><th>Lon</th><th>Relevance</th></tr></thead>"
            f"<tbody>{rows_html}</tbody></table>", unsafe_allow_html=True,
        )
        card_close()

    col1, col2 = st.columns([8, 4])
    with col1:
        card_open("🐘 About The Gathering — Kaudulla National Park")
        st.markdown(
            "<div class='ref-text'><b>Kaudulla National Park</b> was gazetted in 2002 specifically to protect "
            "the elephant corridor between Minneriya and Hurulu Eco Park. Together these three parks form the "
            "'Trincomalee Elephant Triangle'.<br><br>"
            "Every year between July and October, up to <b>300–400 elephants</b> converge at the Kaudulla and "
            "Minneriya tanks in what is known as <b style='color:#2e7d32;'>'The Gathering'</b> — one of the "
            "largest aggregations of Asian elephants in the world (Fernando et al. 2008; BBC Wildlife Magazine "
            "2009).<br><br>This GPS collar dataset documents the movement of <b style='color:#2e7d32;'>14 "
            "individually identified elephants</b> from July 2024 to June 2026, capturing seasonal latitudinal "
            "shifts, boundary excursions, and corridor use.</div>", unsafe_allow_html=True,
        )
        card_close()
    with col2:
        card_open("🏛 Wildlife Department Mandate")
        st.markdown(
            "<div class='ref-text'>The Department of Wildlife Conservation of Sri Lanka (DWC), under the "
            "Ministry of Environment, administers Kaudulla under the Fauna and Flora Protection Ordinance "
            "(FFPO).<br><br>The GPS collar programme contributes to:<br>"
            "• Human–Elephant Conflict (HEC) early warning<br>• Corridor integrity assessment<br>"
            "• Population monitoring<br><br><a href='https://wildlife.gov.lk' target='_blank'>wildlife.gov.lk</a>"
            "</div>", unsafe_allow_html=True,
        )
        card_close()


@st.cache_resource(show_spinner=False)
def _build_reference_map():
    m = folium.Map(location=[8.15, 80.905], zoom_start=12, tiles="OpenStreetMap", prefer_canvas=True)
    b = du.PARK_BOUNDARY
    folium.Polygon(
        [(b["lat_min"], b["lon_min"]), (b["lat_min"], b["lon_max"]),
         (b["lat_max"], b["lon_max"]), (b["lat_max"], b["lon_min"])],
        color="#2e7d32", weight=1, dash_array="6,4", fill=True, fill_color="#2e7d32", fill_opacity=0.06,
        tooltip="Kaudulla National Park — approximate boundary",
    ).add_to(m)

    edges = [
        ("4", "Southern park boundary", [(b["lat_min"], b["lon_min"]), (b["lat_min"], b["lon_max"])], (b["lat_min"], 80.911)),
        ("5", "Northern park boundary", [(b["lat_max"], b["lon_min"]), (b["lat_max"], b["lon_max"])], (b["lat_max"], 80.911)),
        ("6", "Eastern boundary (HEC zone)", [(b["lat_min"], b["lon_max"]), (b["lat_max"], b["lon_max"])], (8.150, b["lon_max"])),
        ("7", "Western park boundary", [(b["lat_min"], b["lon_min"]), (b["lat_max"], b["lon_min"])], (8.150, b["lon_min"])),
    ]
    for num, name, coords, mid in edges:
        folium.PolyLine(coords, color="#c1440e", weight=4, opacity=0.85, dash_array="8,5", tooltip=f"#{num} — {name}").add_to(m)
        folium.map.Marker(
            mid, icon=folium.DivIcon(html=f"<div style='background:#c1440e;color:white;border-radius:50%;width:20px;height:20px;"
                                            f"display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;'>{num}</div>"),
        ).add_to(m)

    points = [
        ("1", "Kaudulla Tank (core reference)", 8.140, 80.895, "Dry-season water source"),
        ("2", "Park entrance / safari zone", 8.111, 80.886, "Southwestern edge of range"),
        ("3", "Kaudulla Wewa (mapped reservoir)", 8.168, 80.926, "Northeastern shoreline"),
    ]
    for num, name, lat, lon, note in points:
        folium.CircleMarker(
            [lat, lon], radius=9, color="#0f766e", weight=2, fill=True, fill_color="#4fc3f7", fill_opacity=0.9,
            popup=f"<b>#{num} — {name}</b><br>{lat:.3f}°N, {lon:.3f}°E<br>{note}",
        ).add_to(m)
        folium.map.Marker(
            [lat, lon], icon=folium.DivIcon(html=f"<div style='background:#0f766e;color:white;border-radius:50%;width:20px;height:20px;"
                                                   f"display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;'>{num}</div>"),
        ).add_to(m)
    return m


def render_reference_map():
    m = _build_reference_map()
    st_folium(m, height=420, use_container_width=True, key="ref_map", returned_objects=[])


# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — HEAT MAPS
# ══════════════════════════════════════════════════════════════════════════
def render_heat_tab():
    df = filtered_df
    if df.empty:
        st.info("No data for the selected filters.")
        return
    d = df.copy()
    d["ym"] = d["datetime_sl"].dt.strftime("%Y-%m")
    dt_naive = d["datetime_sl"].dt.tz_localize(None)
    months_seq = pd.period_range(dt_naive.min().to_period("M"), dt_naive.max().to_period("M"), freq="M")
    months_seq = [str(m) for m in months_seq]
    names_seq = sorted(d["name"].unique())

    agg = d.groupby(["name", "ym"]).agg(mlon=("lon", "mean"), n=("lon", "size")).reset_index()
    mat_lon = pd.DataFrame(np.nan, index=names_seq, columns=months_seq)
    mat_n = pd.DataFrame(np.nan, index=names_seq, columns=months_seq)
    for _, r in agg.iterrows():
        mat_lon.loc[r["name"], r["ym"]] = r["mlon"]
        mat_n.loc[r["name"], r["ym"]] = r["n"]

    card_open(
        "🌐 Average Longitude by Month — Position Heat Map (blank = no data)",
        "Cell colour = mean longitude of GPS fixes for that elephant in that month (warmer = further east, "
        "toward the agricultural boundary; cooler = further west, toward the tank). Blank cells mean no GPS "
        "fixes were recorded that month within the current filters.",
    )
    fig1 = go.Figure(go.Heatmap(
        z=mat_lon.values, x=mat_lon.columns, y=mat_lon.index,
        colorscale=[[0, "#f1faee"], [0.33, "#2a9d8f"], [0.66, "#e9c46a"], [1, "#e63946"]],
        hoverongaps=False, colorbar=dict(title="Mean<br>Lon (°E)"),
        hovertemplate="%{y}<br>%{x}<br>Mean lon %{z:.4f}°E<extra></extra>",
    ))
    fig1.update_layout(paper_bgcolor="#fff", plot_bgcolor="#fff", font=dict(color="#333", family="Segoe UI"),
                        xaxis=dict(title="Month", tickangle=-45), yaxis=dict(title="", autorange="reversed"),
                        margin=dict(t=20, b=70, l=110, r=20), height=440)
    st.plotly_chart(fig1, use_container_width=True, key="heat_lon")
    card_close()

    card_open(
        "📊 Data Coverage by Month — GPS Fix Count Heat Map",
        "Cell colour = number of GPS fixes recorded for that elephant in that month. Darker/blank cells flag "
        "months with sparse or missing collar data.",
    )
    fig2 = go.Figure(go.Heatmap(
        z=mat_n.values, x=mat_n.columns, y=mat_n.index,
        colorscale=[[0, "#f1faee"], [0.5, "#ff9f1c"], [1, "#e63946"]],
        hoverongaps=False, colorbar=dict(title="GPS<br>Fixes"),
        hovertemplate="%{y}<br>%{x}<br>%{z} fixes<extra></extra>",
    ))
    fig2.update_layout(paper_bgcolor="#fff", plot_bgcolor="#fff", font=dict(color="#333", family="Segoe UI"),
                        xaxis=dict(title="Month", tickangle=-45), yaxis=dict(title="", autorange="reversed"),
                        margin=dict(t=20, b=70, l=110, r=20), height=440)
    st.plotly_chart(fig2, use_container_width=True, key="heat_n")
    card_close()


# ══════════════════════════════════════════════════════════════════════════
# TAB 5 — ELEPHANT TRACKING
# ══════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Building tracking map…")
def _build_tracking_map():
    m = base_folium_map()
    for el in ALL_NAMES:
        sub = elephants_df[elephants_df["name"] == el]
        clr = tracking_colors[el]
        # Downsample dense tracks so the map stays fast to build and render —
        # ~150 points/elephant is plenty to see the shape of each elephant's
        # range; a full every-fix popup for 16k+ points made the map's HTML
        # payload balloon to ~9MB and took several seconds just to serialize.
        step = max(1, len(sub) // 150)
        for _, r in sub.iloc[::step].iterrows():
            folium.CircleMarker(
                [r["lat"], r["lon"]], radius=3, color=clr, fill=True, fill_color=clr, fill_opacity=0.8, weight=0,
                tooltip=f"{el} — {r['datetime']:%d %b %Y}",
            ).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    _add_categorical_legend(m, "Elephant Name", tracking_colors)
    return m


@st.cache_resource(show_spinner="Building tracking scatter plot…")
def _build_tracking_scatter_fig():
    fig, ax = plt.subplots(figsize=(11, 7))
    for el in ALL_NAMES:
        sub = elephants_df[elephants_df["name"] == el]
        ax.scatter(sub["lon"], sub["lat"], s=4, alpha=0.5, color=tracking_colors[el], label=el)
    ax.set_title("Tracking Data", fontsize=18, fontweight="bold")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.legend(title="Name", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, markerscale=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


@st.cache_resource(show_spinner="Building monthly playback animation…")
def _build_tracking_playback_fig():
    """One Plotly figure with a built-in Play/Pause button + slider that
    animates through months entirely in the browser — no Streamlit rerun
    per frame. Built manually with make_subplots + explicit go.Frame per
    month (rather than px.scatter's animation_frame + facet_col, which
    doesn't reliably map frames to facet subplots and renders blank)."""
    d = elephants_df.copy()
    d["date_month"] = d["datetime_sl"].dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()

    months = sorted(d["date_month"].unique())
    month_labels = [pd.Timestamp(m).strftime("%Y %b") for m in months]

    xmin, xmax = elephants_df["lon"].min(), elephants_df["lon"].max()
    ymin, ymax = elephants_df["lat"].min(), elephants_df["lat"].max()
    x_pad = (xmax - xmin) * 0.05 or 0.01
    y_pad = (ymax - ymin) * 0.05 or 0.01

    n_col = min(7, len(ALL_NAMES))
    n_row = int(np.ceil(len(ALL_NAMES) / n_col))
    sex_colors = {"Male": "darkblue", "Female": "darkred"}
    sexes = ["Male", "Female"]

    def _xy(month_ts, el, sex):
        sub = d[(d["date_month"] == month_ts) & (d["name"] == el) & (d["sex"] == sex)]
        return sub["lon"].tolist(), sub["lat"].tolist()

    fig = make_subplots(
        rows=n_row, cols=n_col, subplot_titles=ALL_NAMES,
        horizontal_spacing=0.02, vertical_spacing=0.10,
        shared_xaxes=True, shared_yaxes=True,
    )

    # one fixed trace per (elephant, sex) pair, added in a stable order that
    # every frame below will follow exactly — this is what keeps facets and
    # frames correctly lined up.
    trace_order = []
    init_month = months[-1]
    for i, el in enumerate(ALL_NAMES):
        row, col = i // n_col + 1, i % n_col + 1
        for sex in sexes:
            x, y = _xy(init_month, el, sex)
            fig.add_trace(
                go.Scatter(
                    x=x, y=y, mode="markers",
                    marker=dict(size=6, color=sex_colors[sex], opacity=0.85),
                    name=sex, legendgroup=sex, showlegend=(i == 0),
                    hoverinfo="skip",
                ),
                row=row, col=col,
            )
            trace_order.append((el, sex))

    fig.update_xaxes(range=[xmin - x_pad, xmax + x_pad], tickfont=dict(size=7), ticksuffix="°E", tickangle=-45)
    fig.update_yaxes(range=[ymin - y_pad, ymax + y_pad], tickfont=dict(size=7), ticksuffix="°N")
    for ann in fig.layout.annotations:
        ann.update(font=dict(size=11, color="#1e293b"))
    fig.add_annotation(text="Longitude", x=0.5, y=-0.06, xref="paper", yref="paper",
                        showarrow=False, font=dict(size=12, color="#1e293b"))
    fig.add_annotation(text="Latitude", x=-0.045, y=0.5, xref="paper", yref="paper",
                        showarrow=False, textangle=-90, font=dict(size=12, color="#1e293b"))

    frames = []
    for m, label in zip(months, month_labels):
        frame_data = []
        for el, sex in trace_order:
            x, y = _xy(m, el, sex)
            frame_data.append(go.Scatter(x=x, y=y))
        frames.append(go.Frame(data=frame_data, name=label))
    fig.frames = frames

    slider_steps = [
        dict(
            method="animate",
            args=[[label], dict(mode="immediate", frame=dict(duration=900, redraw=True),
                                 transition=dict(duration=300))],
            label=label,
        )
        for label in month_labels
    ]

    fig.update_layout(
        height=820,
        margin=dict(t=70, b=110, l=60, r=10),
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        legend=dict(title_text="Sex:", orientation="h", x=0.5, xanchor="center", y=-0.09,
                    font=dict(size=12)),
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.0, y=1.10, xanchor="left",
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(frame=dict(duration=900, redraw=True),
                                       transition=dict(duration=300), fromcurrent=True)]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                         mode="immediate", transition=dict(duration=0))]),
            ],
        )],
        sliders=[dict(
            active=len(months) - 1, steps=slider_steps, x=0.06, len=0.92, y=-0.02,
            currentvalue=dict(prefix="Month: ", font=dict(size=13, color="#1e293b")),
        )],
    )
    return fig


def render_tracking_playback():
    st.plotly_chart(_build_tracking_playback_fig(), use_container_width=True, key="tracking_playback")


# ══════════════════════════════════════════════════════════════════════════
# TAB — DONA & RECOLLARED  (new: ported from combined_dashboard_app's
# dona_recollared module — highlights Dona and the recollared female
# against everyone else)
# ══════════════════════════════════════════════════════════════════════════
DR_CATEGORY_COLORS = {
    "Recollared Female": "#E31A1C",
    "Dona": "#984EA3",
    "Other Elephants": "#4DAF4A",
}


@st.cache_resource(show_spinner="Building Dona & Recollared map…")
def _build_dona_recollared_map():
    d = elephants_df.copy()
    name_lower = d["name"].str.lower()
    d["display_category"] = np.select(
        [name_lower == "recollared female", name_lower == "dona"],
        ["Recollared Female", "Dona"],
        default="Other Elephants",
    )

    m = base_folium_map()
    # Downsample the "Other Elephants" bucket (large, lower priority) so the
    # map stays responsive; Dona and the recollared female (the actual point
    # of this map) are kept at full resolution.
    for cat, group in d.groupby("display_category"):
        clr = DR_CATEGORY_COLORS[cat]
        sub = group if cat != "Other Elephants" else group.iloc[::max(1, len(group) // 400)]
        for _, r in sub.iterrows():
            folium.CircleMarker(
                [r["lat"], r["lon"]], radius=4, color="white", weight=0.5,
                fill=True, fill_color=clr, fill_opacity=0.85,
                popup=f"<b>{r['name']}</b><br>{r['datetime_sl']:%d %b %Y @ %H:%M}<br>"
                      f"<span style='color:#64748b'>Category:</span> <b>{cat}</b>",
                tooltip=r["name"],
            ).add_to(m)

    folium.LayerControl(collapsed=False, position="bottomright").add_to(m)
    _add_categorical_legend(m, "Elephant Identity", DR_CATEGORY_COLORS)
    return m


def render_dona_recollared_tab():
    st.markdown("<div class='section-title'>🐘 Dona & Recollared — Individual Elephant Tracking</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-title'>Map — GPS Points by Individual</div>"
        "<p style='text-align:center;color:#64748b;font-size:13px;margin-top:-6px;'>"
        "Standard OSM terrain (green parks, blue water, yellow roads).</p>",
        unsafe_allow_html=True,
    )
    st_folium(_build_dona_recollared_map(), height=750, use_container_width=True, key="dona_recollared_map", returned_objects=[])


# ══════════════════════════════════════════════════════════════════════════
# TAB — DENSITY & CLIMATE  (new: ported from combined_dashboard_app's
# density_climate module)
# ══════════════════════════════════════════════════════════════════════════
DC_FOCUS_YEARS = {"2024", "2025", "2026"}


@st.cache_resource(show_spinner=False)
def _dc_hourly_climate():
    hourly = du.load_climate_hourly()
    return {
        "temp": du.build_climate_ym_category(hourly, "T2M", du.DC_TEMP_BREAKS, du.DC_TEMP_LABELS),
        "rainfall": du.build_climate_ym_category(hourly, "PRECTOTCORR", du.DC_RAIN_BREAKS, du.DC_RAIN_LABELS),
    }


@st.cache_data(show_spinner=False)
def _dc_hex_polygons(lons, lats, n_bins):
    """True hexagonal bins over the points (projected to UTM 44N for
    metrically-correct hexagons), returned as (lon/lat ring, count) pairs.
    Reuses matplotlib's hexbin() to get real hexagon geometry — its path
    vertices + offsets are already in the same projected data units, so
    vertices + offset gives the true hexagon corners directly.
    Cached on (lons, lats, n_bins): this only depends on the elephant
    selection, not on the 'Play path' slider, so dragging that slider no
    longer re-runs matplotlib's hexbin() + UTM reprojection on every tick."""
    xs, ys = du.to_utm(lons, lats)
    xs, ys = np.asarray(xs), np.asarray(ys)
    fig, ax = plt.subplots()
    hb = ax.hexbin(xs, ys, gridsize=n_bins, mincnt=1)
    base_verts = hb.get_paths()[0].vertices
    offsets = hb.get_offsets()
    counts = hb.get_array()
    plt.close(fig)

    polys = []
    for (cx, cy), n in zip(offsets, counts):
        verts_utm = base_verts + [cx, cy]
        lon_v, lat_v = du.from_utm(verts_utm[:, 0], verts_utm[:, 1])
        polys.append((list(zip(lat_v, lon_v)), int(n)))
    return polys


def _dc_add_travel_path(m, path_df, steps):
    """Numbered polyline connecting centroids in travel order, up to `steps`
    segments — drawn identically on both maps."""
    if path_df.empty or steps <= 0:
        return
    steps = min(steps, len(path_df) - 1)
    for i in range(steps):
        p1, p2 = path_df.iloc[i], path_df.iloc[i + 1]
        seg_color = p2["fill_color"] if pd.notna(p2["fill_color"]) else "#1e293b"
        folium.PolyLine(
            [(p1["lat"], p1["lon"]), (p2["lat"], p2["lon"])],
            color=seg_color, weight=3, opacity=0.95,
        ).add_to(m)
        mid_lat, mid_lon = (p1["lat"] + p2["lat"]) / 2, (p1["lon"] + p2["lon"]) / 2
        folium.Marker(
            [mid_lat, mid_lon],
            icon=folium.DivIcon(html=(
                f"<div style='background:#1e293b; color:white; border-radius:50%; padding:1px 6px; "
                f"font-weight:bold; font-size:11px; box-shadow:0 0 2px #fff; display:inline-block; "
                f"transform:translate(-50%,-50%);'>{i + 1}</div>"
            )),
        ).add_to(m)


def render_density_climate_tab():
    st.markdown("<div class='section-title'>📍 Movement Density, Path & Climate</div>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center;color:#64748b;font-size:13px;margin-top:-8px;'>"
        "Hexbin fix density · categorical monthly temperature or rainfall with animated, numbered travel path</p>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([3, 9])
    with c1:
        card_open()
        dpt_elephants = st.multiselect("Select elephant(s):", ALL_NAMES, default=ALL_NAMES, key="dpt_elephants")
        dpt_metric = st.radio("Climate variable (map 2 + chart):", ["Temperature", "Rainfall"], key="dpt_metric")
        dpt_period = st.radio(
            "Time of day:", ["Day (06:00-18:00)", "Night (18:00-06:00)"], key="dpt_period",
        )
        period_key = "Day" if dpt_period.startswith("Day") else "Night"
        card_close()

    d = elephants_df[elephants_df["year"].isin(DC_FOCUS_YEARS)].copy()
    if dpt_elephants:
        d = d[d["name"].isin(dpt_elephants)]
    d["year_month_dt"] = pd.to_datetime(d["year_month"] + "-01")

    centroid = (
        d.groupby(["year_month", "year_month_dt"])
        .agg(lon=("lon", "mean"), lat=("lat", "mean"), n=("lon", "size"))
        .reset_index()
        .sort_values("year_month_dt")
    )
    centroid["ym_label"] = centroid["year_month_dt"].dt.strftime("%b %Y")

    metric_key = "temp" if dpt_metric == "Temperature" else "rainfall"
    metric_colors = du.DC_TEMP_COLORS if metric_key == "temp" else du.DC_RAIN_COLORS
    metric_unit = "°C" if metric_key == "temp" else "mm/day"
    clim_table = _dc_hourly_climate()[metric_key]
    clim_table = clim_table[clim_table["period"] == period_key]

    path_df = centroid.merge(
        clim_table.rename(columns={"year_month": "year_month_dt"}), on="year_month_dt", how="left",
    )
    path_df["fill_color"] = path_df["category"].astype(str).map(metric_colors).fillna("#999999")

    with c1:
        card_open("Selection Summary")
        st.markdown(
            f"**Elephant(s):** {'All Elephants' if len(dpt_elephants) == len(ALL_NAMES) else ', '.join(dpt_elephants) or 'None'}  \n"
            f"**Total GPS fixes:** {len(d)}  \n"
            f"**Year-months with data:** {len(centroid)}  \n"
            + (f"**Range:** {centroid['ym_label'].iloc[0]} to {centroid['ym_label'].iloc[-1]}  \n" if len(centroid) else "")
            + f"**Climate variable shown:** {dpt_metric} ({period_key})"
        )
        card_close()

    n_steps = max(len(centroid) - 1, 0)
    steps = st.slider(
        "Play path (connects centroids in travel order):", 0, max(n_steps, 1), value=n_steps,
        key="dpt_connect_step",
    )
    if n_steps > 0:
        step_labels = centroid["ym_label"].tolist()
        st.caption(f"Step {steps} of {n_steps} — {step_labels[min(steps, len(step_labels) - 1)] if step_labels else ''}")

    with c2:
        m1c, m2c = st.columns(2)
        with m1c:
            st.markdown("<div class='sub-title' style='font-size:15px;'>Hexbin density + monthly centroids + numbered travel path</div>", unsafe_allow_html=True)
            if d.empty:
                st.info("No GPS fixes for this elephant selection.")
            else:
                m1 = base_folium_map()
                n_bins = du.suitable_hex_bins(len(d))
                hexes = _dc_hex_polygons(d["lon"].values, d["lat"].values, n_bins)
                counts = [c for _, c in hexes]
                if counts:
                    cmap = plt.get_cmap("magma")
                    norm = plt.Normalize(vmin=min(counts), vmax=max(counts))
                    for ring, n in hexes:
                        color = mcolors.rgb2hex(cmap(norm(n)))
                        folium.Polygon(
                            ring, color="white", weight=0.6, fill=True, fill_color=color, fill_opacity=0.75,
                            tooltip=f"Fixes: {n}",
                        ).add_to(m1)
                for _, r in centroid.iterrows():
                    folium.CircleMarker(
                        [r["lat"], r["lon"]], radius=6, color="black", weight=1.5,
                        fill=True, fill_color="#00E5FF", fill_opacity=1,
                        tooltip=f"{r['ym_label']} centroid (n={r['n']})",
                    ).add_to(m1)
                _dc_add_travel_path(m1, path_df, steps)
                folium.LayerControl(collapsed=False).add_to(m1)
                st_folium(m1, height=640, use_container_width=True, key="dpt_density_map", returned_objects=[])
        with m2c:
            st.markdown(
                f"<div class='sub-title' style='font-size:15px;'>Monthly {dpt_metric.lower()} (categorical) + numbered travel path</div>",
                unsafe_allow_html=True,
            )
            if path_df.empty:
                st.info("No data available.")
            else:
                m2 = base_folium_map()
                for _, r in path_df.iterrows():
                    val_txt = "no data" if pd.isna(r.get("avg_value")) else f"{r['avg_value']:.1f} {metric_unit} - {r['category']}"
                    folium.CircleMarker(
                        [r["lat"], r["lon"]], radius=9, color="black", weight=1,
                        fill=True, fill_color=r["fill_color"], fill_opacity=0.9,
                        tooltip=f"{r['ym_label']} ({period_key}): {val_txt}",
                    ).add_to(m2)
                _dc_add_travel_path(m2, path_df, steps)
                _add_categorical_legend(m2, f"{dpt_metric} class ({period_key})", metric_colors)
                folium.LayerControl(collapsed=False).add_to(m2)
                st_folium(m2, height=640, use_container_width=True, key="dpt_temp_map", returned_objects=[])


# ══════════════════════════════════════════════════════════════════════════
# TAB — DAY / NIGHT  (new: ported from combined_dashboard_app's
# day_night module)
# ══════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Preparing Day/Night data…")
def _dn_data():
    hourly = du.load_climate_hourly()
    return {
        "gps": du.build_dn_gps(elephants_df, hourly),
        "daily_temp": du.build_dn_daily_temp(hourly),
        "daily_rain": du.build_dn_daily_rain(hourly),
    }


def _dn_add_points(m, df, col_var, col_map, legend_title):
    if df.empty:
        return
    # downsample very large point sets so the map stays responsive
    sub = df if len(df) <= 4000 else df.iloc[:: max(1, len(df) // 4000)]
    for _, r in sub.iterrows():
        clr = col_map.get(str(r[col_var]), "#94a3b8")
        folium.CircleMarker(
            [r["lat"], r["lon"]], radius=4, color="transparent", weight=0,
            fill=True, fill_color=clr, fill_opacity=0.70,
            popup=f"<b>{r['name']}</b><br>{r['dt_round']:%d %b %Y  %H:%M}<br>"
                  f"<span style='color:#64748b'>{legend_title}:</span> <b>{r[col_var]}</b>",
        ).add_to(m)
    _add_categorical_legend(m, legend_title, col_map)


def render_day_night_tab():
    st.markdown("<div class='section-title'>🌗 Day / Night & Climate Periods</div>", unsafe_allow_html=True)

    data = _dn_data()
    gps = data["gps"]

    c1, c2 = st.columns([3, 9])
    with c1:
        card_open()
        yr_opts = ["All"] + sorted(gps["year"].unique().tolist())
        dn_year = st.selectbox("Year:", yr_opts, key="dn_year")
        dn_eles = st.multiselect("Elephant(s):", ALL_NAMES, default=[], key="dn_eles",
                                  placeholder="All elephants (leave empty)")
        card_close()

    df = gps if dn_year == "All" else gps[gps["year"] == dn_year]
    if dn_eles:
        df = df[df["name"].isin(dn_eles)]

    with c2:
        st.markdown("<div class='sub-title' style='font-size:15px;'>Map 1 — GPS Points: Day vs Night</div>", unsafe_allow_html=True)
        st.caption("All GPS fixes plotted simultaneously · Red = Day (06:00–18:00) · Blue = Night (18:00–06:00)")
        m_dn = base_folium_map()
        _dn_add_points(m_dn, df, "day_night", {"Day": "#dc2626", "Night": "#2563eb"}, "Time of Day")
        st_folium(m_dn, height=460, use_container_width=True, key="dn_map1", returned_objects=[])

        st.markdown("<div class='sub-title' style='font-size:15px;'>Map 2 — Climate Period</div>", unsafe_allow_html=True)
        climate_tab_map = st.radio(
            "Map 2 mode:",
            ["Temperature Period (Hot/Cool)", "Rainfall Period (Heavy/Low)", "Hourly Rain Filter"],
            horizontal=True, key="dn_climate_tab_map",
        )

        if climate_tab_map.startswith("Temperature"):
            map2_mode = st.radio("", ["2 colours  (Hot / Cool)", "5 periods  (individual shades)"],
                                  horizontal=True, key="dn_map2_mode")
            st.caption("All GPS fixes · Red = Hot period · Blue = Cool period" if map2_mode.startswith("2")
                       else "All GPS fixes · 5 individually-shaded hot/cool sub-periods")
            m_hc = base_folium_map()
            if map2_mode.startswith("2"):
                _dn_add_points(m_hc, df, "hot_cool", {"Hot": "#dc2626", "Cool": "#2563eb"}, "Temperature Period")
            else:
                _dn_add_points(m_hc, df, "period", du.DN_PERIOD_5_COLS, "Temperature Period")
            st_folium(m_hc, height=460, use_container_width=True, key="dn_map2", returned_objects=[])

        elif climate_tab_map.startswith("Rainfall"):
            st.caption("All GPS fixes · Red = Heavy rain period · Green = Low rain period")
            m_hc = base_folium_map()
            _dn_add_points(m_hc, df, "rain", du.DN_RAIN_2_COLS, "Rainfall Period")
            st_folium(m_hc, height=460, use_container_width=True, key="dn_map2", returned_objects=[])

        else:  # Hourly Rain Filter
            st.caption("Filters the map to ONLY show locations where the hourly rainfall exceeded your selected threshold(s).")
            thresh_opts = {
                "Light Rain (> 0 to 2 mm/hr)": "light", "Moderate Rain (2 to 5 mm/hr)": "mod",
                "Heavy Rain (5 to 10 mm/hr)": "heavy", "Extreme Rain (10 to 20 mm/hr)": "extreme",
                "Cloudburst (> 20 mm/hr)": "ultra",
            }
            picked_labels = st.multiselect("Rainfall intensity:", list(thresh_opts.keys()),
                                            default=["Heavy Rain (5 to 10 mm/hr)"], key="dn_rain_thresh")
            picked = [thresh_opts[l] for l in picked_labels]

            df_map2 = df[(df["rain_mm_hr"].notna()) & (df["rain_mm_hr"] > 0)].copy()
            conditions = [df_map2["rain_mm_hr"] > 20, df_map2["rain_mm_hr"] > 10,
                          df_map2["rain_mm_hr"] > 5, df_map2["rain_mm_hr"] > 2]
            df_map2["temp_cat"] = np.select(conditions, ["ultra", "extreme", "heavy", "mod"], default="light")
            df_map2 = df_map2[df_map2["temp_cat"].isin(picked)]
            rain_cat_map = {"ultra": "> 20 mm/hr", "extreme": "10 - 20 mm/hr", "heavy": "5 - 10 mm/hr",
                             "mod": "2 - 5 mm/hr", "light": "> 0 - 2 mm/hr"}
            df_map2["rain_cat"] = df_map2["temp_cat"].map(rain_cat_map)

            m_hc = base_folium_map()
            _dn_add_points(m_hc, df_map2, "rain_cat", du.DN_HOURLY_RAIN_COLS, "Hourly Rainfall")
            st_folium(m_hc, height=460, use_container_width=True, key="dn_map2", returned_objects=[])

    if climate_tab_map.startswith("Temperature"):
        st.markdown("<div class='sub-title'>Temperature Time Series with Hot / Cool Period Boundaries</div>", unsafe_allow_html=True)
        st.caption("Daily mean temperature (NASA POWER) · Orange = Hot · Blue = Cool · Dashed lines = transition dates")
        daily = data["daily_temp"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[pd.Timestamp("2024-06-01")], y=[None], mode="lines",
                                  line=dict(color="rgba(253,186,116,0.7)", width=8), name="Hot period"))
        fig.add_trace(go.Scatter(x=[pd.Timestamp("2024-06-01")], y=[None], mode="lines",
                                  line=dict(color="rgba(147,197,253,0.7)", width=8), name="Cool period"))
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["T_avg"], mode="lines",
                                  line=dict(color="#1e293b", width=1.2), name="Mean daily temp (°C)",
                                  hovertemplate="<b>%{x|%d %b %Y}</b><br>Mean temp: <b>%{y:.2f} °C</b><extra></extra>"))
        for b in du.DN_HC_BANDS:
            fig.add_vrect(x0=b["x0"], x1=b["x1"], fillcolor=b["fill"], line_width=0, layer="below")
        arrow_labels = ["→ Cool", "→ Hot", "→ Cool", "→ Hot"]
        for d_, lbl in zip(du.DN_TRANSITIONS, arrow_labels):
            fig.add_vline(x=pd.Timestamp(d_), line=dict(color="#94a3b8", width=1.2, dash="dash"))
            fig.add_annotation(x=d_, y=1, yref="paper", text=lbl, showarrow=False,
                                font=dict(size=9, color="#64748b"), xanchor="left", yanchor="top", xshift=4)
        fig.update_layout(
            plot_bgcolor="#ffffff", paper_bgcolor="#f8fafc",
            font=dict(color="#1e293b", size=11),
            xaxis=dict(title="", type="date", gridcolor="#e2e8f0", tickfont=dict(size=10, color="#64748b")),
            yaxis=dict(title="Mean daily temperature (°C)", gridcolor="#e2e8f0",
                       tickfont=dict(size=10, color="#64748b"), range=[21, 30.5]),
            legend=dict(orientation="h", x=0.5, xanchor="center", y=1.12, bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=30, l=60, r=20, b=30), hovermode="x unified", height=320,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown("<div class='sub-title'>Rainfall Time Series — Heavy vs Low Rain Periods</div>", unsafe_allow_html=True)
        st.caption("Mean daily rainfall rate (NASA POWER) · Blue shading = Heavy rain · Orange shading = Low rain")
        daily_r = data["daily_rain"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[pd.Timestamp("2024-06-01")], y=[None], mode="lines",
                                  line=dict(color="rgba(147,197,253,0.7)", width=8), name="Heavy rain period"))
        fig.add_trace(go.Scatter(x=[pd.Timestamp("2024-06-01")], y=[None], mode="lines",
                                  line=dict(color="rgba(254,215,170,0.7)", width=8), name="Low rain period"))
        fig.add_trace(go.Bar(x=daily_r["date"], y=daily_r["P_avg"], marker=dict(color="#1d4ed8", opacity=0.75),
                              name="Mean daily rainfall (mm/day)",
                              hovertemplate="<b>%{x|%d %b %Y}</b><br>Rainfall: <b>%{y:.2f} mm/day</b><extra></extra>"))
        for b in du.DN_RAIN_BANDS:
            fig.add_vrect(x0=b["x0"], x1=b["x1"], fillcolor=b["fill"], line_width=0, layer="below")
        fig.update_layout(
            plot_bgcolor="#ffffff", paper_bgcolor="#f8fafc",
            font=dict(color="#1e293b", size=11),
            xaxis=dict(title="", type="date", gridcolor="#e2e8f0", tickfont=dict(size=10, color="#64748b")),
            yaxis=dict(title="Mean daily rainfall rate (mm/day)", gridcolor="#e2e8f0",
                       tickfont=dict(size=10, color="#64748b"), range=[0, 165]),
            legend=dict(orientation="h", x=0.5, xanchor="center", y=1.12, bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=30, l=60, r=20, b=30), hovermode="x unified", height=320, bargap=0,
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB — VEGETATION TRACKING  (new: ported from combined_dashboard_app's
# vegetation_tracking module)
# ══════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Preparing vegetation & hotspot data…")
def _vt_data():
    all_hs, each_hs = du.build_vt_hotspots(elephants_df)
    return {
        "monthly_range": du.build_vt_monthly_range(elephants_df),
        "hs_all": all_hs,
        "hs_each": each_hs,
        "fix_counts": du.build_vt_fix_counts(elephants_df),
        "ele_colors": du.build_elephant_palette(ALL_NAMES, du.VT_ELE_PALETTE),
    }


def _vt_light_layout(fig, **kwargs):
    xaxis = dict(gridcolor="#e2e8f0", tickfont=dict(color="#64748b", size=10))
    yaxis = dict(gridcolor="#e2e8f0", tickfont=dict(color="#64748b", size=10))
    legend = dict(bgcolor="#ffffff", bordercolor="#e2e8f0", borderwidth=1, font=dict(size=10, color="#1e293b"))
    margin = dict(t=10, l=50, r=10, b=40)

    xaxis.update(kwargs.pop("xaxis", {}) or {})
    yaxis.update(kwargs.pop("yaxis", {}) or {})
    legend.update(kwargs.pop("legend", {}) or {})
    margin.update(kwargs.pop("margin", {}) or {})

    fig.update_layout(
        plot_bgcolor="#ffffff", paper_bgcolor="#f8fafc",
        font=dict(color="#1e293b", size=11),
        xaxis=xaxis, yaxis=yaxis, legend=legend, margin=margin,
        **kwargs,
    )
    return fig


def _vt_add_directional_arrow(m, from_lat, from_lon, to_lat, to_lon, color):
    """Shrink the connecting line slightly off each dot and add a small
    chevron arrowhead at the midpoint — ports the R app's trig-based arrow."""
    dlat, dlon = to_lat - from_lat, to_lon - from_lon
    dist = np.hypot(dlat, dlon)
    if dist == 0:
        return
    shrink = min(0.0007, dist * 0.15)
    frac = shrink / dist
    lat_s, lon_s = from_lat + dlat * frac, from_lon + dlon * frac
    lat_e, lon_e = to_lat - dlat * frac, to_lon - dlon * frac
    folium.PolyLine([(lat_s, lon_s), (lat_e, lon_e)], color=color, weight=2.2, opacity=0.9).add_to(m)

    bearing = np.arctan2(dlon, dlat)
    wing_len = dist * 0.18
    angle1, angle2 = bearing + np.radians(140), bearing - np.radians(140)
    mid_lat, mid_lon = (lat_s + lat_e) / 2, (lon_s + lon_e) / 2
    w1 = (mid_lat + wing_len * np.cos(angle1), mid_lon + wing_len * np.sin(angle1))
    w2 = (mid_lat + wing_len * np.cos(angle2), mid_lon + wing_len * np.sin(angle2))
    folium.PolyLine([w1, (mid_lat, mid_lon), w2], color=color, weight=2.2, opacity=0.9).add_to(m)


def _vt_triangle_icon(color, size=26):
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}">'
        f'<polygon points="{size/2},2 {size-2},{size-2} 2,{size-2}" '
        f'fill="{color}" stroke="white" stroke-width="2"/></svg>'
    )
    b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    return folium.CustomIcon(f"data:image/svg+xml;base64,{b64}", icon_size=(size, size), icon_anchor=(size / 2, size - 2))


def render_vegetation_tab():
    st.markdown("<div class='section-title'>🌿 Kaudulla NP — Vegetation & GPS Tracking Dashboard</div>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center;color:#64748b;font-size:13px;margin-top:-8px;'>"
        "NDVI 2018-2026 · GPS collar data 2024-2026 · Hotspot analysis</p>",
        unsafe_allow_html=True,
    )
    data = _vt_data()

    # ── Row 1: veg coverage chart | GPS map by year ─────────────────────────
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        card_open("Vegetation Coverage")
        veg_type = st.selectbox("Class type:", ["Density Classes", "Health Classes"], key="vt_veg_type")
        chart_style = st.radio("Style:", ["Grouped", "Stacked"], horizontal=True, key="vt_chart_style")
        if veg_type == "Density Classes":
            raw, cols, cls = du.VT_DENSITY_RAW, du.VT_DENS_COLS, list(du.VT_DENS_COLS.keys())
        else:
            raw, cols, cls = du.VT_HEALTH_RAW, du.VT_HLTH_COLS, list(du.VT_HLTH_COLS.keys())
        df_focus = raw[raw["year"].isin(du.VT_FOCUS_YEARS)]
        fig = go.Figure()
        for cl in cls:
            fig.add_trace(go.Bar(
                x=df_focus["year"].astype(str), y=df_focus[cl], name=cl,
                marker=dict(color=cols[cl], opacity=0.9, line=dict(width=0.5, color=_hex_alpha("#000000", 0.13))),
                hovertemplate=f"<b>{cl}</b><br>Year: %{{x}}<br>Coverage: <b>%{{y:.2f}}%</b><extra></extra>",
            ))
        _vt_light_layout(fig,
            barmode="group" if chart_style == "Grouped" else "stack",
            xaxis=dict(title="Year"), yaxis=dict(title="Coverage (%)"),
            legend=dict(orientation="h", x=0, y=-0.28, bgcolor="rgba(0,0,0,0)", font=dict(size=9.5)),
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("High Density Forest < 0.4% all years — nearly absent from park.")
        card_close()

    with r1c2:
        card_open("All GPS Tracking Points")
        sel_years = st.multiselect("Years:", ["2024", "2025", "2026"], default=["2024", "2025", "2026"], key="vt_sel_years")
        sel_ele = st.multiselect("Elephants:", ALL_NAMES, default=ALL_NAMES, key="vt_sel_elephant")
        d = elephants_df[elephants_df["year"].isin(sel_years) & elephants_df["name"].isin(sel_ele or ALL_NAMES)]
        m_gps = base_folium_map()
        if len(d) > 8000:
            d = d.sample(n=8000, random_state=0)
        for yr in sorted(sel_years):
            sub = d[d["year"] == yr]
            if sub.empty:
                continue
            col = du.VT_GPS_YEAR_COLS.get(yr, "#888")
            for _, r in sub.iterrows():
                folium.CircleMarker(
                    [r["lat"], r["lon"]], radius=3, color=col, fill=True, fill_color=col,
                    weight=0, fill_opacity=0.65,
                    popup=f"<b>{r['name']}</b><br>Year: <b>{yr}</b><br>Lat: {r['lat']:.5f}  Lon: {r['lon']:.5f}<br>{r['datetime']:%Y-%m-%d %H:%M UTC}",
                ).add_to(m_gps)
        _add_categorical_legend(m_gps, "Year", {y: du.VT_GPS_YEAR_COLS.get(y, "#888") for y in sorted(sel_years)})
        st_folium(m_gps, height=440, use_container_width=True, key="vt_map_gps", returned_objects=[])
        st.caption("Dots coloured by year: 🟠 2024 🔵 2025 🟢 2026. Click a dot for details.")
        card_close()

    # ── Row 2: monthly home range vs canopy | combined hotspot map ─────────
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        card_open("Monthly Home Range vs Vegetation")
        range_ele = st.multiselect("Elephants:", ALL_NAMES, default=ALL_NAMES, key="vt_range_ele")
        range_years = st.multiselect("Years:", ["2024", "2025", "2026"], default=["2024", "2025", "2026"], key="vt_range_years")
        sel_yrs_int = [int(y) for y in (range_years or ["2024", "2025", "2026"])]
        df_r = data["monthly_range"]
        df_r = df_r[df_r["name"].isin(range_ele or ALL_NAMES) & df_r["year"].isin(sel_yrs_int)]
        if df_r.empty:
            st.info("No data for selected elephants / years.")
        else:
            fig = go.Figure()
            for yr in sel_yrs_int:
                mc_row = du.VT_MOD_CANOPY[du.VT_MOD_CANOPY["year"] == yr]
                if mc_row.empty:
                    continue
                mc = mc_row["mod_canopy"].iloc[0]
                col = du.VT_YEAR_COLS.get(str(yr), "#888")
                fig.add_trace(go.Scatter(
                    x=du._MONTH_ABBR, y=[mc] * 12, mode="lines", fill="tozeroy",
                    fillcolor=_hex_alpha(col, 0.094), line=dict(color=_hex_alpha(col, 0.33), width=1, dash="dot"),
                    name=f"Mod.Canopy {yr} ({mc}%)", yaxis="y2",
                    hovertemplate=f"Moderate Canopy {yr}: {mc}%<extra></extra>",
                ))
            for nm in (range_ele or ALL_NAMES):
                e_col = data["ele_colors"].get(nm, "#888")
                nm_years = sorted(df_r.loc[df_r["name"] == nm, "year"].unique())
                first_year = nm_years[0] if nm_years else None
                for yr in sel_yrs_int:
                    sub = df_r[(df_r["name"] == nm) & (df_r["year"] == yr)].sort_values("month")
                    if len(sub) < 2:
                        continue
                    fig.add_trace(go.Scatter(
                        x=sub["month_lbl"], y=sub["radius_km"], mode="lines+markers",
                        line=dict(color=e_col, width=1.8), marker=dict(color=e_col, size=5),
                        name=f"{nm} {yr}", legendgroup=nm, showlegend=bool(yr == first_year),
                        hovertemplate=f"<b>{nm} {yr}</b><br>Month: %{{x}}<br>Range: %{{y:.2f}} km<extra></extra>",
                    ))
            _vt_light_layout(fig,
                xaxis=dict(title="Month", categoryorder="array", categoryarray=du._MONTH_ABBR),
                yaxis=dict(title="Home Range Radius (km)", side="left"),
                yaxis2=dict(title="Moderate Canopy (%)", overlaying="y", side="right", showgrid=False,
                            tickfont=dict(color="#64748b")),
                legend=dict(orientation="v", x=1.08, y=1, font=dict(size=9)),
                hovermode="x unified", margin=dict(t=10, l=50, r=110, b=40), height=380,
            )
            st.plotly_chart(fig, use_container_width=True)
        st.caption("Lines = monthly home range radius per elephant per year. Shaded band = Moderate Canopy % (right axis).")
        card_close()

    with r2c2:
        card_open("Hotspot Trajectory & Per-Elephant Dots")
        ele_pick = st.multiselect("Elephants:", ALL_NAMES, default=ALL_NAMES, key="vt_ele_pick")
        top_n_label = st.selectbox("Show top:", ["Top 1 (most visited)", "Top 5"], key="vt_top_n_each")
        n_each = 1 if top_n_label.startswith("Top 1") else 5
        sel_eles = ele_pick or ALL_NAMES

        m_comb = base_folium_map()
        tri_pts = du.vt_get_top(data["hs_all"], 1)
        for yr in sorted(tri_pts["year"].unique()):
            r = tri_pts[tri_pts["year"] == yr].iloc[0]
            yr_col = du.VT_YEAR_COLS.get(str(yr), "#888")
            folium.Marker(
                [r["lat"], r["lon"]], icon=_vt_triangle_icon(yr_col),
                popup=f"<b>All-Elephant Top Hotspot — {yr}</b><br>Lat: {r['lat']}N Lon: {r['lon']}E<br>"
                      f"Visits: <b>{r['visits']}</b><br>{r['elephants']}",
                tooltip=f"{yr} top hotspot",
            ).add_to(m_comb)
            folium.Marker(
                [r["lat"] + 0.0015, r["lon"]],
                icon=folium.DivIcon(html=f"<div style='color:{yr_col};font-weight:bold;font-size:12px;"
                                          f"text-shadow:0 0 4px #fff,0 0 4px #fff;'>{yr}</div>"),
            ).add_to(m_comb)

        for nm in sel_eles:
            e_col = data["ele_colors"].get(nm, "#888")
            e_pts = du.vt_get_top(data["hs_each"], n_each, name=nm)
            if e_pts.empty:
                continue
            for _, r in e_pts.iterrows():
                star = "  ★ Most Visited" if r["rank"] == 1 else f"  #{int(r['rank'])}"
                folium.CircleMarker(
                    [r["lat"], r["lon"]], radius=9 if r["rank"] == 1 else 6,
                    color="#ffffff", weight=1.5, fill=True, fill_color=e_col, fill_opacity=0.88,
                    popup=f"<b style='color:{e_col}'>{nm}</b> — {r['year']}{star}<br>Visits: <b>{r['visits']}</b>",
                ).add_to(m_comb)
            rank1 = e_pts[e_pts["rank"] == 1].sort_values("year")
            yrs_avail = rank1["year"].tolist()
            for i in range(len(yrs_avail) - 1):
                a = rank1[rank1["year"] == yrs_avail[i]].iloc[0]
                b = rank1[rank1["year"] == yrs_avail[i + 1]].iloc[0]
                _vt_add_directional_arrow(m_comb, a["lat"], a["lon"], b["lat"], b["lon"], e_col)

        _add_categorical_legend(m_comb, "🐘 Elephant hotspot", {nm: data["ele_colors"].get(nm, "#888") for nm in sel_eles})
        st_folium(m_comb, height=440, use_container_width=True, key="vt_map_combined", returned_objects=[])
        st.caption(
            "▲ Triangle = All-elephant #1 hotspot per year (🟠 2024  🔵 2025  🟢 2026). "
            "Dot = per-elephant most-visited spot; chevron shows direction 2024→2025→2026."
        )
        totals = elephants_df[elephants_df["name"].isin(sel_eles)].groupby("name").size()
        st.caption("Selected: " + "  ·  ".join(f"{nm} ({totals.get(nm, 0):,})" for nm in sel_eles))
        card_close()

    # ── Bottom row: fix counts ───────────────────────────────────────────────
    card_open("GPS Fix Counts per Elephant per Year")
    fc = data["fix_counts"]
    fig = go.Figure()
    all_names_sorted = sorted(fc["name"].unique())
    for yr in ["2024", "2025", "2026"]:
        sub = fc[fc["year"] == yr]
        if sub.empty:
            continue
        fig.add_trace(go.Bar(
            x=sub["name"], y=sub["fixes"], name=yr,
            marker=dict(color=du.VT_YEAR_COLS.get(yr, "#888"), opacity=0.88, line=dict(width=0.5, color=_hex_alpha("#000000", 0.13))),
            hovertemplate=f"<b>%{{x}}</b><br>{yr}: <b>%{{y}}</b> fixes<extra></extra>",
        ))
    _vt_light_layout(fig,
        barmode="group",
        xaxis=dict(title="", tickangle=-35, categoryorder="array", categoryarray=all_names_sorted,
                   tickfont=dict(color="#475569", size=10)),
        yaxis=dict(title="GPS fixes"),
        legend=dict(orientation="h", x=1, xanchor="right", y=1.12, bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=5, l=50, r=10, b=80), height=280,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Only years with actual recorded fixes shown per elephant.")
    card_close()


# ══════════════════════════════════════════════════════════════════════════
# TAB — TRACKING (general)  (new: ported from combined_dashboard_app's
# tracking module — timeline scrubber + climate-colored map + correlation
# scatter plots)
# ══════════════════════════════════════════════════════════════════════════
TK_ALL_NAMES = ["Gothami", "recollared female", "female_1", "Mina", "Talatha",
                "Dona", "Dewmi", "Rahu", "Tara Devi", "Kasun", "Wilmini", "Pazhani", "Damien"]
TK_CLIMATE_META = {
    "temp": ("Temperature", "°C"), "rainfall": ("Rainfall", "mm"), "wind_speed": ("Wind Speed", "m/s"),
}


@st.cache_resource(show_spinner="Joining GPS fixes with hourly climate…")
def _tk_dataset():
    hourly = du.load_climate_hourly()
    return du.build_tk_dataset(elephants_df, hourly)


def _tk_gradient_color(value, vmin, vmax):
    if pd.isna(value) or vmax <= vmin:
        return "#94a3b8"
    t = max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
    stops = [(0.0, (255, 107, 107)), (0.5, (255, 217, 61)), (1.0, (107, 203, 119))]
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if t0 <= t <= t1:
            f = (t - t0) / (t1 - t0) if t1 > t0 else 0
            rgb = tuple(int(c0[i] + (c1[i] - c0[i]) * f) for i in range(3))
            return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
    return "#94a3b8"


def render_tracking_general_tab():
    st.markdown("<div class='section-title'>🐘 Elephant Tracker & Climate Dashboard</div>", unsafe_allow_html=True)

    base = _tk_dataset()
    tk_ele_colors = du.build_elephant_palette(TK_ALL_NAMES, du.TK_PALETTE)

    c1, c2, c3 = st.columns([2, 6, 4])
    with c1:
        card_open()
        tk_eles = st.multiselect("Select Elephant(s):", TK_ALL_NAMES, default=["Gothami", "Talatha"], key="tk_eles")
        d_min, d_max = base["date"].min(), base["date"].max()
        tk_date_range = st.date_input("Select Date Range:", value=(d_min, d_max), min_value=d_min, max_value=d_max, key="tk_date_range")
        show_imputed = st.checkbox("Show Imputed Data Points", value=True, key="tk_show_imputed")
        tk_map_type = st.selectbox("Map Type:", ["OpenStreetMap", "Satellite", "Dark", "Light", "Topo"], key="tk_map_type")
        card_close()

    t_df = base[base["name"].isin(tk_eles or TK_ALL_NAMES)]
    if isinstance(tk_date_range, tuple) and len(tk_date_range) == 2:
        t_df = t_df[(t_df["date"] >= tk_date_range[0]) & (t_df["date"] <= tk_date_range[1])]
    if not show_imputed:
        t_df = t_df[~t_df["imputed"]]

    if t_df.empty:
        st.info("No matching tracking/climate data for this selection.")
        return

    with c1:
        card_open()
        t_min, t_max = float(np.floor(t_df["temp"].min())), float(np.ceil(t_df["temp"].max()))
        tk_temp_range = st.slider("Temperature Range (°C):", t_min, t_max, (t_min, t_max), step=1.0, key="tk_temp_range")
        tk_climate_type = st.radio("Climate variable:", ["temp", "rainfall", "wind_speed"],
                                    format_func=lambda k: TK_CLIMATE_META[k][0], key="tk_climate_type")
        card_close()

    df_filtered = t_df[(t_df["temp"] >= tk_temp_range[0]) & (t_df["temp"] <= tk_temp_range[1])]
    if df_filtered.empty:
        st.info("No data in the selected temperature range.")
        return

    unique_times = sorted(df_filtered["dt_round"].unique())
    with c1:
        card_open()
        step_idx = st.slider("Timeline — drag to accumulate points:", 1, len(unique_times), len(unique_times), key="tk_current_time")
        card_close()
    max_t = unique_times[step_idx - 1]
    df_upto = df_filtered[df_filtered["dt_round"] <= max_t]
    current_pt = df_upto.sort_values("dt_round").groupby("name").tail(1)

    c_type = tk_climate_type
    lbl, unit = TK_CLIMATE_META[c_type]

    with c3:
        card_open(f"Map Legend — {lbl}")
        st.markdown(
            "<div style='background:linear-gradient(to right,#FF6B6B,#FFD93D,#6BCB77);"
            "width:100%;height:10px;margin-bottom:4px;'></div>"
            "<div style='display:flex;justify-content:space-between;font-size:11px;color:#64748b;'>"
            "<span>Low</span><span>High</span></div>",
            unsafe_allow_html=True,
        )
        st.caption("⚪ White-bordered dot = current position (by elephant)")
        card_close()

    with c2:
        card_open("GPS Tracking Map")
        tile_map = {
            "OpenStreetMap": dict(tiles="OpenStreetMap"),
            "Satellite": dict(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Esri"),
            "Dark": dict(tiles="CartoDB dark_matter"),
            "Light": dict(tiles="CartoDB positron"),
            "Topo": dict(tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png", attr="OpenTopoMap contributors"),
        }
        m = folium.Map(prefer_canvas=True, **tile_map.get(tk_map_type, dict(tiles="OpenStreetMap")))
        vmin, vmax = df_filtered[c_type].min(), df_filtered[c_type].max()
        bounds = []
        for el in sorted(df_upto["name"].unique()):
            e_col = tk_ele_colors.get(el, "#888")
            e_df = df_upto[df_upto["name"] == el].sort_values("dt_round")
            gap_hrs = e_df["dt_round"].diff().dt.total_seconds() / 3600
            seg_id = (gap_hrs > 1.5).fillna(False).cumsum()
            for _, seg in e_df.groupby(seg_id):
                if len(seg) > 1:
                    pts = list(zip(seg["lat"], seg["lon"]))
                    folium.PolyLine(pts, color=e_col, weight=3, opacity=1.0).add_to(m)
                    bounds.extend(pts)
            for _, r in e_df.iterrows():
                clr = _tk_gradient_color(r[c_type], vmin, vmax)
                folium.CircleMarker(
                    [r["lat"], r["lon"]], radius=4, color="transparent", weight=0,
                    fill=True, fill_color=clr, fill_opacity=0.8,
                    popup=f"<b>{r['name']}</b><br>{r['dt_round']:%Y-%m-%d %H:%M}<br>"
                          f"Temp: {r['temp']:.1f}°C · Rain: {r['rainfall']:.1f}mm · Wind: {r['wind_speed']:.1f}m/s",
                ).add_to(m)
                bounds.append((r["lat"], r["lon"]))
            cur = current_pt[current_pt["name"] == el]
            if not cur.empty:
                r = cur.iloc[0]
                folium.CircleMarker(
                    [r["lat"], r["lon"]], radius=9, color="white", weight=2,
                    fill=True, fill_color=e_col, fill_opacity=1,
                    popup=f"<b>Current — {el}</b><br>{r['dt_round']:%Y-%m-%d %H:%M}<br>"
                          f"Temp: {r['temp']:.1f}°C · Rain: {r['rainfall']:.1f}mm · Wind: {r['wind_speed']:.1f}m/s",
                    tooltip=f"Current — {el}",
                ).add_to(m)
        if bounds:
            m.fit_bounds(bounds)
        st_folium(m, height=450, use_container_width=True, key="tk_map", returned_objects=[])
        card_close()

    with c3:
        card_open("Tracking Data (Longitude / Latitude)")
        st.caption("Points accumulate as you drag the Timeline slider.")
        lat_span = df_upto["lat"].max() - df_upto["lat"].min()
        lon_center = (df_upto["lon"].min() + df_upto["lon"].max()) / 2
        fig = go.Figure()
        for el in sorted(df_upto["name"].unique()):
            sub = df_upto[df_upto["name"] == el]
            fig.add_trace(go.Scatter(
                x=sub["lon"], y=sub["lat"], mode="markers", name=el,
                marker=dict(size=6, opacity=0.5, color=tk_ele_colors.get(el, "#888")),
                hovertemplate=f"<b>{el}</b><br>Lon: %{{x:.5f}}<br>Lat: %{{y:.5f}}<extra></extra>",
            ))
        if not current_pt.empty:
            fig.add_trace(go.Scatter(
                x=current_pt["lon"], y=current_pt["lat"], mode="markers", showlegend=False,
                marker=dict(size=12, color="white", line=dict(color="black", width=2)),
            ))
        fig.update_layout(
            xaxis=dict(title="Longitude", range=[lon_center - lat_span / 2, lon_center + lat_span / 2] if lat_span > 0 else None),
            yaxis=dict(title="Latitude"),
            plot_bgcolor="#f8f9fa", paper_bgcolor="white", margin=dict(t=10, b=40, l=50, r=20), height=440,
        )
        st.plotly_chart(fig, use_container_width=True)
        card_close()

    card_open(f"Climate Data — {lbl} Over Selected Date Range")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_upto["dt_round"], y=df_upto[c_type], mode="lines+markers",
        line=dict(color="#ff7f0e", width=2), marker=dict(size=5, color="#ff7f0e"), name=lbl,
        hovertemplate=f"<b>%{{x|%Y-%m-%d %H:%M}}</b><br>{lbl}: %{{y:.2f}} {unit}<extra></extra>",
    ))
    if not current_pt.empty:
        fig.add_trace(go.Scatter(
            x=current_pt["dt_round"], y=current_pt[c_type], mode="markers", showlegend=False,
            marker=dict(size=12, color="white", line=dict(color="black", width=2)),
        ))
    fig.update_layout(
        xaxis=dict(title="DateTime", showgrid=True, gridcolor="#e0e0e0"),
        yaxis=dict(title=f"{lbl} ({unit})", showgrid=True, gridcolor="#e0e0e0"),
        margin=dict(t=10, b=40, l=50, r=20), dragmode="select",
        plot_bgcolor="#f8f9fa", paper_bgcolor="white", height=380,
    )
    st.caption("Drag a box on the chart to highlight points.")
    st.plotly_chart(fig, use_container_width=True)
    card_close()

    st.markdown(f"<div class='sub-title'>{lbl} vs Location — Does {lbl} Relate to Elephant Movement?</div>", unsafe_allow_html=True)
    lc1, lc2 = st.columns(2)
    for axis_col, axis_lbl, col_slot in [("lat", "Latitude", lc1), ("lon", "Longitude", lc2)]:
        with col_slot:
            card_open(f"{lbl} vs {axis_lbl}")
            fig = go.Figure()
            for el in sorted(df_upto["name"].unique()):
                sub = df_upto[df_upto["name"] == el]
                fig.add_trace(go.Scatter(
                    x=sub[axis_col], y=sub[c_type], mode="markers", name=el,
                    marker=dict(size=6, opacity=0.7, color=tk_ele_colors.get(el, "#888")),
                    hovertemplate=f"<b>{el}</b><br>{axis_lbl}: %{{x:.5f}}<br>{lbl}: %{{y:.2f}} {unit}<extra></extra>",
                ))
            if not current_pt.empty:
                fig.add_trace(go.Scatter(
                    x=current_pt[axis_col], y=current_pt[c_type], mode="markers", showlegend=False,
                    marker=dict(size=12, color="white", line=dict(color="black", width=2)),
                ))
            fig.update_layout(
                xaxis=dict(title=axis_lbl, showgrid=True, gridcolor="#e0e0e0"),
                yaxis=dict(title=f"{lbl} ({unit})", showgrid=True, gridcolor="#e0e0e0"),
                margin=dict(t=10, b=40, l=50, r=20), plot_bgcolor="#f8f9fa", paper_bgcolor="white", height=480,
            )
            st.plotly_chart(fig, use_container_width=True)

            overall_r = df_upto[[axis_col, c_type]].dropna().corr().iloc[0, 1] if len(df_upto) > 1 else np.nan
            lines = [f"Overall correlation ({lbl} vs {axis_lbl}):  r = {overall_r:.3f}" if pd.notna(overall_r) else "Not enough data to compute a correlation."]
            if df_upto["name"].nunique() > 1:
                lines.append("Per elephant:")
                for el, sub in df_upto.groupby("name"):
                    r = sub[[axis_col, c_type]].dropna().corr().iloc[0, 1] if len(sub) > 1 else np.nan
                    lines.append(f"  {el:<20s} r = {r:.3f}" if pd.notna(r) else f"  {el:<20s} r = n/a")
            st.code("\n".join(lines), language=None)
            card_close()


def render_tracking_tab():
    st.markdown("<div class='section-title'>🗺️ Elephant Tracking Overview</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Interactive Map with Satellite View</div>", unsafe_allow_html=True)

    st_folium(_build_tracking_map(), height=600, use_container_width=True, key="tracking_map", returned_objects=[])

    st.markdown(
        "<div style='text-align:center; margin: 18px 0;'>"
        "<a href='https://zubhp3-amali-priyanwada.shinyapps.io/elephants_by_month/' target='_blank' "
        "style='text-decoration:none;'>"
        "<button style='background-color:#2E8B57; color:white; border:none; padding:12px 25px; "
        "font-size:16px; font-weight:bold; border-radius:8px; cursor:pointer;'>"
        "Click here for the Month Wise GPS Tracking Data Analysis</button></a></div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='sub-title'>📊 Tracking Data Visualization</div>", unsafe_allow_html=True)
    st.pyplot(_build_tracking_scatter_fig())

    st.markdown("<div class='sub-title'>👥 GPS Tracking Data by Individual Elephant — Monthly Playback</div>", unsafe_allow_html=True)
    render_tracking_playback()


# ══════════════════════════════════════════════════════════════════════════
# TAB 6 — LIVE ELEPHANT PATH (frame-by-frame playback)
# ══════════════════════════════════════════════════════════════════════════
def render_live_tab():
    c1, c2 = st.columns([4, 8])
    with c1:
        card_open("🐘 Choose Elephant")
        live_elephant = st.selectbox("Elephant", ALL_NAMES, key="live_elephant")
        live_month_label = st.selectbox("Month", list(MONTH_CHOICES.keys()), key="live_month")
        live_month = MONTH_CHOICES[live_month_label]
        st.caption("Elephant + Month here are specific to this page. The sidebar's Date Range still applies too. "
                    "Drag the slider, or press ▶ Play to animate.")

        df = elephants_df[
            (elephants_df["name"] == live_elephant) &
            (elephants_df["date_parsed"] >= date_start) & (elephants_df["date_parsed"] <= date_end)
        ]
        if live_month != "all":
            df = df[df["year_month"] == live_month]
        df = df.sort_values("datetime_sl").reset_index(drop=True)

        if not df.empty:
            st.metric("GPS Fixes", f"{len(df):,}")
        card_close()

    with c2:
        if df.empty:
            card_open("🎬 Playback")
            st.info("No GPS fixes for this elephant in the selected date range.")
            card_close()
            return
        render_live_playback(df, live_elephant)


def _static_center_zoom(pts):
    """Compute a center + zoom level once from the FULL track."""
    import math
    lats = [p[0] for p in pts]
    lons = [p[1] for p in pts]
    lat_span = max(max(lats) - min(lats), 0.002)
    lon_span = max(max(lons) - min(lons), 0.002)
    span = max(lat_span, lon_span) * 1.35
    zoom = math.floor(math.log2(360.0 / span))
    zoom = int(min(max(zoom, 3), 16))
    center = ((max(lats) + min(lats)) / 2, (max(lons) + min(lons)) / 2)
    return center, zoom


def _build_live_map_component(df, live_elephant, clr):
    """A fully self-contained, client-side-animated Leaflet map: the map is
    created ONCE in the browser, and Play/Pause/slider only call incremental
    Leaflet API updates (setLatLng / setLatLngs) — exactly how R Shiny's
    leafletProxy() works. Nothing round-trips through Python per frame, so
    there's no per-frame map rebuild and therefore no flicker."""
    import json

    n = len(df)
    lats = df["lat"].tolist()
    lons = df["lon"].tolist()
    times = df["datetime_sl"].dt.strftime("%d %b %Y %H:%M").tolist()

    step_km = [None]
    for i in range(1, n):
        step_km.append(round(du.haversine_km(lats[i - 1], lons[i - 1], lats[i], lons[i]), 3))

    # Precompute hull polygons at a manageable number of checkpoints (matches
    # the previous "growing hull" sampling) rather than every single frame —
    # concave-hull computation stays server-side (Python/shapely), only the
    # lookup-by-frame happens client-side.
    checkpoints = []
    step = max(1, n // 60)
    idxs = sorted(set(list(range(3, n + 1, step)) + ([n] if n >= 3 else [])))
    for i in idxs:
        h = du.compute_hull(df.iloc[:i], ratio=0.3)
        if h:
            checkpoints.append({"frame": i, "area": round(h["area_km2"], 3),
                                 "poly": list(zip(h["lats"], h["lons"]))})

    pts_full = list(zip(lats, lons))
    center, zoom = _static_center_zoom(pts_full)

    data = {
        "lats": lats, "lons": lons, "times": times, "step": step_km,
        "checkpoints": checkpoints, "center": list(center), "zoom": zoom,
        "color": clr, "name": live_elephant, "n": n,
    }
    data_json = json.dumps(data)

    html = """
<div id="live_root" style="font-family:'Segoe UI',system-ui,sans-serif;">
  <div style="display:flex;gap:14px;align-items:center;margin-bottom:10px;">
    <button id="live_play_btn" style="background:#0f766e;color:white;border:none;border-radius:8px;
      padding:8px 18px;font-size:14px;font-weight:700;cursor:pointer;">▶ Play</button>
    <input id="live_slider" type="range" min="1" style="flex:1;accent-color:#0f766e;">
    <span id="live_frame_label" style="font-size:13px;color:#475569;white-space:nowrap;min-width:110px;text-align:right;"></span>
    <div style="display:flex;gap:2px;background:#e2e8f0;border-radius:8px;padding:2px;">
      <button id="live_basemap_light" class="live_basemap_btn" style="border:none;border-radius:6px;
        padding:6px 12px;font-size:12px;font-weight:700;cursor:pointer;background:#0f766e;color:white;">Light</button>
      <button id="live_basemap_street" class="live_basemap_btn" style="border:none;border-radius:6px;
        padding:6px 12px;font-size:12px;font-weight:700;cursor:pointer;background:transparent;color:#475569;">Street</button>
      <button id="live_basemap_satellite" class="live_basemap_btn" style="border:none;border-radius:6px;
        padding:6px 12px;font-size:12px;font-weight:700;cursor:pointer;background:transparent;color:#475569;">Satellite</button>
    </div>
  </div>
  <div style="background:white;border-radius:12px;padding:14px 18px;margin-bottom:12px;box-shadow:0 2px 10px rgba(0,0,0,.06);
              display:grid;grid-template-columns:repeat(4,1fr);gap:10px 18px;font-size:13px;color:#334155;">
    <div><b>Elephant</b><br><span id="f_elephant"></span></div>
    <div><b>Time (SL)</b><br><span id="f_time"></span></div>
    <div><b>Latitude</b><br><span id="f_lat"></span></div>
    <div><b>Longitude</b><br><span id="f_lon"></span></div>
    <div><b>Step distance</b><br><span id="f_step"></span></div>
    <div><b>Hull area so far</b><br><span id="f_hull"></span></div>
    <div style="grid-column:span 2;"><b>Progress</b><br><span id="f_progress"></span>
      <div style="background:#e2e8f0;border-radius:6px;height:6px;margin-top:5px;overflow:hidden;">
        <div id="f_progress_bar" style="background:#0f766e;height:100%;width:0%;"></div>
      </div>
    </div>
  </div>
  <div id="live_map_div" style="height:480px;border-radius:12px;overflow:hidden;box-shadow:0 2px 10px rgba(0,0,0,.08);"></div>
</div>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
(function() {
  const D = __DATA_JSON__;

  const map = L.map('live_map_div', {
    zoomAnimation: false, fadeAnimation: false, markerZoomAnimation: false,
  }).setView(D.center, D.zoom);

  // Same three tile sources as the Home Range & Speed map's folium LayerControl
  const lightLayer = L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; OpenStreetMap &copy; CARTO', maxZoom: 19,
  });
  const streetLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; OpenStreetMap contributors', maxZoom: 19,
  });
  const satelliteLayer = L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    { attribution: 'Tiles &copy; Esri', maxZoom: 19 }
  );
  lightLayer.addTo(map);

  const layers = { light: lightLayer, street: streetLayer, satellite: satelliteLayer };
  let currentBasemap = 'light';
  const basemapBtns = {
    light: document.getElementById('live_basemap_light'),
    street: document.getElementById('live_basemap_street'),
    satellite: document.getElementById('live_basemap_satellite'),
  };
  function setBasemap(which) {
    if (which === currentBasemap) return;
    map.removeLayer(layers[currentBasemap]);
    layers[which].addTo(map);
    currentBasemap = which;
    for (const key in basemapBtns) {
      const active = key === which;
      basemapBtns[key].style.background = active ? '#0f766e' : 'transparent';
      basemapBtns[key].style.color = active ? 'white' : '#475569';
    }
  }
  basemapBtns.light.addEventListener('click', function() { setBasemap('light'); });
  basemapBtns.street.addEventListener('click', function() { setBasemap('street'); });
  basemapBtns.satellite.addEventListener('click', function() { setBasemap('satellite'); });

  const fullTrack = D.lats.map((la, i) => [la, D.lons[i]]);
  L.polyline(fullTrack, {color: D.color, weight: 1, opacity: 0.2}).addTo(map);
  const traveled = L.polyline([], {color: D.color, weight: 3, opacity: 0.95}).addTo(map);
  const hullPoly = L.polygon([], {color: D.color, weight: 2, fillColor: D.color, fillOpacity: 0.15}).addTo(map);
  const marker = L.circleMarker(fullTrack[0], {
    radius: 8, color: '#ffffff', weight: 2, fillColor: D.color, fillOpacity: 1,
  }).addTo(map);

  function hullForFrame(f) {
    let best = null;
    for (const cp of D.checkpoints) { if (cp.frame <= f) best = cp; else break; }
    return best;
  }

  function setFrame(f) {
    f = Math.max(1, Math.min(D.n, f));
    frame = f;
    traveled.setLatLngs(fullTrack.slice(0, f));
    marker.setLatLng(fullTrack[f - 1]);
    const cp = hullForFrame(f);
    hullPoly.setLatLngs(cp ? cp.poly : []);
    document.getElementById('f_elephant').innerText = D.name;
    document.getElementById('f_time').innerText = D.times[f - 1];
    document.getElementById('f_lat').innerText = D.lats[f - 1].toFixed(5) + ' °N';
    document.getElementById('f_lon').innerText = D.lons[f - 1].toFixed(5) + ' °E';
    document.getElementById('f_step').innerText = (D.step[f - 1] != null) ? (D.step[f - 1] + ' km') : '—';
    document.getElementById('f_hull').innerText = cp ? (cp.area + ' km²') : '— (need ≥ 3 fixes)';
    document.getElementById('f_progress').innerText = f + ' / ' + D.n + ' fixes';
    document.getElementById('f_progress_bar').style.width = (100 * f / D.n) + '%';
    document.getElementById('live_frame_label').innerText = 'Fix ' + f + ' of ' + D.n;
    document.getElementById('live_slider').value = f;
  }

  let frame = D.n;
  let playing = false;
  let timer = null;
  const slider = document.getElementById('live_slider');
  const playBtn = document.getElementById('live_play_btn');
  slider.max = D.n;
  slider.value = D.n;

  slider.addEventListener('input', function() {
    pause();
    setFrame(parseInt(this.value));
  });

  function pause() {
    playing = false;
    if (timer) { clearInterval(timer); timer = null; }
    playBtn.innerText = '▶ Play';
  }
  function play() {
    if (frame >= D.n) { setFrame(1); }
    playing = true;
    playBtn.innerText = '⏸ Pause';
    timer = setInterval(function() {
      if (frame < D.n) { setFrame(frame + 1); } else { pause(); }
    }, 250);
  }
  playBtn.addEventListener('click', function() { playing ? pause() : play(); });

  if (fullTrack.length) { map.fitBounds(fullTrack, {padding: [20, 20]}); }
  setFrame(D.n);
})();
</script>
"""
    html = html.replace("__DATA_JSON__", data_json)
    return html


def render_live_playback(df, live_elephant):
    clr = du.get_color(live_elephant)
    card_open(
        "🎬 Live Playback — Path Drawn in Real Time",
        "The map updates in place as you play or drag the slider (no page/map reload each frame), same as the "
        "original R app's live-updating map. The shaded polygon is the elephant's home-range (concave hull) "
        "built only from the fixes seen so far.",
    )
    st.components.v1.html(_build_live_map_component(df, live_elephant, clr), height=760, scrolling=False)
    card_close()

    card_open("📐 Home-Range (Hull) Area — Growth Over The Full Track",
               "Concave-hull area (km²) as more fixes accumulate, shown for the whole selected period.")
    hull_rows = []
    step = max(1, len(df) // 60)
    idxs = sorted(set(list(range(3, len(df) + 1, step)) + ([len(df)] if len(df) >= 3 else [])))
    for i in idxs:
        h = du.compute_hull(df.iloc[:i], ratio=0.3)
        if h:
            hull_rows.append({"datetime_sl": df.iloc[i - 1]["datetime_sl"], "area_km2": h["area_km2"]})
    if hull_rows:
        hdf = pd.DataFrame(hull_rows)
        fig = go.Figure(go.Scatter(x=hdf["datetime_sl"], y=hdf["area_km2"], mode="lines+markers",
                                     line=dict(color=clr, width=2.5), marker=dict(color=clr, size=5)))
        fig.update_layout(paper_bgcolor="#fff", plot_bgcolor="#fff", font=dict(color="#333", family="Segoe UI"),
                           yaxis=dict(title="Home-range / hull area (km²)", gridcolor="#e5e5e5"),
                           xaxis=dict(title="", gridcolor="#e5e5e5"), margin=dict(t=20, b=40, l=60, r=20), height=300)
        st.plotly_chart(fig, use_container_width=True, key="live_hull_plot")
    else:
        st.info("Need at least 3 fixes to compute a hull.")
    card_close()

    lc1, lc2 = st.columns(2)
    with lc1:
        card_open("📍 Latitude vs Time (full track)")
        fig = go.Figure(go.Scatter(x=df["datetime_sl"], y=df["lat"], mode="lines+markers",
                                     line=dict(color=clr, width=1.5), marker=dict(color=clr, size=3), showlegend=False))
        fig.update_layout(paper_bgcolor="#fff", plot_bgcolor="#fff", font=dict(color="#333", family="Segoe UI"),
                           yaxis=dict(title="Latitude (°N)", gridcolor="#e5e5e5"), xaxis=dict(title="", gridcolor="#e5e5e5"),
                           margin=dict(t=20, b=40, l=60, r=20), height=320)
        st.plotly_chart(fig, use_container_width=True, key="live_lat_plot")
        card_close()
    with lc2:
        card_open("📍 Longitude vs Time (full track)")
        fig = go.Figure(go.Scatter(x=df["datetime_sl"], y=df["lon"], mode="lines+markers",
                                     line=dict(color=clr, width=1.5), marker=dict(color=clr, size=3), showlegend=False))
        fig.update_layout(paper_bgcolor="#fff", plot_bgcolor="#fff", font=dict(color="#333", family="Segoe UI"),
                           yaxis=dict(title="Longitude (°E)", gridcolor="#e5e5e5"), xaxis=dict(title="", gridcolor="#e5e5e5"),
                           margin=dict(t=20, b=40, l=60, r=20), height=320)
        st.plotly_chart(fig, use_container_width=True, key="live_lon_plot")
        card_close()



# ══════════════════════════════════════════════════════════════════════════
# TAB 7 — MIGRATION & CLIMATE
# ══════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Building availability calendar…")
def _build_availability_calendar(avail_elephant):
    sub = elephants_df[elephants_df["name"] == avail_elephant].copy()
    sub["day"] = sub["datetime_sl"].dt.date
    daily = sub.groupby("day").size().reset_index(name="valid_records").rename(columns={"day": "date"})
    full_range = pd.date_range(elephants_df["datetime_sl"].dt.date.min(),
                                elephants_df["datetime_sl"].dt.date.max(), freq="D").date
    daily = pd.DataFrame({"date": full_range}).merge(daily, on="date", how="left")
    daily["valid_records"] = daily["valid_records"].fillna(0)
    daily["availability"] = np.minimum(100, 100 * daily["valid_records"] / 24)
    if daily.empty:
        return None
    return make_calendar_heatmap(
        daily["date"], daily["availability"],
        title=f"Daily GPS Availability Calendar Heatmap – {avail_elephant}",
        discrete_breaks=du.AVAILABILITY_BREAKS, discrete_labels=du.AVAILABILITY_LABELS,
        discrete_colors=du.AVAILABILITY_COLORS,
    )


@st.cache_resource(show_spinner="Building migration map…")
def _build_migration_map_cached(mig_year, mig_month, mig_elephant, mig_weeks_tuple, show_numbers):
    """Bundles the map build AND the HTML render + base64 encode (used for
    the 'open in new tab' button) into one cached call — both were
    previously redone on every rerun of this page, including reruns
    triggered by unrelated widgets like the availability-calendar dropdown
    above it."""
    dat = elephants_df[
        (elephants_df["year"] == mig_year) & (elephants_df["month"] == mig_month) &
        (elephants_df["name"] == mig_elephant)
    ].copy()
    if not dat.empty:
        dat["week_of_month"] = du.week_of_month(dat["datetime_sl"])
        if mig_weeks_tuple and "All Weeks" not in mig_weeks_tuple:
            dat = dat[dat["week_of_month"].isin(mig_weeks_tuple)]
    m = build_migration_map(dat, show_numbers=show_numbers)
    html_bytes = m.get_root().render().encode("utf-8")
    b64 = base64.b64encode(html_bytes).decode("utf-8")
    return m, b64


@st.cache_data(show_spinner="Building climate calendar…")
def _build_climate_calendar(clim_var):
    info = du.CLIMATE_PLOT_INFO[clim_var]
    vals = climate_df[info["col"]].astype(float)
    return make_calendar_heatmap(
        climate_df["date"], vals, title=info["title"],
        discrete_breaks=info["breaks"], discrete_labels=info["labels"], discrete_colors=du.CALENDAR_COLORS,
        height=500,
    )


def render_climate_tab():
    st.markdown("<div class='section-title'>🐘 Elephant Tracking Data Availability</div>", unsafe_allow_html=True)
    card_open()
    fc1, fc2 = st.columns([3, 9])
    with fc1:
        avail_elephant = st.selectbox("Select Elephant Name:", ALL_NAMES, key="avail_elephant")
    with fc2:
        st.caption("This heatmap shows the percentage of valid GPS records captured per day (max 24 records/day).")
    card_close()
    card_open()
    fig = _build_availability_calendar(avail_elephant)
    if fig is None:
        st.info("No data available for the selected elephant.")
    else:
        st.plotly_chart(fig, use_container_width=True, key="calendar_plot")
    card_close()

    st.markdown("<div class='section-title'>🐘 Elephant Migration Map</div>", unsafe_allow_html=True)
    card_open()
    mf1, mf2, mf3, mf4, mf5 = st.columns([2, 2, 2, 3, 2])
    with mf1:
        years_avail = sorted(elephants_df["year"].unique())
        mig_year = st.selectbox("Select Year", years_avail, key="mig_year")
    with mf2:
        mig_month = st.selectbox("Select Month", [f"{m:02d}" for m in range(1, 13)], key="mig_month")
    with mf3:
        mig_elephant = st.selectbox("Select Elephant", ALL_NAMES, key="mig_elephant")
    with mf4:
        mig_weeks = st.multiselect(
            "Select Week", ["All Weeks", "Week 1", "Week 2", "Week 3", "Week 4"],
            default=["All Weeks"], key="mig_weeks",
        )
    with mf5:
        show_seq_numbers = st.checkbox("Show point sequence numbers", value=False, key="show_seq_numbers")
    card_close()
    card_open()
    m, b64 = _build_migration_map_cached(
        mig_year, mig_month, mig_elephant, tuple(sorted(mig_weeks)), show_seq_numbers
    )
    components.html(
        f"""
        <div style="text-align:right;">
          <button onclick="openMigrationMap()" style="background:#2E8B57; color:white; border:none;
            padding:6px 14px; font-size:13px; font-weight:600; border-radius:6px; cursor:pointer;">
            🔗 Open Map in New Tab
          </button>
        </div>
        <script>
        function openMigrationMap() {{
            const b64 = "{b64}";
            const byteChars = atob(b64);
            const byteNumbers = new Array(byteChars.length);
            for (let i = 0; i < byteChars.length; i++) {{
                byteNumbers[i] = byteChars.charCodeAt(i);
            }}
            const byteArray = new Uint8Array(byteNumbers);
            const blob = new Blob([byteArray], {{ type: "text/html" }});
            const url = URL.createObjectURL(blob);
            window.open(url, "_blank");
        }}
        </script>
        """,
        height=44,
    )
    st_folium(m, height=600, use_container_width=True, key="migration_map", returned_objects=[])
    card_close()

    st.markdown("<div class='section-title'>🌡 Climate Calendar Analysis</div>", unsafe_allow_html=True)
    card_open()
    cc1, cc2 = st.columns([3, 9])
    with cc1:
        clim_var = st.selectbox("Climate Variable", list(du.CLIMATE_PLOT_INFO.keys()), key="clim_var")
    with cc2:
        st.caption("This calendar heatmap displays daily values of the selected climate variable.")
    card_close()

    info = du.CLIMATE_PLOT_INFO[clim_var]
    vals = climate_df[info["col"]].astype(float)

    card_open()
    fig = _build_climate_calendar(clim_var)
    st.plotly_chart(fig, use_container_width=True, key="climate_calendar_plot")
    card_close()

    # Full-width summary table below both columns — a Measure x Year table
    # needs real horizontal room, which the narrow sidebar column doesn't have.
    card_open()
    st.markdown("#### Summary Statistics")

    yrs = sorted(climate_df["date"].dt.year.unique())
    measures = ["Minimum", "First Quantile (Q1)", "Median", "Mean",
                "Third Quantile (Q3)", "Maximum", "Standard Deviation", "Missing Values (NA)"]
    per_year = {}
    for yr in yrs:
        v = climate_df.loc[climate_df["date"].dt.year == yr, info["col"]].astype(float)
        per_year[yr] = [v.min(), v.quantile(0.25), v.median(), v.mean(),
                         v.quantile(0.75), v.max(), v.std(), v.isna().sum()]

    rows_html = ""
    for i, m in enumerate(measures):
        cells = "".join(
            f"<td style='padding:7px 14px;text-align:right;border:1px solid #d1d5db;'>"
            f"{per_year[yr][i]:.2f}</td>"
            for yr in yrs
        )
        rows_html += f"<tr><td style='padding:7px 14px;border:1px solid #d1d5db;color:#334155;'>{m}</td>{cells}</tr>"
    header_cells = "".join(
        f"<th style='padding:7px 14px;text-align:right;border:1px solid #d1d5db;background:#f1f5f9;'>{yr}</th>"
        for yr in yrs
    )
    st.markdown(
        f"""
        <table style='width:100%;border-collapse:collapse;border:1px solid #d1d5db;font-size:13px;'>
            <thead><tr>
                <th style='padding:7px 14px;text-align:left;border:1px solid #d1d5db;background:#f1f5f9;'>Measure</th>
                {header_cells}
            </tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )
    card_close()


# ══════════════════════════════════════════════════════════════════════════
# TAB 8 — DATA TABLE
# ══════════════════════════════════════════════════════════════════════════
def render_data_tab():
    card_open("📋 GPS Observation Records")
    df = filtered_df[["name", "sex", "datetime_sl", "lat", "lon"]].copy()
    df["datetime_sl"] = df["datetime_sl"].dt.strftime("%d %b %Y %H:%M")
    df["lat"] = df["lat"].round(6)
    df["lon"] = df["lon"].round(6)
    df = df.rename(columns={"name": "Elephant", "sex": "Sex", "datetime_sl": "Date/Time (SL)",
                             "lat": "Latitude", "lon": "Longitude"})
    st.dataframe(df, use_container_width=True, height=560, hide_index=True)
    st.download_button(
        "⬇ Download CSV", df.to_csv(index=False).encode("utf-8"),
        file_name="kaudulla_gps_records.csv", mime="text/csv",
    )
    card_close()


# ══════════════════════════════════════════════════════════════════════════
# TAB 9 — HOME RANGE & SPEED (MCP module)
# ══════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Building home range map…")
def _build_mcp_hull_map(df, hulls, mcp_colors_subset, summary_df, elephants_present):
    """df/hulls/summary_df/elephants_present together fully determine the
    map, so caching on them means dragging an unrelated widget (e.g. the
    focus-elephant picker staying the same) or a rerun triggered elsewhere
    on the page reuses this map instead of rebuilding every polygon and
    point marker from scratch."""
    m = folium.Map(location=[mcp_df["lat"].mean(), mcp_df["lon"].mean()], zoom_start=12, control_scale=True, prefer_canvas=True)
    folium.TileLayer("CartoDB positron", name="Light").add_to(m)
    folium.TileLayer("OpenStreetMap", name="Street").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satellite",
    ).add_to(m)
    for el in elephants_present:
        color = mcp_colors_subset[el]
        h = hulls.get(el)
        info_row = summary_df[summary_df["Elephant"] == el]
        if h and not info_row.empty:
            r = info_row.iloc[0]
            popup_html = (
                f"<div style='width:220px;'><h4>{el} — MCP</h4>"
                f"<b>Sex:</b> {r['Sex']}<br><b>Area:</b> {r['Area (km²)']} km² ({r['Area (ha)']:,.0f} ha)<br>"
                f"<b>GPS fixes:</b> {r['GPS Fixes']:,}<br><b>Total distance:</b> {r['Total Dist. (km)']} km<br>"
                f"<b>Avg speed:</b> {r['Avg Speed (km/h)']} km/h</div>"
            )
            folium.Polygon(list(zip(h["lats"], h["lons"])), color=color, weight=2, fill=True,
                            fill_color=color, fill_opacity=0.15, popup=popup_html,
                            tooltip=f"{el} — Hull").add_to(m)
    for el in elephants_present:
        edata = df[df["name"] == el]
        color = mcp_colors_subset[el]
        fg = folium.FeatureGroup(name=f"{el} — points", show=False)
        step = max(1, len(edata) // 150)
        for _, r in edata.iloc[::step].iterrows():
            folium.CircleMarker([r["lat"], r["lon"]], radius=3, color=color, fill=True, fill_color=color,
                                  fill_opacity=0.6, weight=1, tooltip=f"{el} — {r['datetime']:%d %b %Y}").add_to(fg)
        fg.add_to(m)
        if len(edata):
            folium.CircleMarker([edata["lat"].mean(), edata["lon"].mean()], radius=7, color="#ffffff",
                                  weight=2, fill=True, fill_color=color, fill_opacity=1,
                                  popup=f"<b>Center:</b> {el}").add_to(m)
    if elephants_present:
        all_pts = list(zip(df["lat"], df["lon"]))
        if all_pts:
            m.fit_bounds(all_pts)
    folium.LayerControl(collapsed=False).add_to(m)
    _add_categorical_legend(m, "Elephant", {el: mcp_colors_subset[el] for el in elephants_present})
    return m


def render_mcp_tab():
    st.markdown("<div class='section-title'>🧭 Home Range, Movement & Speed</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-description'>Minimum convex / concave polygon home ranges, GPS movement tracks, "
        "and speed/direction metrics for elephants tracked at Kaudulla National Park.</div>",
        unsafe_allow_html=True,
    )

    card_open("Filters")
    fc1, fc2, fc3 = st.columns([3, 3, 6])
    with fc1:
        mcp_sex_filter = st.multiselect("Sex", SEXES, default=SEXES, key="mcp_sex_filter")
    with fc2:
        focus_options = sorted(
            elephants_df.loc[elephants_df["name"].isin(sel_elephants), "name"].unique()
        )
        focus_elephants = st.multiselect(
            "Focus on specific elephant(s)", focus_options, key="mcp_focus_elephants",
            help="Pick one or more to hide every other elephant's hull/plots on this tab only "
                 "— doesn't change your sidebar selection. Leave empty to show everyone selected "
                 "in the sidebar.",
        )
    with fc3:
        st.markdown(
            "<div style='padding-top:6px;color:#64748b;font-size:13px;line-height:1.6;'>"
            "ℹ️ Elephant, Date Range, and Month are controlled from the <b>sidebar</b> on the left and apply "
            "to this tab too, so it stays in sync with the other plots.</div>", unsafe_allow_html=True,
        )
    card_close()

    df = get_mcp_filtered()
    if focus_elephants:
        df = df[df["name"].isin(focus_elephants)]

    c1, c2, c3, c4 = st.columns(4)
    with c1: vbox(df["name"].nunique(), "Elephants Shown", "🚏", "green")
    with c2: vbox(f"{len(df):,}", "GPS Points", "📍", "olive")
    with c3:
        s = df["speed_kmh"].replace([np.inf, -np.inf], np.nan).dropna()
        vbox(f"{s.mean():.2f} km/h" if len(s) else "—", "Avg. Speed", "🚀", "teal")
    with c4:
        dist = df["step_km"].replace([np.inf, -np.inf], np.nan).dropna().sum()
        vbox(f"{dist:,.1f} km", "Total Distance", "📏", "blue")

    # ---- hull + summary computation ----
    elephants_present = sorted(df["name"].unique())
    hulls, summary_rows = {}, []
    for el in elephants_present:
        edata = df[df["name"] == el]
        h = du.compute_hull_cached(tuple(edata["lon"]), tuple(edata["lat"]), ratio=0.3)
        hulls[el] = h
        step_dist = edata["step_km"].replace([np.inf, -np.inf], np.nan).dropna()
        step_speed = edata["speed_kmh"].replace([np.inf, -np.inf], np.nan).dropna()
        days_tracked = (edata["datetime"].max() - edata["datetime"].min()).total_seconds() / 86400
        summary_rows.append({
            "Elephant": el, "Sex": edata["sex"].iloc[0], "Days Tracked": round(days_tracked),
            "Total Dist. (km)": round(step_dist.sum(), 1),
            "km / day": round(step_dist.sum() / max(days_tracked, 1), 2),
            "Max Step (km)": round(step_dist.max(), 2) if len(step_dist) else np.nan,
            "GPS Fixes": len(edata),
            "Area (km²)": round(h["area_km2"], 3) if h else np.nan,
            "Area (ha)": round(h["area_km2"] * 100) if h else np.nan,
            "Avg Speed (km/h)": round(step_speed.mean(), 2) if len(step_speed) else np.nan,
            "Max Speed (km/h)": round(step_speed.max(), 2) if len(step_speed) else np.nan,
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("Area (km²)", ascending=False) if summary_rows else pd.DataFrame()

    card_open("Tracking Points & Minimum Non Convex Polygons")
    mcp_colors_subset = {el: mcp_colors[el] for el in elephants_present}
    m = _build_mcp_hull_map(df, hulls, mcp_colors_subset, summary_df, tuple(elephants_present))
    st_folium(m, height=600, use_container_width=True, key="mcp_hull_map", returned_objects=[])
    card_close()

    card_open("Cumulative Distance Traveled Over Time")
    if df.empty:
        st.info("No data for the selected filters.")
    else:
        cdf = df.dropna(subset=["step_km"]).sort_values(["name", "datetime"]).copy()
        cdf["cum_dist_km"] = cdf.groupby("name")["step_km"].cumsum()
        fig = go.Figure()
        for el in elephants_present:
            sub = cdf[cdf["name"] == el]
            fig.add_trace(go.Scatter(x=sub["datetime"], y=sub["cum_dist_km"], mode="lines", name=el,
                                       line=dict(color=mcp_colors[el])))
        fig.update_layout(paper_bgcolor="#fff", plot_bgcolor="#fff", font=dict(color="#333", family="Segoe UI"),
                           xaxis=dict(title="", gridcolor="#e5e5e5"),
                           yaxis=dict(title="Cumulative distance (km)", gridcolor="#e5e5e5"),
                           legend=dict(orientation="h", y=-0.2), height=550)
        st.plotly_chart(fig, use_container_width=True, key="mcp_timeline_plot")
    card_close()

    card_open("Home Range Area by Elephant")
    if summary_df.empty or summary_df["Area (km²)"].isna().all():
        st.info("No elephant in this selection has enough GPS fixes for a home range.")
    else:
        bar_df = summary_df.dropna(subset=["Area (km²)"]).sort_values("Area (km²)")
        fig = go.Figure(go.Bar(
            x=bar_df["Area (km²)"], y=bar_df["Elephant"], orientation="h",
            marker=dict(color=[mcp_colors[e] for e in bar_df["Elephant"]]),
            text=[f"{v} km²" for v in bar_df["Area (km²)"]], textposition="auto",
        ))
        fig.update_layout(paper_bgcolor="#fff", plot_bgcolor="#fff", font=dict(color="#333", family="Segoe UI"),
                           xaxis=dict(title="Area (km²)", gridcolor="#e5e5e5"), yaxis=dict(title=""),
                           margin=dict(l=10, r=20, t=10, b=40), height=550)
        st.plotly_chart(fig, use_container_width=True, key="mcp_area_bar_chart")
    card_close()

    rc1, rc2 = st.columns(2)
    with rc1:
        card_open("Movement Direction by Elephant (16 compass sectors)")
        rose_df = df[np.isfinite(df["bearing"])]
        if rose_df.empty:
            st.info("No movement data for the selected filters.")
        else:
            present = sorted(rose_df["name"].unique())
            n_cols = 2
            for row_start in range(0, len(present), n_cols):
                cols = st.columns(n_cols)
                for j, el in enumerate(present[row_start:row_start + n_cols]):
                    with cols[j]:
                        bd = du.bin_bearings(rose_df[rose_df["name"] == el]["bearing"])
                        fig = go.Figure(go.Barpolar(
                            r=bd["r"], theta=bd["theta"], marker_color=mcp_colors[el],
                            marker_line_color="white", marker_line_width=0.5,
                        ))
                        fig.update_layout(
                            polar=dict(angularaxis=dict(tickmode="array", tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                                                          ticktext=["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
                                                          direction="clockwise", rotation=90),
                                        radialaxis=dict(gridcolor="#e5e5e5")),
                            showlegend=False, margin=dict(l=20, r=20, t=30, b=20), height=280,
                            title=dict(text=el, font=dict(size=12)),
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"rose_{el}")
        card_close()
    with rc2:
        card_open("Overall Movement Direction — All Selected Elephants Combined")
        rose_df = df[np.isfinite(df["bearing"])]
        if rose_df.empty:
            st.info("No movement data for the selected filters.")
        else:
            bd = du.bin_bearings(rose_df["bearing"])
            fig = go.Figure(go.Barpolar(
                r=bd["r"], theta=bd["theta"],
                marker=dict(color=bd["r"], colorscale=[[0, "#1a237e"], [0.25, "#1565C0"], [0.5, "#00BCD4"],
                                                          [0.75, "#4CAF50"], [1, "#FF5252"]], showscale=True,
                             colorbar=dict(title="Fixes")),
            ))
            fig.update_layout(
                polar=dict(angularaxis=dict(tickmode="array", tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                                              ticktext=["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
                                              direction="clockwise", rotation=90),
                            radialaxis=dict(gridcolor="#e5e5e5")),
                showlegend=False, margin=dict(l=60, r=60, t=40, b=40), height=600,
            )
            st.plotly_chart(fig, use_container_width=True, key="mcp_rose_population")
        card_close()

    card_open("Per-Elephant Summary")
    if summary_df.empty:
        st.info("No data for the selected filters.")
    else:
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    card_close()


# ══════════════════════════════════════════════════════════════════════════
# PAGE DISPATCH
# ══════════════════════════════════════════════════════════════════════════
PAGES = {
    "Latitude vs Time": render_lat_tab,
    "Longitude vs Time": render_lon_tab,
    "Both Coordinates": render_both_tab,
    "Heat Maps": render_heat_tab,
    "Elephant Tracking": render_tracking_tab,
    "Live Elephant Path": render_live_tab,
    "Migration & Climate": render_climate_tab,
    "Data Table": render_data_tab,
    "Home Range & Speed": render_mcp_tab,
    "Dona & Recollared": render_dona_recollared_tab,
    "Density & Climate": render_density_climate_tab,
    "Day / Night": render_day_night_tab,
    "Vegetation Tracking": render_vegetation_tab,
    "Tracking": render_tracking_general_tab,
}

PAGES[nav]()
