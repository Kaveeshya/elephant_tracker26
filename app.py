"""
Kaudulla Elephant Tracker — Streamlit port of the original R Shiny app.
Run with:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import folium
from folium.plugins import Fullscreen
from streamlit_folium import st_folium
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

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
         "Data Table", "Home Range & Speed"],
        icons=["graph-up", "graph-up", "layers", "fire", "map", "play-circle",
               "globe-americas", "table", "compass"],
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
    st_folium(m, height=420, use_container_width=True, key=f"map_{coord_col}_{nav}")


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
    st_folium(m, height=420, use_container_width=True, key="ref_map")


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


@st.cache_resource(show_spinner="Building per-elephant grid…")
def _build_tracking_facet_fig():
    n_col = min(4, len(ALL_NAMES))
    n_row = int(np.ceil(len(ALL_NAMES) / n_col))
    sex_colors = {"Male": "darkblue", "Female": "darkred"}
    fig2, axes = plt.subplots(n_row, n_col, figsize=(4 * n_col, 3.6 * n_row), squeeze=False)
    for i, el in enumerate(ALL_NAMES):
        ax = axes[i // n_col][i % n_col]
        sub = elephants_df[elephants_df["name"] == el]
        for sex, grp in sub.groupby("sex"):
            ax.scatter(grp["lon"], grp["lat"], s=6, alpha=0.5, color=sex_colors.get(sex, "gray"), label=sex)
        ax.set_title(el, fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
    for j in range(len(ALL_NAMES), n_row * n_col):
        axes[j // n_col][j % n_col].axis("off")
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, label=s, markersize=8)
               for s, c in sex_colors.items()]
    fig2.legend(handles=handles, loc="lower center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig2.suptitle("GPS Tracking Data by Elephant", fontsize=18, fontweight="bold", y=1.01)
    fig2.tight_layout()
    return fig2


def render_tracking_tab():
    st.markdown("<div class='section-title'>🗺️ Elephant Tracking Overview</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Interactive Map with Satellite View</div>", unsafe_allow_html=True)

    st_folium(_build_tracking_map(), height=600, use_container_width=True, key="tracking_map")

    st.markdown("<div class='sub-title'>📊 Tracking Data Visualization</div>", unsafe_allow_html=True)
    st.pyplot(_build_tracking_scatter_fig())

    st.markdown("<div class='sub-title'>👥 GPS Tracking Data by Individual Elephant</div>", unsafe_allow_html=True)
    st.pyplot(_build_tracking_facet_fig())


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
def render_climate_tab():
    st.markdown("<div class='section-title'>🐘 Elephant Tracking Data Availability</div>", unsafe_allow_html=True)
    c1, c2 = st.columns([3, 9])
    with c1:
        card_open()
        avail_elephant = st.selectbox("Select Elephant Name:", ALL_NAMES, key="avail_elephant")
        st.caption("This heatmap shows the percentage of valid GPS records captured per day (max 24 records/day).")
        card_close()
    with c2:
        card_open()
        sub = elephants_df[elephants_df["name"] == avail_elephant].copy()
        # elephants_df already drops rows with NaN lat/lon, so counting rows per
        # day gives the number of valid GPS fixes captured that day.
        sub["day"] = sub["datetime_sl"].dt.date
        daily = sub.groupby("day").size().reset_index(name="valid_records").rename(columns={"day": "date"})
        daily["availability"] = np.minimum(100, 100 * daily["valid_records"] / 24)
        if daily.empty:
            st.info("No data available for the selected elephant.")
        else:
            fig = make_calendar_heatmap(
                daily["date"], daily["availability"],
                title=f"Daily GPS Availability Calendar Heatmap – {avail_elephant}",
                discrete_breaks=du.AVAILABILITY_BREAKS, discrete_labels=du.AVAILABILITY_LABELS,
                discrete_colors=du.AVAILABILITY_COLORS,
            )
            st.plotly_chart(fig, use_container_width=True, key="calendar_plot")
        card_close()

    st.markdown("<div class='section-title'>🐘 Elephant Migration Map</div>", unsafe_allow_html=True)
    c1, c2 = st.columns([3, 9])
    with c1:
        card_open()
        years_avail = sorted(elephants_df["year"].unique())
        mig_year = st.selectbox("Select Year", years_avail, key="mig_year")
        mig_month = st.selectbox("Select Month", [f"{m:02d}" for m in range(1, 13)], key="mig_month")
        mig_elephant = st.selectbox("Select Elephant", ALL_NAMES, key="mig_elephant")
        card_close()
    with c2:
        card_open()
        dat = elephants_df[
            (elephants_df["year"] == mig_year) & (elephants_df["month"] == mig_month) &
            (elephants_df["name"] == mig_elephant)
        ]
        if dat.empty:
            m = folium.Map(location=[7.0, 80.0], zoom_start=7, tiles="OpenStreetMap", prefer_canvas=True)
            folium.Marker([7.0, 80.0], popup="No elephant data available for selected year & month",
                          icon=folium.Icon(color="gray")).add_to(m)
        else:
            m = folium.Map(tiles="OpenStreetMap", prefer_canvas=True)
            year_months = sorted(elephants_df["year_month"].unique())
            palette = px.colors.qualitative.Dark24
            ym_colors = {ym: palette[i % len(palette)] for i, ym in enumerate(year_months)}
            pts = []
            for _, r in dat.sort_values("datetime_sl").iterrows():
                clr = ym_colors.get(r["year_month"], "#333333")
                folium.CircleMarker(
                    [r["lat"], r["lon"]], radius=5, color=clr, fill=True, fill_color=clr, fill_opacity=1, weight=0,
                    popup=f"<b>Elephant:</b> {r['name']}<br><b>Date:</b> {r['datetime']}<br>"
                          f"<b>Year:</b> {r['year']}<br><b>Month:</b> {r['month']}",
                ).add_to(m)
                pts.append((r["lat"], r["lon"]))
            m.fit_bounds(pts)
        st_folium(m, height=600, use_container_width=True, key="migration_map")
        card_close()

    st.markdown("<div class='section-title'>🌡 Climate Calendar Analysis</div>", unsafe_allow_html=True)
    c1, c2 = st.columns([3, 9])
    with c1:
        card_open()
        clim_var = st.selectbox("Climate Variable", list(du.CLIMATE_PLOT_INFO.keys()), key="clim_var")
        info = du.CLIMATE_PLOT_INFO[clim_var]
        vals = climate_df[info["col"]].astype(float)
        st.markdown("#### Summary Statistics")
        stats_df = pd.DataFrame({
            "Measure": ["Minimum", "First Quantile (Q1)", "Median", "Mean", "Third Quantile (Q3)",
                        "Maximum", "Standard Deviation", "Missing Values (NA)"],
            "Value": [vals.min(), vals.quantile(0.25), vals.median(), vals.mean(), vals.quantile(0.75),
                      vals.max(), vals.std(), vals.isna().sum()],
        })
        stats_df["Value"] = stats_df["Value"].round(2)
        st.dataframe(stats_df, hide_index=True, use_container_width=True)
        card_close()
    with c2:
        card_open()
        fig = make_calendar_heatmap(
            climate_df["date"], vals, title=info["title"],
            discrete_breaks=info["breaks"], discrete_labels=info["labels"], discrete_colors=du.CALENDAR_COLORS,
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True, key="climate_calendar_plot")
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
        h = du.compute_hull(edata, ratio=0.3)
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

    card_open("Tracking Points & Minimum Convex Polygons")
    m = folium.Map(location=[mcp_df["lat"].mean(), mcp_df["lon"].mean()], zoom_start=12, control_scale=True, prefer_canvas=True)
    folium.TileLayer("CartoDB positron", name="Light").add_to(m)
    folium.TileLayer("OpenStreetMap", name="Street").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satellite",
    ).add_to(m)
    for el in elephants_present:
        edata = df[df["name"] == el]
        color = mcp_colors[el]
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
        color = mcp_colors[el]
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
    _add_categorical_legend(m, "Elephant", {el: mcp_colors[el] for el in elephants_present})
    st_folium(m, height=600, use_container_width=True, key="mcp_hull_map")
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
}

PAGES[nav]()
