"""
calendar_heatmap.py  —  matplotlib-based calendar heatmap
Replicates R's calendarHeat() pixel-perfectly using matplotlib patches.
Returns a Plotly Figure wrapping the matplotlib PNG so it fits Streamlit's
st.plotly_chart() call in app.py without any other changes.
"""

import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import plotly.graph_objects as go


DAY_LABELS = ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"]
CELL       = 1.0
GAP        = 0.08


def _year_grid(dates, values, year):
    start    = pd.Timestamp(year, 1, 1)
    end      = pd.Timestamp(year, 12, 31)
    all_days = pd.date_range(start, end, freq="D")
    dow_map  = {0:1,1:2,2:3,3:4,4:5,5:6,6:0}
    dow      = np.array([dow_map[d] for d in all_days.dayofweek])
    woty     = ((all_days - start).days + int(start.strftime("%w"))) // 7
    n_cols   = int(woty.max()) + 1

    z       = np.full((7, n_cols), np.nan)
    val_map = {pd.Timestamp(d).normalize(): float(v)
               for d, v in zip(pd.to_datetime(pd.Series(dates)).values,
                               pd.Series(values).values)}
    for day, row, col in zip(all_days, dow, woty):
        z[row, col] = val_map.get(day.normalize(), np.nan)

    month_info = []
    for mo in range(1, 13):
        first = pd.Timestamp(year, mo, 1)
        if first > end: break
        col_idx = ((first - start).days + int(start.strftime("%w"))) // 7
        month_info.append((col_idx, first.strftime("%b")))

    return z, month_info, n_cols, dow, all_days, start


def _bin_color(value, breaks, colors):
    if np.isnan(value):
        return None
    for i in range(len(breaks) - 1):
        if breaks[i] <= value <= breaks[i + 1]:
            return colors[i]
    return colors[-1]


def make_calendar_heatmap(
    dates, values,
    title="",
    discrete_breaks=None,
    discrete_labels=None,
    discrete_colors=None,
    colorscale=None,
    height=None,
):
    dates   = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    values  = pd.Series(values).astype(float).reset_index(drop=True)
    years   = sorted(dates.dt.year.unique())
    n_years = len(years)

    strip_h = 2.6
    fig_h   = n_years * strip_h + 0.6
    fig_w   = 16

    fig, axes = plt.subplots(n_years, 1, figsize=(fig_w, fig_h), facecolor="white")
    if n_years == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=13, fontweight="bold", color="#1e293b", y=1.002)

    for ax, yr in zip(axes, years):
        mask     = dates.dt.year == yr
        yr_vals  = values[mask].values
        yr_dates = dates[mask].values

        z, month_info, n_cols, dow_arr, all_days, start = _year_grid(yr_dates, yr_vals, yr)

        ax.set_xlim(-0.5, n_cols)
        ax.set_ylim(-0.5, 7)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.axis("off")

        # ── cells ─────────────────────────────────────────────────────────────
        vmin = np.nanmin(z) if not np.all(np.isnan(z)) else 0
        vmax = np.nanmax(z) if not np.all(np.isnan(z)) else 1
        cmap = plt.cm.YlOrRd

        for row in range(7):
            for col in range(n_cols):
                v = z[row, col]
                if np.isnan(v):
                    continue
                if discrete_breaks is not None:
                    fill = _bin_color(v, discrete_breaks, discrete_colors)
                else:
                    t    = (v - vmin) / (vmax - vmin + 1e-9)
                    fill = cmap(t)
                if fill is None:
                    continue
                ax.add_patch(Rectangle(
                    (col - 0.5 + GAP/2, row - 0.5 + GAP/2),
                    CELL - GAP, CELL - GAP,
                    linewidth=0, facecolor=fill,
                ))

        # ── outer border ──────────────────────────────────────────────────────
        ax.add_patch(Rectangle((-0.5, -0.5), n_cols, 7,
                               linewidth=1.5, edgecolor="#222", facecolor="none"))

        # ── month boundary staircases ─────────────────────────────────────────
        dow_map = {0:1,1:2,2:3,3:4,4:5,5:6,6:0}
        end     = pd.Timestamp(yr, 12, 31)

        for mo in range(1, 12):
            try:
                fn = pd.Timestamp(yr, mo + 1, 1)
            except Exception:
                break
            if fn > end: break
            day_offset   = (fn - start).days
            col_of_first = (day_offset + int(start.strftime("%w"))) // 7
            row_of_first = dow_map[all_days[day_offset].dayofweek]

            if row_of_first == 0:
                ax.plot([col_of_first - 0.5, col_of_first - 0.5],
                        [-0.5, 6.5], color="#222", lw=1.5)
            else:
                xs = [col_of_first - 0.5, col_of_first - 0.5,
                      col_of_first + 0.5, col_of_first + 0.5]
                ys = [6.5, row_of_first - 0.5,
                      row_of_first - 0.5, -0.5]
                ax.plot(xs, ys, color="#222", lw=1.5)

        # ── month labels ──────────────────────────────────────────────────────
        for col_idx, label in month_info:
            ax.text(col_idx - 0.5, -0.75, label,
                    fontsize=9, color="#444", va="top", ha="left")

        # ── day labels ────────────────────────────────────────────────────────
        for row, dl in enumerate(DAY_LABELS):
            ax.text(-0.6, row, dl, fontsize=9,
                    color="#555", va="center", ha="right")

        # ── year label ────────────────────────────────────────────────────────
        ax.text(n_cols / 2 - 0.5, -1.35, str(yr),
                fontsize=11, fontweight="bold",
                color="#1e293b", va="top", ha="center")

    # ── legend ────────────────────────────────────────────────────────────────
    if discrete_breaks is not None and discrete_labels is not None:
        handles = [
            mpatches.Patch(facecolor=c, edgecolor="#666",
                           linewidth=0.8, label=l)
            for c, l in zip(discrete_colors, discrete_labels)
        ]
        fig.legend(
            handles=handles,
            loc="center right",
            bbox_to_anchor=(1.0, 0.5),
            frameon=False,
            fontsize=10,
            handlelength=1.4,
            handleheight=1.4,
        )

    plt.tight_layout(rect=[0, 0, 0.92 if discrete_breaks else 1.0, 1.0])

    # ── export PNG → base64 → Plotly Figure ──────────────────────────────────
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    src = f"data:image/png;base64,{b64}"

    pfig = go.Figure()
    pfig.add_layout_image(
        source=src,
        xref="paper", yref="paper",
        x=0, y=1, sizex=1, sizey=1,
        xanchor="left", yanchor="top",
        sizing="stretch", layer="above",
    )
    h = height or max(220 * n_years, 260)
    pfig.update_layout(
        margin=dict(t=0, b=0, l=0, r=0),
        height=h,
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1]),
    )
    return pfig
