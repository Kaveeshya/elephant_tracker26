"""
calendar_heatmap.py
Clean, visually appealing Plotly calendar heatmap — better than R's calendarHeat().
One row per year, weeks across x-axis, days-of-week down y-axis.
Uses SVG shapes for month borders only — no backing layer, clean white cells.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _bin_values(values, breaks, n_bins):
    values = pd.Series(values).astype(float)
    bins = pd.cut(values, bins=breaks, labels=False, include_lowest=True, right=True)
    result = bins.copy()
    result[values.isna()] = -1
    return result.fillna(-1).astype(int)


def _year_grid(dates, bin_values, year):
    start    = pd.Timestamp(year, 1, 1)
    end      = pd.Timestamp(year, 12, 31)
    all_days = pd.date_range(start, end, freq="D")

    # Sunday = 0, Monday = 1 ... Saturday = 6  (same as R %w)
    dow  = all_days.dayofweek.map({0:1,1:2,2:3,3:4,4:5,5:6,6:0})
    woty = ((all_days - start).days + int(start.strftime("%w"))) // 7
    n_cols = int(woty.max()) + 1

    z     = np.full((7, n_cols), np.nan)
    hover = np.empty((7, n_cols), dtype=object)

    val_map = {}
    for d, bv in zip(pd.to_datetime(pd.Series(dates)).values,
                     pd.Series(bin_values).values):
        val_map[pd.Timestamp(d).normalize()] = int(bv)

    for day, row, col in zip(all_days, dow, woty):
        bv = val_map.get(day.normalize(), -1)
        z[row, col]     = np.nan if bv < 0 else bv
        hover[row, col] = day.strftime("%d %b %Y")

    month_cols, month_labels = [], []
    for mo in range(1, 13):
        first = pd.Timestamp(year, mo, 1)
        if first > end: break
        col_idx = ((first - start).days + int(start.strftime("%w"))) // 7
        month_cols.append(col_idx)
        month_labels.append(first.strftime("%b"))

    return z, hover, month_cols, month_labels, n_cols


def _month_borders(year, n_cols, x_domain, y_domain):
    start    = pd.Timestamp(year, 1, 1)
    end      = pd.Timestamp(year, 12, 31)
    all_days = pd.date_range(start, end, freq="D")
    dow      = all_days.dayofweek.map({0:1,1:2,2:3,3:4,4:5,5:6,6:0})

    x0, x1 = x_domain
    y0, y1 = y_domain

    def px(col): return x0 + col / n_cols * (x1 - x0)
    def py(row): return y1 - row / 7   * (y1 - y0)

    shapes = []
    def seg(xa, ya, xb, yb):
        shapes.append(dict(type="line", xref="paper", yref="paper",
                           x0=xa, y0=ya, x1=xb, y1=yb,
                           line=dict(color="#333333", width=1.8)))

    # Outer border
    seg(px(0), py(0), px(n_cols), py(0))
    seg(px(0), py(7), px(n_cols), py(7))
    seg(px(0), py(0), px(0),      py(7))
    seg(px(n_cols), py(0), px(n_cols), py(7))

    for mo in range(1, 12):
        try:
            fn = pd.Timestamp(year, mo + 1, 1)
        except Exception:
            break
        if fn > end: break
        day_offset   = (fn - start).days
        col_of_first = (day_offset + int(start.strftime("%w"))) // 7
        row_of_first = int(dow[day_offset])

        if row_of_first == 0:
            seg(px(col_of_first), py(0), px(col_of_first), py(7))
        else:
            seg(px(col_of_first),     py(row_of_first), px(col_of_first),     py(7))
            seg(px(col_of_first),     py(row_of_first), px(col_of_first + 1), py(row_of_first))
            seg(px(col_of_first + 1), py(0),            px(col_of_first + 1), py(row_of_first))

    return shapes


def make_calendar_heatmap(
    dates, values,
    title="",
    discrete_breaks=None,
    discrete_labels=None,
    discrete_colors=None,
    colorscale=None,
    height=None,
):
    dates  = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    values = pd.Series(values).reset_index(drop=True)
    years  = sorted(dates.dt.year.unique())
    n_years = len(years)

    # ── colour setup ──────────────────────────────────────────────────────────
    if discrete_breaks is not None:
        colors = list(discrete_colors or ["#f1faee","#a8dadc","#457b9d","#1d3557"])
        n_bins = len(discrete_breaks) - 1
        if len(colors) < n_bins:
            colors = (colors * (n_bins // len(colors) + 1))[:n_bins]
        cs = []
        for i, c in enumerate(colors):
            cs.append([i / n_bins, c])
            cs.append([(i + 1) / n_bins, c])
        bin_vals = _bin_values(values, discrete_breaks, n_bins)
        zmin, zmax = 0, n_bins - 1
    else:
        colors   = None
        n_bins   = None
        bin_vals = values.astype(float)
        cs       = colorscale or "YlOrRd"
        zmin     = float(np.nanmin(values)) if len(values) else 0
        zmax     = float(np.nanmax(values)) if len(values) else 1

    # ── layout constants ──────────────────────────────────────────────────────
    LEFT_MARGIN  = 0.11   # space for day labels
    RIGHT_MARGIN = 0.85 if discrete_breaks else 0.97
    ROW_FRAC     = 1.0 / n_years
    TOP_PAD      = 0.14   # fraction of row height for year label + month labels
    BOT_PAD      = 0.05

    fig    = go.Figure()
    shapes = []

    for i, yr in enumerate(years):
        mask     = dates.dt.year == yr
        yr_bins  = bin_vals[mask].values if discrete_breaks else values[mask].astype(float).values
        yr_dates = dates[mask].values

        z, hover, month_cols, month_labels, n_cols = _year_grid(yr_dates, yr_bins, yr)

        # Paper-space y extents for this strip
        strip_top = 1.0 - i * ROW_FRAC
        strip_bot = 1.0 - (i + 1) * ROW_FRAC
        cell_top  = strip_top - ROW_FRAC * TOP_PAD
        cell_bot  = strip_bot + ROW_FRAC * BOT_PAD

        trace_x  = "x"      if i == 0 else f"x{i+1}"
        trace_y  = "y"      if i == 0 else f"y{i+1}"
        layout_x = "xaxis"  if i == 0 else f"xaxis{i+1}"
        layout_y = "yaxis"  if i == 0 else f"yaxis{i+1}"

        # ── heatmap ───────────────────────────────────────────────────────────
        fig.add_trace(go.Heatmap(
            z=z, text=hover, hoverinfo="text",
            colorscale=cs, zmin=zmin, zmax=zmax,
            showscale=False,
            xgap=3, ygap=3,          # clean white gaps between cells
            x0=0, dx=1, y0=0, dy=1,
            xaxis=trace_x, yaxis=trace_y,
        ))

        # ── axes ──────────────────────────────────────────────────────────────
        fig.update_layout(**{
            layout_x: dict(
                domain=[LEFT_MARGIN, RIGHT_MARGIN],
                anchor=trace_y,
                tickmode="array", tickvals=month_cols, ticktext=month_labels,
                showgrid=False, side="bottom", zeroline=False, showline=False,
                tickfont=dict(size=12, color="#444444", family="Segoe UI"),
                tickangle=0,
            ),
            layout_y: dict(
                domain=[cell_bot, cell_top],
                anchor=trace_x,
                tickmode="array",
                tickvals=[0, 1, 2, 3, 4, 5, 6],
                ticktext=["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
                autorange="reversed",
                showgrid=False, zeroline=False, showline=False,
                tickfont=dict(size=11, color="#555555", family="Segoe UI"),
            ),
        })

        # ── year label centred above strip ────────────────────────────────────
        fig.add_annotation(
            x=(LEFT_MARGIN + RIGHT_MARGIN) / 2,
            y=cell_top + ROW_FRAC * 0.07,
            xref="paper", yref="paper",
            text=f"<b>{yr}</b>",
            showarrow=False, xanchor="center",
            font=dict(size=14, color="#1e293b", family="Segoe UI"),
        )

        # ── month boundary shapes ─────────────────────────────────────────────
        shapes += _month_borders(
            yr, n_cols,
            x_domain=(LEFT_MARGIN, RIGHT_MARGIN),
            y_domain=(cell_bot, cell_top),
        )

    # ── discrete legend — clean stacked boxes ────────────────────────────────
    if discrete_breaks is not None and discrete_labels is not None:
        leg_x   = RIGHT_MARGIN + 0.025
        box_w   = 0.038
        box_h   = 0.030
        gap     = 0.060
        start_y = 0.94

        for j, (lab, col) in enumerate(zip(discrete_labels, colors)):
            y_mid = start_y - j * gap
            fig.add_shape(
                type="rect", xref="paper", yref="paper",
                x0=leg_x,        x1=leg_x + box_w,
                y0=y_mid - box_h/2, y1=y_mid + box_h/2,
                fillcolor=col,
                line=dict(color="#555555", width=0.8),
            )
            fig.add_annotation(
                x=leg_x + box_w + 0.010, y=y_mid,
                xref="paper", yref="paper",
                text=f"<b>{lab}</b>", showarrow=False,
                xanchor="left",
                font=dict(size=11, color="#1e293b", family="Segoe UI"),
            )

    # ── global layout ─────────────────────────────────────────────────────────
    row_px = 210
    fig.update_layout(
        title=dict(
            text=title, x=0.5, xanchor="center",
            font=dict(size=15, color="#1e293b", family="Segoe UI"),
        ),
        height=height or max(row_px * n_years + 60, 280),
        margin=dict(t=55, b=35, l=10, r=175 if discrete_breaks else 20),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        shapes=shapes,
    )
    return fig
