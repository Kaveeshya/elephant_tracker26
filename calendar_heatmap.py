"""
calendar_heatmap.py
Plotly calendar heatmap that replicates R's calendarHeat() (lattice/grid) output:
  - One row per year, weeks across x-axis, days-of-week down y-axis
  - Discrete colour bins with a legend matching the R app exactly
  - Thick BLACK month-boundary lines drawn as Plotly shapes (not as cell gaps)
  - Sun=row 0 … Sat=row 6  (matching R's %w convention)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ── helpers ──────────────────────────────────────────────────────────────────

def _bin_values(values, breaks, n_bins):
    """Map continuous values → integer bin index [0 .. n_bins-1]."""
    values = pd.Series(values).astype(float)
    bins = pd.cut(values, bins=breaks, labels=False, include_lowest=True)
    return bins.fillna(-1).astype(int)          # -1 = no data


def _year_grid(dates, bin_values, year):
    """
    Build the 7-row × 54-col grid for one year.
    Returns z (int array, NaN = no day), hover (str array),
    month_tick_cols, month_tick_labels.
    """
    start   = pd.Timestamp(year, 1, 1)
    end     = pd.Timestamp(year, 12, 31)
    all_days = pd.date_range(start, end, freq="D")

    # %w: Sunday=0 … Saturday=6  (same as R)
    dow  = all_days.dayofweek.map({0:1,1:2,2:3,3:4,4:5,5:6,6:0})   # Mon=1..Sun=0→row
    # week column: Sunday-starting ISO-like week number
    woty = ((all_days - start).days + int(start.strftime("%w"))) // 7

    n_cols = int(woty.max()) + 1
    z      = np.full((7, n_cols), np.nan)
    hover  = np.empty((7, n_cols), dtype=object)

    val_map = {}
    for d, bv in zip(pd.to_datetime(pd.Series(dates)).values,
                     pd.Series(bin_values).values):
        val_map[pd.Timestamp(d).normalize()] = int(bv)

    for day, row, col in zip(all_days, dow, woty):
        bv = val_map.get(day.normalize(), -1)
        z[row, col]     = np.nan if bv == -1 else bv
        hover[row, col] = day.strftime("%d %b %Y")

    # Month tick positions: first column of each month
    month_cols, month_labels = [], []
    for mo in range(1, 13):
        first = pd.Timestamp(year, mo, 1)
        if first > end: break
        col_idx = ((first - start).days + int(start.strftime("%w"))) // 7
        month_cols.append(col_idx)
        month_labels.append(first.strftime("%b"))

    return z, hover, month_cols, month_labels, n_cols


def _month_border_shapes(year, z, xdomain, ydomain):
    """
    Return Plotly shape dicts that draw thick black month boundaries
    matching R's calendarHeat() grid lines exactly.
    Coordinates are in *paper* space (0-1) so they overlay the heatmap cell axes.
    """
    start    = pd.Timestamp(year, 1, 1)
    end      = pd.Timestamp(year, 12, 31)
    all_days = pd.date_range(start, end, freq="D")

    dow  = all_days.dayofweek.map({0:1,1:2,2:3,3:4,4:5,5:6,6:0})
    woty = ((all_days - start).days + int(start.strftime("%w"))) // 7
    n_cols = int(woty.max()) + 1

    # Build {date → (row, col)} lookup
    cell = {day.normalize(): (int(r), int(c))
            for day, r, c in zip(all_days, dow, woty)}

    def frac_x(col):   return xdomain[0] + (col + 0.5) / n_cols * (xdomain[1] - xdomain[0])
    def frac_y(row):   return ydomain[0] + (6.5 - row) / 7   * (ydomain[1] - ydomain[0])

    shapes = []

    def seg(x0, y0, x1, y1):
        shapes.append(dict(
            type="line", xref="paper", yref="paper",
            x0=x0, y0=y0, x1=x1, y1=y1,
            line=dict(color="black", width=1.8),
        ))

    # Outer border
    seg(xdomain[0], ydomain[0], xdomain[1], ydomain[0])
    seg(xdomain[0], ydomain[1], xdomain[1], ydomain[1])
    seg(xdomain[0], ydomain[0], xdomain[0], ydomain[1])
    seg(xdomain[1], ydomain[0], xdomain[1], ydomain[1])

    # Month-boundary vertical (and partial) lines
    for mo in range(1, 13):
        first = pd.Timestamp(year, mo, 1)
        if first > end:
            break
        r0, c0 = cell[first.normalize()]

        # Top border of this month's first partial column (rows 0..r0-1 belong to prev month)
        if r0 > 0:
            # Vertical line at left edge of c0 from top down to row r0
            seg(frac_x(c0) - 0.5/n_cols*(xdomain[1]-xdomain[0])*2,
                frac_y(-0.5),
                frac_x(c0) - 0.5/n_cols*(xdomain[1]-xdomain[0])*2,
                frac_y(r0 - 0.5))
            # Horizontal line at top of row r0 from c0 to c0+1
            seg(frac_x(c0) - 0.5/n_cols*(xdomain[1]-xdomain[0])*2,
                frac_y(r0 - 0.5),
                frac_x(c0) + 0.5/n_cols*(xdomain[1]-xdomain[0])*2,
                frac_y(r0 - 0.5))

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
    """
    Public API — called from app.py.

    discrete_breaks / discrete_labels / discrete_colors control the
    binned colour scheme (matching R's calendarHeat `at` / `colors`).
    If these are None, a continuous YlOrRd scale is used instead.
    """
    dates  = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    values = pd.Series(values).reset_index(drop=True)
    years  = sorted(dates.dt.year.unique())
    n_years = len(years)

    # ── colour setup ──────────────────────────────────────────────────────────
    if discrete_breaks is not None:
        colors  = list(discrete_colors or ["#f1faee","#a8dadc","#457b9d","#1d3557"])
        n_bins  = len(discrete_breaks) - 1
        if len(colors) < n_bins:
            colors = (colors * (n_bins // len(colors) + 1))[:n_bins]

        # Build a step colorscale: each bin gets its own solid colour
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

    # ── build figure ──────────────────────────────────────────────────────────
    fig    = go.Figure()
    row_h  = 1.0 / n_years
    shapes = []

    for i, yr in enumerate(years):
        mask = dates.dt.year == yr
        yr_bins  = bin_vals[mask].values if discrete_breaks else values[mask].astype(float).values
        yr_dates = dates[mask].values

        z, hover, month_cols, month_labels, n_cols = _year_grid(yr_dates, yr_bins, yr)

        # y-domain for this year's strip (top year first)
        y0 = 1.0 - (i + 1) * row_h
        y1 = 1.0 - i * row_h
        inner_y0 = y0 + row_h * 0.08
        inner_y1 = y1 - row_h * 0.10
        x0_dom, x1_dom = 0.06, 0.88 if discrete_breaks else 0.95

        # Trace-level axis refs: "x","y" for the first subplot, "x2","y2" for
        # the second, etc. — there is no "x1"/"y1", that's a Plotly gotcha.
        trace_x = "x" if i == 0 else f"x{i+1}"
        trace_y = "y" if i == 0 else f"y{i+1}"
        # update_layout() keys use the *long* form instead: "xaxis"/"yaxis"
        # for the first subplot, "xaxis2"/"yaxis2" for the second, etc.
        layout_x = "xaxis" if i == 0 else f"xaxis{i+1}"
        layout_y = "yaxis" if i == 0 else f"yaxis{i+1}"

        # ── black backing layer (gives cells a thin black border) ────────────
        z_back = np.where(np.isnan(z), np.nan, 0.0)
        fig.add_trace(go.Heatmap(
            z=z_back,
            colorscale=[[0,"black"],[1,"black"]],
            zmin=0, zmax=1, showscale=False, hoverinfo="skip",
            xgap=0, ygap=0, x0=0, dx=1, y0=0, dy=1,
            xaxis=trace_x, yaxis=trace_y,
        ))

        # ── colour layer ──────────────────────────────────────────────────────
        fig.add_trace(go.Heatmap(
            z=z,
            text=hover, hoverinfo="text",
            colorscale=cs, zmin=zmin, zmax=zmax,
            showscale=False,              # legend built manually below
            xgap=1.5, ygap=1.5,
            x0=0, dx=1, y0=0, dy=1,
            xaxis=trace_x, yaxis=trace_y,
        ))

        # ── axis layout ───────────────────────────────────────────────────────
        fig.update_layout(**{
            layout_x: dict(
                domain=[x0_dom, x1_dom], anchor=trace_y,
                tickmode="array", tickvals=month_cols, ticktext=month_labels,
                showgrid=False, side="bottom", zeroline=False,
                showline=False,
            ),
            layout_y: dict(
                domain=[inner_y0, inner_y1], anchor=trace_x,
                tickmode="array",
                tickvals=[0,1,2,3,4,5,6],
                ticktext=["Sun","Mon","Tue","Wed","Thu","Fri","Sat"],
                autorange="reversed", showgrid=False,
                zeroline=False, showline=False,
            ),
        })

        # ── year label on the left ────────────────────────────────────────────
        fig.add_annotation(
            x=0.0, y=(inner_y0 + inner_y1) / 2,
            xref="paper", yref="paper",
            text=f"<b>{yr}</b>", showarrow=False,
            xanchor="right", font=dict(size=13, color="#1e293b"),
        )

        # ── month boundary shapes ─────────────────────────────────────────────
        shapes += _month_borders(yr, z, n_cols,
                                 x_domain=(x0_dom, x1_dom),
                                 y_domain=(inner_y0, inner_y1))

    # ── discrete legend (right margin) ───────────────────────────────────────
    if discrete_breaks is not None and discrete_labels is not None:
        legend_x = 0.91
        n_lab    = len(discrete_labels)
        for j, (lab, col) in enumerate(zip(discrete_labels, colors)):
            ypos = 0.95 - j * (0.85 / max(n_lab - 1, 1))
            # coloured square
            fig.add_shape(
                type="rect",
                xref="paper", yref="paper",
                x0=legend_x, x1=legend_x + 0.035,
                y0=ypos - 0.025, y1=ypos + 0.025,
                fillcolor=col, line=dict(color="black", width=0.8),
            )
            fig.add_annotation(
                x=legend_x + 0.042, y=ypos,
                xref="paper", yref="paper",
                text=lab, showarrow=False,
                xanchor="left", font=dict(size=11, color="#1e293b"),
            )

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=14, color="#1e293b")),
        height=height or max(200 * n_years, 240),
        margin=dict(t=50, b=30, l=55, r=150 if discrete_breaks else 30),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#1e293b", family="Segoe UI"),
        shapes=shapes,
    )
    return fig


# ── month borders (proper R-style) ────────────────────────────────────────────

def _month_borders(year, z, n_cols, x_domain, y_domain):
    """
    Compute the thick black month-boundary polyline segments in paper coords,
    replicating R calendarHeat()'s grid.lines() calls exactly.
    Each month boundary is a staircase line that steps between columns.
    """
    start    = pd.Timestamp(year, 1, 1)
    end      = pd.Timestamp(year, 12, 31)
    all_days = pd.date_range(start, end, freq="D")

    dow  = all_days.dayofweek.map({0:1,1:2,2:3,3:4,4:5,5:6,6:0})
    woty = ((all_days - start).days + int(start.strftime("%w"))) // 7

    x0, x1 = x_domain
    y0, y1 = y_domain

    def px(col):
        """left edge of column `col` in paper coords"""
        return x0 + col / n_cols * (x1 - x0)

    def py(row):
        """top edge of row `row` in paper coords (row 0 = top = Sunday)"""
        return y1 - row / 7 * (y1 - y0)

    shapes = []

    def seg(xa, ya, xb, yb):
        shapes.append(dict(
            type="line", xref="paper", yref="paper",
            x0=xa, y0=ya, x1=xb, y1=yb,
            line=dict(color="black", width=1.6),
        ))

    # Outer box
    seg(px(0), py(0), px(n_cols), py(0))   # top
    seg(px(0), py(7), px(n_cols), py(7))   # bottom
    seg(px(0), py(0), px(0),      py(7))   # left
    seg(px(n_cols), py(0), px(n_cols), py(7))  # right

    # One staircase per month boundary (between month m and m+1)
    for mo in range(1, 12):
        # First day of next month
        try:
            first_next = pd.Timestamp(year, mo + 1, 1)
        except Exception:
            break
        if first_next > end:
            break

        day_offset = (first_next - start).days
        col_of_first = (day_offset + int(start.strftime("%w"))) // 7
        row_of_first = int(all_days.dayofweek.map({0:1,1:2,2:3,3:4,4:5,5:6,6:0})[day_offset])

        if row_of_first == 0:
            # Boundary falls at start of a column — simple vertical line
            seg(px(col_of_first), py(0), px(col_of_first), py(7))
        else:
            # Staircase: vertical up to the split row, horizontal, then vertical down
            seg(px(col_of_first),     py(row_of_first), px(col_of_first),     py(7))
            seg(px(col_of_first),     py(row_of_first), px(col_of_first + 1), py(row_of_first))
            seg(px(col_of_first + 1), py(0),            px(col_of_first + 1), py(row_of_first))

    return shapes
