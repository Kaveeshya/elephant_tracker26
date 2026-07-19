"""
calendar_heatmap.py  —  SVG-based calendar heatmap
Draws every cell, border and legend element as raw SVG — no Plotly Heatmap traces.
This gives pixel-perfect control over cell size, gaps and month-border lines,
producing output visually identical to R's calendarHeat().
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ── grid builder ─────────────────────────────────────────────────────────────

def _year_grid(dates, values, year):
    """Return (7-row × n_col z-array, hover-array, month_tick_cols, month_tick_labels, n_cols)."""
    start    = pd.Timestamp(year, 1, 1)
    end      = pd.Timestamp(year, 12, 31)
    all_days = pd.date_range(start, end, freq="D")

    dow  = all_days.dayofweek.map({0:1,1:2,2:3,3:4,4:5,5:6,6:0})   # Sun=0…Sat=6
    woty = ((all_days - start).days + int(start.strftime("%w"))) // 7
    n_cols = int(woty.max()) + 1

    z     = np.full((7, n_cols), np.nan)
    hover = np.empty((7, n_cols), dtype=object)

    val_map = {pd.Timestamp(d).normalize(): float(v)
               for d, v in zip(pd.to_datetime(pd.Series(dates)).values,
                               pd.Series(values).values)}

    for day, row, col in zip(all_days, dow, woty):
        v = val_map.get(day.normalize(), np.nan)
        z[row, col]     = v
        hover[row, col] = day.strftime("%d %b %Y")

    month_cols, month_labels = [], []
    for mo in range(1, 13):
        first = pd.Timestamp(year, mo, 1)
        if first > end: break
        col_idx = ((first - start).days + int(start.strftime("%w"))) // 7
        month_cols.append(col_idx)
        month_labels.append(first.strftime("%b"))

    return z, hover, month_cols, month_labels, n_cols


# ── colour helpers ────────────────────────────────────────────────────────────

def _bin_color(value, breaks, colors):
    """Return the fill colour for a scalar value given discrete breaks+colors."""
    if np.isnan(value):
        return None                  # no cell for missing days
    for i in range(len(breaks) - 1):
        lo = breaks[i]
        hi = breaks[i + 1]
        if i == 0 and value <= hi:
            return colors[i]
        if lo < value <= hi:
            return colors[i]
    return colors[-1]                # catch the exact upper bound


def _continuous_color(value, vmin, vmax, colorscale_name="YlOrRd"):
    """Map a scalar to an RGB hex using a simple two-stop interpolation."""
    # Very simple built-in colour ramp for continuous use
    # Users can extend this if needed
    stops = {
        "YlOrRd": ["#FFFFCC","#FED976","#FD8D3C","#E31A1C","#800026"],
    }
    ramp = stops.get(colorscale_name, stops["YlOrRd"])
    if np.isnan(value) or vmax == vmin:
        return None
    t = max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
    idx = t * (len(ramp) - 1)
    lo  = int(idx)
    hi  = min(lo + 1, len(ramp) - 1)
    frac = idx - lo
    def _hex(h):
        h = h.lstrip("#")
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    r1, g1, b1 = _hex(ramp[lo])
    r2, g2, b2 = _hex(ramp[hi])
    r = int(r1 + frac*(r2-r1))
    g = int(g1 + frac*(g2-g1))
    b = int(b1 + frac*(b2-b1))
    return f"#{r:02x}{g:02x}{b:02x}"


# ── SVG builder ───────────────────────────────────────────────────────────────

CELL   = 14      # cell width & height in px
GAP    = 2       # gap between cells in px
STEP   = CELL + GAP
LEFT   = 46      # px for day labels
TOP    = 30      # px for year label + month labels per strip
BOT    = 6       # bottom padding per strip
STRIP  = 7 * STEP + TOP + BOT   # total height per year strip


def _make_svg(years_data, discrete_breaks, discrete_colors, discrete_labels,
              title, n_cols_max):
    """
    years_data: list of dicts with keys year, z (7×n), hover (7×n),
                month_cols, month_labels, n_cols
    """
    n_years   = len(years_data)
    svg_w     = LEFT + n_cols_max * STEP + 2
    leg_w     = 110 if discrete_breaks else 0
    total_w   = svg_w + leg_w
    title_h   = 28
    total_h   = title_h + n_years * STRIP + 2

    day_labels = ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"]
    lines      = []

    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" '
                 f'width="{total_w}" height="{total_h}" '
                 f'style="font-family:Segoe UI,Arial,sans-serif;background:#fff;">')

    # Title
    lines.append(f'<text x="{svg_w//2}" y="20" text-anchor="middle" '
                 f'font-size="13" font-weight="bold" fill="#1e293b">{title}</text>')

    for si, yd in enumerate(years_data):
        yr        = yd["year"]
        z         = yd["z"]
        hover     = yd["hover"]
        m_cols    = yd["month_cols"]
        m_labels  = yd["month_labels"]
        n_cols    = yd["n_cols"]

        y_base    = title_h + si * STRIP   # top of this strip
        y_cells   = y_base + TOP           # top of the 7-row cell grid
        x_cells   = LEFT

        # ── Year label ────────────────────────────────────────────────────────
        lines.append(f'<text x="{x_cells + n_cols*STEP//2}" y="{y_base+13}" '
                     f'text-anchor="middle" font-size="12" font-weight="bold" fill="#1e293b">{yr}</text>')

        # ── Month labels ──────────────────────────────────────────────────────
        for mc, ml in zip(m_cols, m_labels):
            mx = x_cells + mc * STEP
            lines.append(f'<text x="{mx+4}" y="{y_base+26}" '
                         f'font-size="10" fill="#555">{ml}</text>')

        # ── Day labels ────────────────────────────────────────────────────────
        for row, dl in enumerate(day_labels):
            dy = y_cells + row * STEP + CELL - 2
            lines.append(f'<text x="{x_cells-4}" y="{dy}" text-anchor="end" '
                         f'font-size="10" fill="#555">{dl}</text>')

        # ── Cells ─────────────────────────────────────────────────────────────
        for row in range(7):
            for col in range(n_cols):
                v = z[row, col]
                if np.isnan(v):
                    continue
                if discrete_breaks is not None:
                    fill = _bin_color(v, discrete_breaks, discrete_colors)
                else:
                    fill = _continuous_color(v, float(np.nanmin(z)), float(np.nanmax(z)))
                if fill is None:
                    continue
                cx = x_cells + col * STEP
                cy = y_cells + row * STEP
                ht = hover[row, col] or ""
                lines.append(
                    f'<rect x="{cx}" y="{cy}" width="{CELL}" height="{CELL}" '
                    f'fill="{fill}" rx="1">'
                    f'<title>{ht}</title></rect>'
                )

        # ── Month boundary lines ───────────────────────────────────────────────
        start    = pd.Timestamp(yr, 1, 1)
        end      = pd.Timestamp(yr, 12, 31)
        all_days = pd.date_range(start, end, freq="D")
        dow_map  = {0:1,1:2,2:3,3:4,4:5,5:6,6:0}
        dow_arr  = np.array([dow_map[d] for d in all_days.dayofweek])

        def px(col): return x_cells + col * STEP - GAP//2
        def py(row): return y_cells + row * STEP - GAP//2
        grid_h = 7 * STEP

        # Outer border
        bx = px(0); by = py(0)
        bw = n_cols * STEP; bh = grid_h
        lines.append(f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" '
                     f'fill="none" stroke="#333" stroke-width="1.5"/>')

        for mo in range(1, 12):
            try:
                fn = pd.Timestamp(yr, mo+1, 1)
            except Exception:
                break
            if fn > end: break
            day_offset   = (fn - start).days
            col_of_first = (day_offset + int(start.strftime("%w"))) // 7
            row_of_first = int(dow_arr[day_offset])

            if row_of_first == 0:
                lines.append(f'<line x1="{px(col_of_first)}" y1="{py(0)}" '
                             f'x2="{px(col_of_first)}" y2="{py(7)}" '
                             f'stroke="#333" stroke-width="1.5"/>')
            else:
                # staircase
                lines.append(f'<polyline points="'
                             f'{px(col_of_first)},{py(7)} '
                             f'{px(col_of_first)},{py(row_of_first)} '
                             f'{px(col_of_first+1)},{py(row_of_first)} '
                             f'{px(col_of_first+1)},{py(0)}'
                             f'" fill="none" stroke="#333" stroke-width="1.5"/>')

    # ── Discrete legend ───────────────────────────────────────────────────────
    if discrete_breaks is not None and discrete_labels is not None:
        lx   = svg_w + 12
        box  = 16
        gap  = 28
        ly0  = title_h + 10
        for j, (lab, col) in enumerate(zip(discrete_labels, discrete_colors)):
            ly = ly0 + j * gap
            lines.append(f'<rect x="{lx}" y="{ly}" width="{box}" height="{box}" '
                         f'fill="{col}" rx="2" stroke="#666" stroke-width="0.8"/>')
            lines.append(f'<text x="{lx+box+6}" y="{ly+box-3}" '
                         f'font-size="11" fill="#1e293b">{lab}</text>')

    lines.append("</svg>")
    return "\n".join(lines)


# ── public API ────────────────────────────────────────────────────────────────

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
    Returns a Plotly Figure containing a single SVG image trace.
    The SVG is built cell-by-cell for pixel-perfect rendering.
    """
    dates  = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    values = pd.Series(values).reset_index(drop=True)
    years  = sorted(dates.dt.year.unique())

    years_data  = []
    n_cols_max  = 0
    for yr in years:
        mask     = dates.dt.year == yr
        yr_vals  = values[mask].astype(float).values
        yr_dates = dates[mask].values
        z, hover, mc, ml, nc = _year_grid(yr_dates, yr_vals, yr)
        years_data.append(dict(year=yr, z=z, hover=hover,
                               month_cols=mc, month_labels=ml, n_cols=nc))
        n_cols_max = max(n_cols_max, nc)

    svg_str = _make_svg(years_data, discrete_breaks, discrete_colors,
                        discrete_labels, title, n_cols_max)

    n_years   = len(years)
    svg_w     = LEFT + n_cols_max * STEP + 2
    leg_w     = 110 if discrete_breaks else 0
    total_w   = svg_w + leg_w
    title_h   = 28
    total_h   = title_h + n_years * STRIP + 2

    fig = go.Figure()
    fig.add_layout_image(
        source="data:image/svg+xml;charset=utf-8," + svg_str.replace("#", "%23"),
        xref="paper", yref="paper",
        x=0, y=1, sizex=1, sizey=1,
        xanchor="left", yanchor="top",
        sizing="stretch", layer="above",
    )
    h = height or min(max(total_h, 220), 1400)
    fig.update_layout(
        margin=dict(t=0, b=0, l=0, r=0),
        height=h,
        paper_bgcolor="#fff",
        plot_bgcolor="#fff",
        xaxis=dict(visible=False, range=[0,1]),
        yaxis=dict(visible=False, range=[0,1]),
    )
    return fig
