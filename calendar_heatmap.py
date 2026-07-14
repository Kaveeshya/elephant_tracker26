"""
calendar_heatmap.py
A Plotly GitHub-style calendar heatmap that replicates the visual role of the
R `calendarHeat()` (lattice) function used in the original Shiny app:
one row per year, weeks across the x-axis, days-of-week down the y-axis,
cells coloured by value, with either a continuous or a discrete legend.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _year_calendar_grid(dates, values, year):
    """Build day-of-week x week-of-year grid data for a single year."""
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year, month=12, day=31)
    all_days = pd.date_range(start, end, freq="D")
    val_map = dict(zip(pd.to_datetime(pd.Series(dates)).dt.normalize(), values))

    dow = all_days.dayofweek.map({0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 0})  # Sun=0..Sat=6
    woty = ((all_days - start).days + start.dayofweek + 1) // 7  # approx week-of-year, Sun-start

    z = np.full((7, woty.max() + 1), np.nan)
    hover = np.empty((7, woty.max() + 1), dtype=object)
    for d, day_of_week, week, day in zip(all_days, dow, woty, all_days):
        v = val_map.get(d.normalize(), np.nan)
        z[day_of_week, week] = v
        hover[day_of_week, week] = f"{d.strftime('%d %b %Y')}<br>Value: {v:.2f}" if pd.notna(v) else f"{d.strftime('%d %b %Y')}<br>No data"

    month_starts = pd.date_range(start, end, freq="MS")
    month_ticks = [((m - start).days + start.dayofweek + 1) // 7 for m in month_starts]
    month_labels = [m.strftime("%b") for m in month_starts]

    return z, hover, month_ticks, month_labels


def make_calendar_heatmap(dates, values, title="", colorscale=None, discrete_breaks=None,
                            discrete_labels=None, discrete_colors=None, height=None):
    """Create a Plotly calendar heatmap figure (one panel per year, stacked)."""
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    values = pd.Series(values).reset_index(drop=True)
    years = sorted(dates.dt.year.unique())

    if discrete_breaks is not None:
        colors = discrete_colors or ["#f1faee", "#a8dadc", "#457b9d", "#1d3557"]
        n = len(discrete_breaks) - 1
        if len(colors) < n:
            # cycle / interpolate
            colors = (colors * (n // len(colors) + 1))[:n]
        bins = pd.cut(values, bins=discrete_breaks, labels=False, include_lowest=True)
        colorscale = []
        for i in range(n):
            colorscale.append([i / n, colors[i]])
            colorscale.append([(i + 1) / n, colors[i]])
        plot_values = bins
        zmin, zmax = 0, n - 1
    else:
        plot_values = values
        colorscale = colorscale or "YlOrRd"
        zmin, zmax = float(np.nanmin(values)) if len(values) else 0, float(np.nanmax(values)) if len(values) else 1

    fig = go.Figure()
    n_years = len(years)
    row_h = 1 / n_years

    for i, yr in enumerate(years):
        mask = dates.dt.year == yr
        z, hover, month_ticks, month_labels = _year_calendar_grid(
            dates[mask], plot_values[mask], yr
        )
        y0 = 1 - (i + 1) * row_h
        fig.add_trace(go.Heatmap(
            z=z,
            text=hover,
            hoverinfo="text",
            colorscale=colorscale,
            zmin=zmin, zmax=zmax,
            showscale=(discrete_breaks is None) and (i == 0),
            xgap=2, ygap=2,
            x0=0, dx=1, y0=0, dy=1,
            xaxis=f"x{i + 1}", yaxis=f"y{i + 1}",
        ))
        fig.update_layout(**{
            f"xaxis{i + 1}": dict(
                domain=[0.06, 1], anchor=f"y{i + 1}",
                tickmode="array", tickvals=month_ticks, ticktext=month_labels,
                showgrid=False, side="bottom",
            ),
            f"yaxis{i + 1}": dict(
                domain=[y0 + row_h * 0.12, y0 + row_h * 0.92], anchor=f"x{i + 1}",
                tickmode="array", tickvals=[0, 1, 2, 3, 4, 5, 6],
                ticktext=["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
                autorange="reversed", showgrid=False,
            ),
        })
        fig.add_annotation(
            x=-0.01, y=y0 + row_h / 2, xref="paper", yref="paper",
            text=f"<b>{yr}</b>", showarrow=False, xanchor="right", font=dict(size=13),
        )

    fig.update_layout(
        title=title,
        height=height or max(220 * n_years, 260),
        margin=dict(t=50, b=30, l=60, r=30),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#333333", family="Segoe UI"),
    )

    if discrete_breaks is not None and discrete_labels is not None:
        # Fake discrete legend using annotations/shapes on the right margin
        for i, (lab, col) in enumerate(zip(discrete_labels, colors)):
            fig.add_annotation(
                x=1.05, y=0.95 - i * 0.07, xref="paper", yref="paper",
                text=f"<span style='color:{col}'>\u25A0</span> {lab}",
                showarrow=False, xanchor="left", font=dict(size=11), align="left",
            )
        fig.update_layout(margin=dict(t=50, b=30, l=60, r=140))

    return fig
