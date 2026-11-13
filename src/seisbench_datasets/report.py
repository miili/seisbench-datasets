"""
Interactive HTML report of a compiled dataset, ``--report``: the statistics that
matter for training, from the metadata alone (no waveforms are read).

The figures are plotly.js specs. Colors are written as CSS custom properties
(``"--series-1"``) that the page resolves for its light or dark theme, see
``report.html``. plotly.js is copied next to the page, so it opens from a file;
only the map borders are loaded from the plotly CDN.
"""

from __future__ import annotations

import html
import json
import re
from collections.abc import Callable
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from string import Template

import numpy as np
import pandas as pd

from seisbench_datasets.data.resume import NO_DATA_FILE
from seisbench_datasets.recheck import NOISE_REJECTED_FILE

SPLITS = ("train", "dev", "test")
#: Each split keeps its color in every figure
SPLIT_COLORS = {"train": "--series-1", "dev": "--series-2", "test": "--series-3"}
#: Color of single-series figures that aren't about a split
NEUTRAL_COLOR = "--series-7"
#: Points of the per-trace scatter plots, drawn at random above this
MAX_POINTS = 20_000
#: Bars of the per-station and per-category figures
MAX_BARS = 40
MAGNITUDE_BIN = 0.1
#: Events above the completeness magnitude needed for a b-value
MIN_EVENTS_B_VALUE = 50
EARTH_RADIUS_KM = 6371.0

_BUCKET_SHAPE = re.compile(r"\$\d+,:(\d+),:(\d+)$")
_STATION = re.compile(r"_([^._:]+)\.([^._:]+)\.[^.]*\.[^.]+$")


# Data ------------------------------------------------------------------------


def _phase_columns(metadata: pd.DataFrame, phase: str) -> list[str]:
    """Arrival columns of a phase type, e.g. P, Pg and Pn for ``P``."""
    pattern = re.compile(rf"^trace_({phase}\w*)_arrival_sample$")
    return [column for column in metadata if pattern.match(column)]


def _arrival_s(metadata: pd.DataFrame, phase: str) -> pd.Series:
    """The first arrival of a phase type after the trace start, in seconds."""
    columns = _phase_columns(metadata, phase)
    if not columns:
        return pd.Series(np.nan, index=metadata.index)
    samples = metadata[columns].apply(pd.to_numeric, errors="coerce").min(axis=1)
    return samples / metadata["trace_sampling_rate_hz"]


def trace_samples(trace_names: pd.Series) -> pd.Series:
    """
    The samples of each trace, from the bucket of its name
    (``bucket0$12,:3,:13205``), NaN for traces not in a bucket.
    """
    samples = trace_names.str.extract(_BUCKET_SHAPE)[1]
    return pd.to_numeric(samples, errors="coerce")


def epicentral_distance_km(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def prepare(metadata: pd.DataFrame) -> pd.DataFrame:
    """
    The metadata with the columns the report derives: ``is_noise``,
    ``station``, ``origin_time``, ``p_s``, ``s_s`` (arrivals after the trace
    start), ``s_minus_p_s``, ``distance_km`` and ``duration_s``.
    """
    m = metadata.copy()
    if "trace_category" in m:
        m["is_noise"] = (m["trace_category"] == "noise").to_numpy()
    else:
        m["is_noise"] = False
    m["station"] = m["station_network_code"].astype(str) + "." + m["station_code"]
    if "source_id" not in m:  # A dataset without events
        m["source_id"] = pd.Series(np.nan, index=m.index, dtype=object)
    for column in ("source_origin_time", "trace_start_time"):
        if column in m:
            m[column] = pd.to_datetime(m[column], utc=True, format="ISO8601")
    m["p_s"] = _arrival_s(m, "P")
    m["s_s"] = _arrival_s(m, "S")
    m["s_minus_p_s"] = m["s_s"] - m["p_s"]
    if "source_latitude_deg" in m:
        m["distance_km"] = epicentral_distance_km(
            m["source_latitude_deg"].to_numpy(float),
            m["source_longitude_deg"].to_numpy(float),
            m["station_latitude_deg"].to_numpy(float),
            m["station_longitude_deg"].to_numpy(float),
        )
    else:
        m["distance_km"] = np.nan
    m["duration_s"] = trace_samples(m["trace_name"]) / m["trace_sampling_rate_hz"]
    return m


def events_table(m: pd.DataFrame) -> pd.DataFrame:
    """One row per event, with its split and number of traces."""
    traces = m[~m["is_noise"] & m["source_id"].notna()]
    columns = [
        column
        for column in traces
        if column.startswith("source_") and column != "source_id"
    ]
    events = traces.groupby("source_id", sort=False)[columns + ["split"]].first()
    events["traces"] = traces.groupby("source_id", sort=False).size()
    return events


def completeness_magnitude(magnitudes: np.ndarray) -> float:
    """Maximum curvature: the most frequent magnitude bin."""
    binned = np.round(magnitudes / MAGNITUDE_BIN) * MAGNITUDE_BIN
    values, counts = np.unique(np.round(binned, 6), return_counts=True)
    return float(values[np.argmax(counts)])


def b_value(magnitudes: np.ndarray, mc: float) -> float:
    """Aki–Utsu maximum likelihood b-value above ``mc``, NaN for few events."""
    above = magnitudes[magnitudes >= mc - 1e-9]
    if len(above) < MIN_EVENTS_B_VALUE:
        return float("nan")
    return float(np.log10(np.e) / (above.mean() - (mc - MAGNITUDE_BIN / 2)))


# Figure helpers ---------------------------------------------------------------


def _list(values, decimals: int | None = None) -> list:
    """JSON-ready values, None for NaN."""
    values = np.asarray(values)
    if values.dtype.kind in "fc" or decimals is not None:
        values = values.astype(float)
        if decimals is not None:
            values = np.round(values, decimals)
        return [None if np.isnan(v) else float(v) for v in values]
    if values.dtype.kind in "iub":
        return values.tolist()
    return [None if pd.isna(v) else str(v) for v in values]


def _times(values: pd.Series) -> list:
    return [None if pd.isna(v) else v.strftime("%Y-%m-%d %H:%M:%S") for v in values]


def _splits(frame: pd.DataFrame) -> list[str]:
    present = set(frame["split"].dropna())
    return [split for split in SPLITS if split in present] + sorted(
        present - set(SPLITS)
    )


def _split_color(split: str) -> str:
    return SPLIT_COLORS.get(split, "--text-muted")


def _column(frame: pd.DataFrame, column: str) -> pd.Series:
    """A column, NaN if the dataset has none."""
    if column in frame:
        return frame[column]
    return pd.Series(np.nan, index=frame.index)


def _sample(frame: pd.DataFrame, n: int = MAX_POINTS) -> pd.DataFrame:
    if len(frame) <= n:
        return frame
    rng = np.random.default_rng(0)
    return frame.iloc[np.sort(rng.choice(len(frame), n, replace=False))]


def _per_split(frame: pd.DataFrame, trace: Callable[[str, pd.DataFrame], dict]):
    return [
        {
            "name": split,
            "legendgroup": split,
            **trace(split, frame[frame["split"] == split]),
        }
        for split in _splits(frame)
    ]


def _distribution(
    frame: pd.DataFrame,
    column: str,
    xtitle: str,
    bins: int = 40,
    log_x: bool = False,
    unit: str = "",
    integer: bool = False,
    value_range: tuple[float, float] | None = None,
) -> dict | None:
    """
    Distribution of a column per split as step lines in percent of the split,
    so that splits of different sizes compare.
    """
    values = frame[column].to_numpy(float)
    finite = values[np.isfinite(values) & ((values > 0) if log_x else True)]
    if len(finite) == 0:
        return None
    low, high = value_range or np.nanpercentile(finite, [0, 99.9])
    if high <= low:
        high = low + 1
    if integer:
        edges = np.arange(np.floor(low), np.ceil(high) + 1) + 0.5
        edges = np.concatenate([[edges[0] - 1], edges])
    elif log_x:
        edges = np.geomspace(low, high, bins + 1)
    else:
        edges = np.linspace(low, high, bins + 1)
    centers = np.sqrt(edges[1:] * edges[:-1]) if log_x else (edges[1:] + edges[:-1]) / 2

    def trace(split, part):
        counts, _ = np.histogram(np.clip(part[column], low, high), edges)
        percent = 100 * counts / max(counts.sum(), 1)
        return {
            "type": "scatter",
            "mode": "lines",
            "x": _list(centers, 3),
            "y": _list(percent, 3),
            "line": {"shape": "hvh", "width": 2, "color": _split_color(split)},
            "hovertemplate": f"%{{x:.3~g}} {unit}: %{{y:.1f}}%<extra>{split}</extra>",
        }

    return {
        "data": _per_split(frame.dropna(subset=[column]), trace),
        "layout": {
            "xaxis": {"title": {"text": xtitle}, "type": "log" if log_x else "linear"},
            "yaxis": {
                "title": {"text": "Share of the split (%)"},
                "rangemode": "tozero",
            },
        },
    }


def _split_bars(
    counts: pd.DataFrame,
    title: str,
    horizontal: bool = False,
    stack: bool = False,
    percent: bool = False,
) -> dict:
    """Bars of categories (index) per split (columns)."""
    categories = [str(c) for c in counts.index]
    data = []
    for split in [s for s in SPLITS if s in counts] + [
        s for s in counts if s not in SPLITS
    ]:
        values = _list(counts[split], 2 if percent else None)
        bar = {
            "type": "bar",
            "name": split,
            "legendgroup": split,
            "marker": {
                "color": _split_color(split),
                "line": {"color": "--surface-1", "width": 1.5},
            },
            "hovertemplate": (
                f"%{{{'y' if horizontal else 'x'}}}: "
                f"%{{{'x' if horizontal else 'y'}:{'.1f' if percent else ','}}}"
                f"{'%' if percent else ''}<extra>{split}</extra>"
            ),
        }
        if horizontal:
            bar.update(orientation="h", y=categories, x=values)
        else:
            bar.update(x=categories, y=values)
        data.append(bar)
    value_axis = {"title": {"text": title}}
    category_axis = {"type": "category", "automargin": True}
    if horizontal:
        category_axis["autorange"] = "reversed"
    return {
        "data": data,
        "layout": {
            "barmode": "stack" if stack else "group",
            "xaxis": value_axis if horizontal else category_axis,
            "yaxis": category_axis if horizontal else value_axis,
        },
        "height": max(360, 22 * len(categories) + 120) if horizontal else None,
    }


def _crosstab(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Counts of a column (rows) per split (columns, in split order)."""
    counts = pd.crosstab(frame[column].astype(str), frame["split"])
    return counts[_splits(frame)]


def _geo(lat: np.ndarray, lon: np.ndarray) -> dict:
    """
    A map fitted to the points: plotly.js geo, SVG with Natural Earth borders, as
    it works from a file (a tile map needs a web worker, which browsers block
    there).
    """
    lat, lon = lat[np.isfinite(lat)], lon[np.isfinite(lon)]
    extent = max(np.ptp(lat), np.ptp(lon)) if len(lat) else 360
    local = extent < 60
    return {
        "fitbounds": "locations",
        "resolution": 50,
        "projection": {"type": "mercator" if local else "natural earth"},
        "showcountries": True,
        "showlakes": True,
        "showrivers": bool(extent < 15),
        "showsubunits": bool(local),
        "showframe": False,
    }


# Figures --------------------------------------------------------------------


def _events_map(events: pd.DataFrame) -> dict:
    def trace(split, part):
        return {
            "type": "scattergeo",
            "mode": "markers",
            "lat": _list(part["source_latitude_deg"], 4),
            "lon": _list(part["source_longitude_deg"], 4),
            "marker": {
                "size": 5,
                "color": _split_color(split),
                "opacity": 0.7,
                "line": {"width": 0},
            },
            "text": _list(part.index),
            "customdata": list(
                zip(
                    _times(part["source_origin_time"]),
                    _list(_column(part, "source_magnitude"), 2),
                    _list(_column(part, "source_depth_km"), 1),
                    _list(part["traces"]),
                    strict=True,
                )
            ),
            "hovertemplate": "<b>%{text}</b><br>%{customdata[0]}<br>"
            "M %{customdata[1]} · %{customdata[2]} km deep<br>"
            f"%{{customdata[3]}} traces<extra>{split}</extra>",
        }

    return {
        "data": _per_split(events, trace),
        "layout": {
            "geo": _geo(
                events["source_latitude_deg"].to_numpy(float),
                events["source_longitude_deg"].to_numpy(float),
            ),
            "margin": {"l": 0, "r": 0, "t": 0, "b": 0},
        },
        "height": 560,
    }


def _magnitude_time(events: pd.DataFrame) -> dict:
    events = events.dropna(subset=["source_magnitude"])

    def trace(split, part):
        return {
            "type": "scattergl",
            "mode": "markers",
            "x": _times(part["source_origin_time"]),
            "y": _list(part["source_magnitude"], 2),
            "text": _list(part.index),
            "marker": {"size": 5, "color": _split_color(split), "opacity": 0.7},
            "hovertemplate": f"%{{text}}<br>%{{x}}<br>M %{{y}}<extra>{split}</extra>",
        }

    return {
        "data": _per_split(_sample(events), trace),
        "layout": {
            "xaxis": {"type": "date"},
            "yaxis": {"title": {"text": "Magnitude"}},
        },
    }


def _magnitude_frequency(events: pd.DataFrame) -> dict:
    events = events.dropna(subset=["source_magnitude"])

    def trace(split, part):
        magnitudes = np.sort(part["source_magnitude"].to_numpy(float))
        values = np.unique(np.round(magnitudes / MAGNITUDE_BIN) * MAGNITUDE_BIN)
        values = np.round(values, 6)
        cumulative = len(magnitudes) - np.searchsorted(
            magnitudes, values - MAGNITUDE_BIN / 2
        )
        return {
            "type": "scatter",
            "mode": "lines+markers",
            "x": _list(values, 2),
            "y": _list(cumulative),
            "line": {"width": 2, "color": _split_color(split), "shape": "hv"},
            "marker": {"size": 5, "color": _split_color(split)},
            "hovertemplate": f"N(M ≥ %{{x}}) = %{{y:,}}<extra>{split}</extra>",
        }

    return {
        "data": _per_split(events, trace),
        "layout": {
            "xaxis": {"title": {"text": "Magnitude"}},
            "yaxis": {"title": {"text": "Events with M ≥ magnitude"}, "type": "log"},
        },
    }


def _per_year(events: pd.DataFrame) -> dict:
    years = events["source_origin_time"].dt.year
    counts = pd.crosstab(years, events["split"]).sort_index()
    figure = _split_bars(counts, "Events", stack=True)
    figure["layout"]["xaxis"] = {
        "type": "linear",
        "dtick": 1 if len(counts) <= 20 else 5,
    }
    for bar in figure["data"]:
        bar["x"] = [int(year) for year in counts.index]
    return figure


def _stations_map(m: pd.DataFrame, stations: pd.DataFrame, events) -> dict:
    count = stations["event_traces"].to_numpy(float)
    log_count = np.log10(np.maximum(count, 1))
    top = max(1, int(np.ceil(log_count.max())))
    data = []
    if events is not None and len(events):
        sampled = _sample(events)
        data.append(
            {
                "type": "scattergeo",
                "name": "events",
                "mode": "markers",
                "lat": _list(sampled["source_latitude_deg"], 3),
                "lon": _list(sampled["source_longitude_deg"], 3),
                "marker": {
                    "size": 3,
                    "color": "--text-muted",
                    "opacity": 0.4,
                    "line": {"width": 0},
                },
                "hoverinfo": "skip",
            }
        )
    data.append(
        {
            "type": "scattergeo",
            "name": "stations",
            "mode": "markers",
            "lat": _list(stations["latitude"], 4),
            "lon": _list(stations["longitude"], 4),
            "text": _list(stations.index),
            "customdata": list(
                zip(
                    _list(stations["event_traces"]),
                    _list(stations["noise_traces"]),
                    _list(stations["p_picks"]),
                    _list(stations["s_picks"]),
                    _list(stations["channels"]),
                    strict=True,
                )
            ),
            "marker": {
                "size": 9,
                "symbol": "triangle-up",
                "line": {"color": "--surface-1", "width": 1},
                "color": _list(log_count, 3),
                "cmin": 0,
                "cmax": top,
                "colorscale": [[i / 4, f"--seq-{i}"] for i in range(5)],
                "colorbar": {
                    "title": {"text": "Event traces", "side": "right"},
                    "tickvals": list(range(top + 1)),
                    "ticktext": [_compact(10**i) for i in range(top + 1)],
                    "thickness": 10,
                    "len": 0.6,
                },
            },
            "hovertemplate": "<b>%{text}</b> %{customdata[4]}<br>"
            "%{customdata[0]:,} event traces · %{customdata[1]:,} noise<br>"
            "%{customdata[2]:,} P · %{customdata[3]:,} S picks<extra></extra>",
        }
    )
    return {
        "data": data,
        "layout": {
            "geo": _geo(
                stations["latitude"].to_numpy(float),
                stations["longitude"].to_numpy(float),
            ),
            "margin": {"l": 0, "r": 0, "t": 0, "b": 0},
        },
        "height": 560,
    }


def _sp_distance(traces: pd.DataFrame) -> dict | None:
    traces = traces.dropna(subset=["s_minus_p_s", "distance_km"])
    if traces.empty:
        return None

    def trace(split, part):
        return {
            "type": "scattergl",
            "mode": "markers",
            "x": _list(part["distance_km"], 2),
            "y": _list(part["s_minus_p_s"], 3),
            "text": _list(
                part["trace_name_original"]
                if "trace_name_original" in part
                else part["trace_name"]
            ),
            "marker": {"size": 4, "color": _split_color(split), "opacity": 0.5},
            "hovertemplate": "%{text}<br>%{x:.1f} km · S − P %{y:.2f} s"
            f"<extra>{split}</extra>",
        }

    return {
        "data": _per_split(_sample(traces), trace),
        "layout": {
            "xaxis": {"title": {"text": "Epicentral distance (km)"}},
            "yaxis": {"title": {"text": "S − P time (s)"}},
        },
    }


def _back_azimuth(traces: pd.DataFrame) -> dict | None:
    if "path_back_azimuth_deg" not in traces:
        return None
    traces = traces.dropna(subset=["path_back_azimuth_deg"])
    if traces.empty:
        return None
    edges = np.arange(0, 361, 10)

    def trace(split, part):
        counts, _ = np.histogram(part["path_back_azimuth_deg"] % 360, edges)
        return {
            "type": "barpolar",
            "r": _list(counts),
            "theta": _list(edges[:-1] + 5),
            "width": 9,
            "marker": {"color": _split_color(split), "line": {"width": 0}},
            "hovertemplate": f"%{{theta}}° ± 5°: %{{r:,}} traces<extra>{split}</extra>",
        }

    return {
        "data": _per_split(traces, trace),
        "layout": {
            "barmode": "stack",
            "polar": {
                "angularaxis": {"direction": "clockwise", "rotation": 90},
                "radialaxis": {"showticklabels": False},
            },
        },
    }


def _pick_availability(traces: pd.DataFrame) -> dict:
    has_p, has_s = traces["p_s"].notna(), traces["s_s"].notna()
    category = np.select(
        [has_p & has_s, has_p, has_s], ["P and S", "P only", "S only"], "No pick"
    )
    counts = pd.crosstab(pd.Series(category, index=traces.index), traces["split"])
    order = [c for c in ("P and S", "P only", "S only", "No pick") if c in counts.index]
    percent = 100 * counts.loc[order] / counts.sum()
    return _split_bars(percent, "Share of the event traces (%)", percent=True)


def _single_bars(values: pd.Series, title: str, horizontal: bool = True) -> dict:
    categories = [str(c) for c in values.index]
    bar = {
        "type": "bar",
        "marker": {"color": NEUTRAL_COLOR},
        "hovertemplate": "%{y}: %{x:,}<extra></extra>"
        if horizontal
        else "%{x}: %{y:,}<extra></extra>",
    }
    if horizontal:
        bar.update(orientation="h", y=categories, x=_list(values))
    else:
        bar.update(x=categories, y=_list(values))
    category_axis = {"type": "category", "automargin": True}
    if horizontal:
        category_axis["autorange"] = "reversed"
    return {
        "data": [bar],
        "layout": {
            "showlegend": False,
            "xaxis": {"title": {"text": title}} if horizontal else category_axis,
            "yaxis": category_axis if horizontal else {"title": {"text": title}},
        },
        "height": max(300, 22 * len(categories) + 100) if horizontal else None,
    }


def _histogram(values: np.ndarray, title: str, bins: int = 30) -> dict:
    counts, edges = np.histogram(values, bins)
    return {
        "data": [
            {
                "type": "bar",
                "x": _list((edges[1:] + edges[:-1]) / 2, 3),
                "y": _list(counts),
                "width": float(edges[1] - edges[0]) * 0.9,
                "marker": {"color": NEUTRAL_COLOR},
                "hovertemplate": "%{x:.2f}: %{y:,}<extra></extra>",
            }
        ],
        "layout": {
            "showlegend": False,
            "xaxis": {"title": {"text": title}},
            "yaxis": {"title": {"text": "Traces"}},
        },
    }


# Tables and page ------------------------------------------------------------


def _compact(n: float) -> str:
    for size, suffix in ((1e6, "M"), (1e3, "K")):
        if abs(n) >= size:
            return f"{n / size:.3g}{suffix}"
    return f"{n:.3g}"


def _fmt(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "–"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    if isinstance(value, (float, np.floating)):
        return f"{value:,.2f}" if abs(value) < 100 else f"{value:,.0f}"
    return html.escape(str(value))


def _table(
    frame: pd.DataFrame, index: str, table_id: str | None = None, sortable=False
) -> str:
    def cell(value):
        numeric = isinstance(value, (int, float, np.integer, np.floating))
        sort = f' data-sort="{value}"' if numeric and not pd.isna(value) else ""
        return f'<td class="{"num" if numeric else ""}"{sort}>{_fmt(value)}</td>'

    numeric = [pd.api.types.is_numeric_dtype(frame[c]) for c in frame.columns]
    head = f"<th>{html.escape(str(index))}</th>" + "".join(
        f'<th class="{"num" if n else ""}">{html.escape(str(c))}</th>'
        for c, n in zip(frame.columns, numeric, strict=True)
    )
    rows = "".join(
        f"<tr><th scope=row>{html.escape(str(i))}</th>"
        + "".join(cell(v) for v in row)
        + "</tr>"
        for i, row in zip(frame.index, frame.itertuples(index=False), strict=True)
    )
    attributes = f' id="{table_id}"' if table_id else ""
    if sortable:
        attributes += ' class="sortable"'
    return (
        f'<div class="table-wrap"><table{attributes}><thead><tr>{head}</tr></thead>'
        f"<tbody>{rows}</tbody></table></div>"
    )


def _split_summary(m: pd.DataFrame, events: pd.DataFrame | None) -> pd.DataFrame:
    rows = {}
    for split in _splits(m):
        part = m[m["split"] == split]
        traces = part[~part["is_noise"]]
        n_noise = int(part["is_noise"].sum())
        rows[split] = {
            "Events": int((events["split"] == split).sum())
            if events is not None
            else 0,
            "Event traces": len(traces),
            "Noise traces": n_noise,
            "Noise / event traces (%)": 100 * n_noise / len(traces)
            if len(traces)
            else float("nan"),
            "Stations": traces["station"].nunique(),
            "With P (%)": 100 * traces["p_s"].notna().mean() if len(traces) else 0.0,
            "With S (%)": 100 * traces["s_s"].notna().mean() if len(traces) else 0.0,
            "Median distance (km)": float(traces["distance_km"].median()),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def _group_summary(m: pd.DataFrame, events: pd.DataFrame, group: str) -> pd.DataFrame:
    rows = {}
    for name, part in m.groupby(group, sort=True):
        traces = part[~part["is_noise"]]
        n_noise = int(part["is_noise"].sum())
        row = {
            "Events": traces["source_id"].nunique(),
            "Event traces": len(traces),
            "Noise traces": n_noise,
            "Noise / event traces (%)": 100 * n_noise / len(traces)
            if len(traces)
            else float("nan"),
            "Stations": traces["station"].nunique(),
        }
        for split in _splits(m):
            row[f"Events {split}"] = int(
                ((events[group] == name) & (events["split"] == split)).sum()
            )
        rows[name] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def _magnitude_summary(events: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    events = events.dropna(subset=["source_magnitude"])
    for split in _splits(events):
        magnitudes = events.loc[events["split"] == split, "source_magnitude"]
        magnitudes = magnitudes.to_numpy(float)
        mc = completeness_magnitude(magnitudes)
        rows[split] = {
            "Events": len(magnitudes),
            "Min": magnitudes.min(),
            "Median": float(np.median(magnitudes)),
            "Max": magnitudes.max(),
            "Mc (max. curvature)": mc,
            "b-value": b_value(magnitudes, mc),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def _trace_summary(m: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for name, part in m.groupby(np.where(m["is_noise"], "noise", "event")):
        duration = part["duration_s"]
        row = {
            "Traces": len(part),
            "Duration min (s)": float(duration.min()),
            "Duration median (s)": float(duration.median()),
            "Duration max (s)": float(duration.max()),
            "Sampling rates (Hz)": ", ".join(
                f"{rate:g}" for rate in sorted(part["trace_sampling_rate_hz"].unique())
            ),
        }
        if "trace_completeness" in part:
            row["Incomplete (%)"] = 100 * float((part["trace_completeness"] < 1).mean())
        if "trace_has_spikes" in part:
            spikes = part["trace_has_spikes"].astype(str).str.lower() == "true"
            row["With spikes (%)"] = 100 * float(spikes.mean())
        rows[name] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def _station_table(m: pd.DataFrame) -> pd.DataFrame:
    traces = m[~m["is_noise"]]
    grouped = m.groupby("station")
    stations = pd.DataFrame(
        {
            "network": grouped["station_network_code"].first(),
            "latitude": grouped["station_latitude_deg"].first(),
            "longitude": grouped["station_longitude_deg"].first(),
            "channels": grouped["trace_channel"].agg(
                lambda c: ", ".join(sorted(set(map(str, c))))
            ),
        }
    )
    stations["event_traces"] = traces.groupby("station").size()
    stations["noise_traces"] = m[m["is_noise"]].groupby("station").size()
    stations["p_picks"] = traces.groupby("station")["p_s"].count()
    stations["s_picks"] = traces.groupby("station")["s_s"].count()
    stations["median_distance_km"] = traces.groupby("station")["distance_km"].median()
    stations["events"] = traces.groupby("station")["source_id"].nunique()
    for column in ("event_traces", "noise_traces", "p_picks", "s_picks", "events"):
        stations[column] = stations[column].fillna(0).astype(int)
    return stations.sort_values("event_traces", ascending=False)


def _no_data_stations(path: Path) -> pd.Series | None:
    file = path / NO_DATA_FILE
    if not file.is_file():
        return None
    names = pd.read_csv(file, dtype=str, keep_default_na=False)["trace_name"]
    stations = names.str.extract(_STATION).dropna()
    if stations.empty:
        return None
    return (stations[0] + "." + stations[1]).value_counts()


def _noise_rejected(path: Path) -> pd.DataFrame | None:
    file = path / NOISE_REJECTED_FILE
    if not file.is_file():
        return None
    return pd.read_csv(file, keep_default_na=False)


class _Page:
    """Collects the sections, figures and tables of the report."""

    def __init__(self):
        self.sections: list[tuple[str, str, list[str]]] = []
        self.figures: dict[str, dict] = {}

    def section(self, key: str, title: str, intro: str = "") -> None:
        self.sections.append(
            (key, title, [f'<p class="intro">{intro}</p>'] if intro else [])
        )

    def _card(self, title: str, caption: str, body: str, wide: bool) -> None:
        self.sections[-1][2].append(
            f'<figure class="card{" wide" if wide else ""}"><figcaption>'
            f"<h3>{title}</h3><p>{caption}</p></figcaption>{body}</figure>"
        )

    def figure(
        self, key: str, title: str, caption: str, figure: dict | None, wide=False
    ) -> None:
        if figure is None or not figure["data"]:
            return
        height = figure.pop("height", None) or (560 if wide else 380)
        self.figures[key] = figure
        self._card(
            title,
            caption,
            f'<div class="plot" id="fig-{key}" data-figure="{key}" '
            f'style="height:{height}px"></div>',
            wide,
        )

    def table(self, title: str, caption: str, table: str, wide=True) -> None:
        self._card(title, caption, table, wide)

    def html(self) -> tuple[str, str]:
        nav = "".join(
            f'<a href="#{key}">{html.escape(title)}</a>'
            for key, title, _ in self.sections
        )
        body = "".join(
            f'<section id="{key}"><h2>{html.escape(title)}</h2>'
            f'<div class="grid">{"".join(parts)}</div></section>'
            for key, title, parts in self.sections
        )
        return nav, body


def _tiles(tiles: list[tuple[str, str, str]]) -> str:
    return "".join(
        f'<div class="tile"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="note">{note}</div></div>'
        for label, value, note in tiles
    )


def write_report(
    path: Path,
    directory: Path,
    title: str,
    group: str = "split",
    noise_fraction: float | None = None,
    status: str = "complete",
) -> Path:
    """
    Writes the report of the dataset in ``path`` to ``directory/index.html``, with
    plotly.js next to it.

    :param group: The column the noise fraction is kept per, e.g. ``source_region``
    :param noise_fraction: The noise fraction the dataset aims for
    :param status: ``complete``, ``incomplete`` (interrupted) or ``compiling``
    :return: The path of the page
    """
    from plotly.offline import get_plotlyjs

    metadata = pd.read_csv(
        path / "metadata.csv",
        low_memory=False,
        dtype={
            "station_network_code": str,
            "station_code": str,
            "station_location_code": str,
        },
    )
    m = prepare(metadata)
    has_events = "source_id" in m and m["source_id"].notna().any()
    events = events_table(m) if has_events else None
    traces = m[~m["is_noise"]]
    stations = _station_table(m)
    page = _Page()

    # Overview
    n_noise = int(m["is_noise"].sum())
    period = ""
    if events is not None and len(events):
        times = events["source_origin_time"]
        period = f"{times.min():%Y-%m-%d} – {times.max():%Y-%m-%d}"
    overall = 100 * n_noise / len(traces) if len(traces) else float("nan")
    target = f"aim {noise_fraction:.0%}" if noise_fraction is not None else ""
    tiles = [
        ("Event traces", _compact(len(traces)), f"{len(traces):,}"),
        ("Events", _compact(len(events) if events is not None else 0), period),
        (
            "Noise traces",
            _compact(n_noise),
            f"{overall:.1f}% of event traces, {target}",
        ),
        (
            "Stations",
            f"{len(stations):,}",
            f"{m['station_network_code'].nunique()} networks",
        ),
        (
            "Picks",
            _compact(int(traces["p_s"].notna().sum() + traces["s_s"].notna().sum())),
            f"{traces['p_s'].notna().sum():,} P · {traces['s_s'].notna().sum():,} S",
        ),
        (
            "Sampling rate",
            ", ".join(f"{r:g}" for r in sorted(m["trace_sampling_rate_hz"].unique()))
            + " Hz",
            f"{m['trace_component_order'].iloc[0]} components"
            if "trace_component_order" in m
            else "",
        ),
    ]

    page.section(
        "overview",
        "Overview",
        "Splits are by event: all traces of an event are in one split. Noise "
        "traces are counted per event trace of the same "
        + ("split." if group == "split" else f"{html.escape(group)}."),
    )
    page.table(
        "Splits",
        "Events, traces and picks per split.",
        _table(_split_summary(m, events), "Split"),
    )
    if group != "split" and group in m and events is not None:
        page.table(
            f"Per {html.escape(group)}",
            "The noise fraction is kept per group, events per split in the last "
            "columns.",
            _table(_group_summary(m, events, group), group),
        )
    noise_counts = pd.crosstab(m[group].astype(str), [m["split"], m["is_noise"]])
    noise_percent = pd.DataFrame(
        {
            split: 100
            * noise_counts.get((split, True), 0)
            / noise_counts.get((split, False), 0).replace(0, np.nan)
            for split in _splits(m)
        }
    )
    noise_figure = _split_bars(noise_percent, "Noise / event traces (%)", percent=True)
    if group == "split":  # One bar per split, centered, in split order
        noise_percent = noise_percent.loc[_splits(m)]
        noise_figure = _split_bars(
            noise_percent, "Noise / event traces (%)", percent=True
        )
        noise_figure["layout"]["barmode"] = "overlay"
    if noise_fraction is not None:
        noise_figure["layout"]["shapes"] = [
            {
                "type": "line",
                "xref": "paper",
                "x0": 0,
                "x1": 1,
                "y0": 100 * noise_fraction,
                "y1": 100 * noise_fraction,
                "line": {"color": "--text-muted", "width": 1},
            }
        ]
        noise_figure["layout"]["annotations"] = [
            {
                "xref": "paper",
                "x": 1,
                "y": 100 * noise_fraction,
                "text": f"aim {noise_fraction:.0%}",
                "showarrow": False,
                "xanchor": "right",
                "yanchor": "bottom",
                "font": {"color": "--text-secondary"},
            }
        ]
    page.figure(
        "noise-fraction",
        "Noise fraction",
        "Noise traces per event trace, per split"
        + ("." if group == "split" else f" and {html.escape(group)}."),
        noise_figure,
    )
    for column, name in (("source_region", "region"), ("source_event_class", "class")):
        if events is not None and column in events and column != group:
            page.figure(
                f"events-per-{name}",
                f"Events per {name}",
                f"Events of each {name}, per split.",
                _split_bars(_crosstab(events, column), "Events"),
            )

    # Events
    if events is not None and len(events):
        page.section(
            "events",
            "Events",
            "One point or count per event. Click a split in a legend, or above, to "
            "hide it.",
        )
        page.figure(
            "events-map",
            "Epicenters",
            f"{len(events):,} events, colored by split.",
            _events_map(events.dropna(subset=["source_latitude_deg"])),
            wide=True,
        )
        if "source_magnitude" in events and events["source_magnitude"].notna().any():
            sampled = (
                f" {MAX_POINTS:,} events drawn at random."
                if len(events) > MAX_POINTS
                else ""
            )
            page.figure(
                "magnitude-time",
                "Magnitude over time",
                "Gaps are periods without events, or without data." + sampled,
                _magnitude_time(events),
                wide=True,
            )
            page.figure(
                "magnitude-frequency",
                "Frequency–magnitude distribution",
                "Cumulative events per split. Parallel lines mean the splits sample "
                "the same magnitudes.",
                _magnitude_frequency(events),
            )
            page.table(
                "Magnitudes",
                "Completeness magnitude Mc by maximum curvature, Aki–Utsu b-value "
                f"above it (at least {MIN_EVENTS_B_VALUE} events).",
                _table(_magnitude_summary(events), "Split"),
                wide=False,
            )
        page.figure(
            "events-per-year",
            "Events per year",
            "Stacked by split.",
            _per_year(events.dropna(subset=["source_origin_time"])),
        )
        if "source_depth_km" in events:
            page.figure(
                "depth",
                "Depth",
                "Share of the events of each split per depth bin.",
                _distribution(events, "source_depth_km", "Depth (km)", unit="km"),
            )
        page.figure(
            "traces-per-event",
            "Traces per event",
            "Stations that recorded an event, as share of the events of each split.",
            _distribution(events, "traces", "Traces per event", integer=True),
        )
        if "source_volcano" in events and events["source_volcano"].notna().any():
            volcano = _crosstab(
                events.dropna(subset=["source_volcano"]), "source_volcano"
            )
            volcano = volcano.loc[
                volcano.sum(axis=1).sort_values(ascending=False).index
            ]
            page.figure(
                "events-per-volcano",
                "Events per volcano",
                f"The {min(MAX_BARS, len(volcano))} volcanoes with most events, "
                "stacked by split.",
                _split_bars(
                    volcano.iloc[:MAX_BARS], "Events", horizontal=True, stack=True
                ),
            )

    # Stations
    page.section(
        "stations",
        "Stations",
        "Traces per station, from the event and the noise traces.",
    )
    page.figure(
        "stations-map",
        "Stations",
        f"{len(stations):,} stations, colored by event traces (log scale), "
        "with the events in gray.",
        _stations_map(m, stations, events),
        wide=True,
    )
    per_station = pd.crosstab(traces["station"], traces["split"])
    per_station = per_station.reindex(stations.index[:MAX_BARS]).fillna(0)
    per_station = per_station[per_station.sum(axis=1) > 0].astype(int)
    page.figure(
        "traces-per-station",
        "Event traces per station",
        f"The {len(per_station)} stations with most event traces, stacked by split."
        + (" All stations are in the table below." if len(stations) > MAX_BARS else ""),
        _split_bars(per_station, "Event traces", horizontal=True, stack=True),
    )
    networks = pd.crosstab(traces["station_network_code"], traces["split"])
    networks = networks.loc[networks.sum(axis=1).sort_values(ascending=False).index]
    page.figure(
        "traces-per-network",
        "Event traces per network",
        "Stacked by split.",
        _split_bars(
            networks.iloc[:MAX_BARS], "Event traces", horizontal=True, stack=True
        ),
    )
    station_columns = {
        "network": "Network",
        "channels": "Channels",
        "events": "Events",
        "event_traces": "Event traces",
        "noise_traces": "Noise traces",
        "p_picks": "P picks",
        "s_picks": "S picks",
        "median_distance_km": "Median distance (km)",
        "latitude": "Latitude",
        "longitude": "Longitude",
    }
    page.table(
        "All stations",
        'Click a column to sort. <input type="search" id="station-filter" '
        'placeholder="Filter stations" aria-label="Filter stations">',
        _table(
            stations[list(station_columns)].rename(columns=station_columns),
            "Station",
            table_id="station-table",
            sortable=True,
        ),
    )

    # Picks
    if len(traces):
        page.section(
            "picks",
            "Picks and geometry",
            "Arrivals are the first pick of each phase type (P, Pg, Pn … and S, Sg, "
            "Sn …).",
        )
        page.figure(
            "pick-availability",
            "Pick availability",
            "Share of the event traces of each split with P and S picks.",
            _pick_availability(traces),
        )
        pick_status = []
        for phase in ("P", "S"):
            for column in _phase_columns(m, phase):
                status_column = column.replace("_arrival_sample", "_status")
                if status_column in traces:
                    counts = traces[status_column].dropna().astype(str).value_counts()
                    pick_status.append(pd.Series(counts, name=column.split("_")[1]))
        if pick_status:
            page.table(
                "Pick status",
                "Picks per phase and status.",
                _table(pd.concat(pick_status, axis=1).fillna(0).astype(int).T, "Phase"),
                wide=False,
            )
        sampled = (
            f" {MAX_POINTS:,} traces drawn at random."
            if traces[["s_minus_p_s", "distance_km"]].notna().all(axis=1).sum()
            > MAX_POINTS
            else ""
        )
        page.figure(
            "sp-distance",
            "S − P time over distance",
            "Picks off the trend are suspect: wrong phase, wrong event or a bad "
            "station location." + sampled,
            _sp_distance(traces),
            wide=True,
        )
        page.figure(
            "distance",
            "Epicentral distance",
            "Share of the event traces of each split per distance bin.",
            _distribution(traces, "distance_km", "Epicentral distance (km)", unit="km"),
        )
        page.figure(
            "back-azimuth",
            "Back azimuth",
            "Event traces per direction from the station to the event, stacked by "
            "split.",
            _back_azimuth(traces),
        )
        # Arrivals over the whole trace, in 0.5 s bins
        times = traces[["duration_s", "p_s", "s_s"]].to_numpy(float)
        duration = float(np.max(times[np.isfinite(times)], initial=0)) or 1.0
        bins_05s = max(1, int(np.ceil(duration / 0.5)))
        page.figure(
            "p-position",
            "P arrival in the trace",
            "Seconds after the trace start, over the whole trace. Training windows "
            "cut around it.",
            _distribution(
                traces,
                "p_s",
                "P arrival after trace start (s)",
                bins=bins_05s,
                unit="s",
                value_range=(0, duration),
            ),
        )
        page.figure(
            "s-position",
            "S arrival in the trace",
            "Seconds after the trace start, over the whole trace.",
            _distribution(
                traces,
                "s_s",
                "S arrival after trace start (s)",
                bins=bins_05s,
                unit="s",
                value_range=(0, duration),
            ),
        )

    # Traces
    page.section("traces", "Traces", "Waveform properties, event and noise traces.")
    page.table(
        "Event and noise traces",
        "Duration from the stored samples. Incomplete traces have gaps, filled "
        "with zeros.",
        _table(_trace_summary(m), "Category"),
    )
    if "trace_completeness" in m:
        incomplete = m[m["trace_completeness"] < 1]
        if len(incomplete):
            page.figure(
                "completeness",
                "Completeness of the incomplete traces",
                f"{len(incomplete):,} traces ({100 * len(incomplete) / len(m):.1f}%) "
                "have gaps, filled with zeros.",
                _histogram(
                    incomplete["trace_completeness"].to_numpy(float),
                    "Completeness",
                    bins=20,
                ),
            )
    channels = pd.crosstab(m["trace_channel"], m["split"])
    page.figure(
        "channels",
        "Channels",
        "Instrument and band codes of all traces, stacked by split.",
        _split_bars(
            channels.loc[channels.sum(axis=1).sort_values(ascending=False).index],
            "Traces",
            stack=True,
        ),
    )

    # Downloads
    no_data = _no_data_stations(path)
    rejected = _noise_rejected(path)
    if no_data is not None or rejected is not None:
        page.section(
            "downloads",
            "Downloads",
            "Traces the data centers had no data for, and noise traces removed by "
            "the earthquake detection.",
        )
    if no_data is not None:
        page.figure(
            "no-data",
            "No data",
            f"{no_data.sum():,} traces at {len(no_data)} stations were skipped "
            f"(HTTP 204 / 404, {NO_DATA_FILE}). The {min(MAX_BARS, len(no_data))} "
            "stations with most.",
            _single_bars(no_data.iloc[:MAX_BARS], "Traces without data"),
        )
    if rejected is not None and len(rejected):
        if {"reason", "split"} <= set(rejected):
            page.table(
                "Removed noise traces",
                f"From {NOISE_REJECTED_FILE}: noise traces with an earthquake, and "
                "those above the noise fraction.",
                _table(_crosstab(rejected, "reason"), "Reason"),
                wide=False,
            )
        if "probability" in rejected:
            probability = pd.to_numeric(rejected["probability"], errors="coerce")
            probability = probability[probability.notna()].to_numpy(float)
            if len(probability):
                page.figure(
                    "rejected-probability",
                    "Phase probability of the removed noise",
                    "The highest P or S probability of PhaseNet in the noise traces "
                    "removed as earthquakes.",
                    _histogram(probability, "Phase probability", bins=25),
                )

    nav, body = page.html()
    template = Template(
        resources.files("seisbench_datasets").joinpath("report.html").read_text()
    )
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    status = {
        "complete": '<span class="badge good">Complete</span>',
        "incomplete": '<span class="badge warning">Incomplete: the compilation was '
        "interrupted</span>",
        "compiling": '<span class="badge warning">Compiling: up to the last '
        "checkpoint</span>",
    }[status]
    figures = json.dumps(page.figures, allow_nan=False, separators=(",", ":"))
    page_html = template.substitute(
        title=html.escape(title),
        path=html.escape(str(path)),
        generated=generated,
        status=status,
        tiles=_tiles(tiles),
        splits="".join(
            f'<button type="button" class="chip" aria-pressed="true" '
            f'data-split="{s}"><span class="swatch" style="background:var({_split_color(s)})">'
            f"</span>{s}</button>"
            for s in _splits(m)
        ),
        nav=nav,
        body=body,
        figures=figures.replace("</", "<\\/"),
    )
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "plotly.min.js").write_text(get_plotlyjs())
    index = directory / "index.html"
    index.write_text(page_html)
    return index
