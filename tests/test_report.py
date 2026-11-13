import json
import re

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("plotly")

from typer.testing import CliRunner  # noqa: E402

from seisbench_datasets import report  # noqa: E402
from seisbench_datasets.cli import app  # noqa: E402


def _metadata(n_events=30, phases=("Pg", "Sg")):
    rng = np.random.default_rng(1)
    rows = []
    for i_event in range(n_events):
        split = ("train", "dev", "test")[i_event % 3]
        for i_station in range(3):
            row = {
                "trace_name": f"bucket0${len(rows)},:3,:12000",
                "trace_name_original": f"ev{i_event}_XX.ST{i_station}..HH",
                "split": split,
                "trace_category": "earthquake",
                "source_id": f"ev{i_event}",
                "source_origin_time": f"20{10 + i_event % 10}-01-01T00:00:00.000000Z",
                "source_latitude_deg": 50 + rng.normal(0, 0.1),
                "source_longitude_deg": 12 + rng.normal(0, 0.1),
                "source_depth_km": 8 + rng.normal(),
                "source_magnitude": round(rng.exponential(0.5), 1),
                "source_region": "north" if i_event % 2 else "south",
                "station_network_code": "XX",
                "station_code": f"ST{i_station}",
                "station_location_code": "00" if i_station == 0 else "",
                "station_latitude_deg": 50.2,
                "station_longitude_deg": 12 + 0.1 * i_station,
                "station_elevation_m": 400.0,
                "trace_channel": "HH",
                "trace_start_time": "2010-01-01T00:00:00.000000Z",
                "trace_sampling_rate_hz": 100.0,
                "trace_completeness": 0.8 if i_station == 2 else 1.0,
                "trace_has_spikes": False,
                "trace_component_order": "ZNE",
                "path_back_azimuth_deg": rng.uniform(0, 360),
                f"trace_{phases[0]}_arrival_sample": 6000,
                f"trace_{phases[0]}_status": "manual",
                # No S pick on the last station
                f"trace_{phases[1]}_arrival_sample": 6300 if i_station < 2 else None,
                f"trace_{phases[1]}_status": "manual" if i_station < 2 else None,
            }
            rows.append(row)
    for i in range(9):
        rows.append(
            {
                "trace_name": f"bucket0${len(rows)},:3,:12000",
                "trace_name_original": f"noise_ev{i}_XX.ST0..HH",
                "split": ("train", "dev", "test")[i % 3],
                "trace_category": "noise",
                "source_region": "north" if i % 2 else "south",
                "station_network_code": "XX",
                "station_code": "ST0",
                "station_latitude_deg": 50.2,
                "station_longitude_deg": 12.0,
                "trace_channel": "HH",
                "trace_start_time": "2010-01-02T00:00:00.000000Z",
                "trace_sampling_rate_hz": 100.0,
                "trace_completeness": 1.0,
                "trace_has_spikes": False,
                "trace_component_order": "ZNE",
            }
        )
    return pd.DataFrame(rows)


def _figures(index):
    page = index.read_text()
    match = re.search(
        r'<script type="application/json" id="figures">(.*?)</script>', page, re.S
    )
    return page, json.loads(match.group(1).replace("<\\/", "</"))


def test_helpers():
    names = pd.Series(["bucket3$17,:3,:13205", "plain"])
    samples = report.trace_samples(names)
    assert samples[0] == 13205 and np.isnan(samples[1])

    # One degree of latitude
    assert report.epicentral_distance_km(
        np.array([50.0]), np.array([12.0]), np.array([51.0]), np.array([12.0])
    )[0] == pytest.approx(111.19, abs=0.01)

    # Gutenberg-Richter with b = 1 above Mc 1.0
    rng = np.random.default_rng(0)
    magnitudes = 0.95 + rng.exponential(np.log10(np.e), 20000)
    magnitudes = np.round(magnitudes, 1)
    mc = report.completeness_magnitude(magnitudes)
    assert mc == pytest.approx(1.0)
    assert report.b_value(magnitudes, mc) == pytest.approx(1.0, abs=0.05)
    assert np.isnan(report.b_value(magnitudes[:10], mc))


def test_prepare_phases():
    m = report.prepare(_metadata(n_events=2, phases=("P", "S")))
    traces = m[~m["is_noise"]]
    assert (traces["p_s"] == 60).all()
    assert traces["s_minus_p_s"].dropna().eq(3).all()
    assert traces["s_minus_p_s"].isna().sum() == 2  # No S pick on ST2
    assert (m["duration_s"] == 120).all()
    assert m["is_noise"].sum() == 9


def test_write_report(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _metadata().to_csv(dataset / "metadata.csv", index=False)
    (dataset / "no_data.csv").write_text(
        "trace_name,reason\nev1_XX.ST3..HH,no data\nnoise_ev2_XX.ST3..HH,no data\n"
    )
    (dataset / "noise_rejected.csv").write_text(
        "trace_name,split,reason,probability,weights\n"
        "noise_a,train,earthquake,0.7,original\n"
        "noise_b,dev,above_fraction,,original\n"
    )

    index = report.write_report(
        dataset,
        tmp_path / "report",
        title="Toy",
        group="source_region",
        noise_fraction=0.3,
        status="compiling",
    )
    assert index == tmp_path / "report" / "index.html"
    assert (tmp_path / "report" / "plotly.min.js").stat().st_size > 1e6

    page, figures = _figures(index)
    assert "Compiling" in page
    for key in (
        "noise-fraction",
        "events-map",
        "magnitude-time",
        "magnitude-frequency",
        "events-per-year",
        "depth",
        "traces-per-event",
        "stations-map",
        "traces-per-station",
        "pick-availability",
        "sp-distance",
        "distance",
        "back-azimuth",
        "p-position",
        "s-position",
        "completeness",
        "no-data",
        "rejected-probability",
    ):
        assert key in figures, key
        assert f'data-figure="{key}"' in page
    # Splits in their order and color, in every figure
    assert [t["name"] for t in figures["events-map"]["data"]] == [
        "train",
        "dev",
        "test",
    ]
    assert {t["marker"]["color"] for t in figures["events-map"]["data"]} == {
        "--series-1",
        "--series-2",
        "--series-3",
    }
    # The noise fraction per region and split: 30 % of the event traces is 9
    noise = {t["name"]: t["y"] for t in figures["noise-fraction"]["data"]}
    assert set(noise) == {"train", "dev", "test"}
    # No data, per station
    assert figures["no-data"]["data"][0]["y"] == ["XX.ST3"]
    assert figures["no-data"]["data"][0]["x"] == [2]
    # Pick availability: two of three stations with S
    availability = {
        t["name"]: dict(zip(t["x"], t["y"], strict=True))
        for t in figures["pick-availability"]["data"]
    }
    assert availability["train"]["P and S"] == pytest.approx(66.67)
    assert availability["train"]["P only"] == pytest.approx(33.33)
    # The station table lists every station
    assert page.count("<tr><th scope=row>XX.ST") == 3


def test_write_report_minimal(tmp_path):
    """Only the columns every SeisBench dataset has: no events, no picks."""
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    pd.DataFrame(
        {
            "trace_name": ["a", "b"],
            "split": ["train", "test"],
            "station_network_code": ["XX", "XX"],
            "station_code": ["A", "B"],
            "station_latitude_deg": [50.0, 51.0],
            "station_longitude_deg": [12.0, 13.0],
            "trace_channel": ["HH", "HH"],
            "trace_sampling_rate_hz": [100.0, 100.0],
        }
    ).to_csv(dataset / "metadata.csv", index=False)
    index = report.write_report(dataset, tmp_path / "report", title="Minimal")
    _, figures = _figures(index)
    assert "stations-map" in figures
    assert "events-map" not in figures


def test_cli_report(tmp_path):
    from seisbench_datasets.data.bohemia import BohemiaSaxony

    path = BohemiaSaxony._path_internal()
    path.mkdir(parents=True)
    _metadata().drop(columns="source_region").to_csv(path / "metadata.csv", index=False)

    result = CliRunner().invoke(app, ["bohemia", "--report", str(tmp_path / "out")])
    assert result.exit_code == 0, result.output
    page, figures = _figures(tmp_path / "out" / "index.html")
    assert "BohemiaSaxony" in page and "Complete" in page
    assert not (path / "metadata.csv.lock").exists()
    # Noise per split, one bar each, in split order
    assert [t["name"] for t in figures["noise-fraction"]["data"]] == [
        "train",
        "dev",
        "test",
    ]

    result = CliRunner().invoke(
        app, ["bohemia", "--report", str(tmp_path / "out"), "--check"]
    )
    assert result.exit_code != 0
