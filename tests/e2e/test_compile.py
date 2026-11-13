"""
End-to-end compilation of tiny subsets of the datasets, against the real data
centers. Deselected by default; run with

    EIDA_TOKEN=<token file> DLF_CATALOG_DIR=<quakeml dir> \\
    HINET_USER=<user> HINET_PASSWORD=<password> uv run pytest -m e2e tests/e2e

Everything is written to temporary folders, see ``tests/conftest.py``. The
Bohemia catalog is copied from the SeisBench cache if it is there.
"""

import os
import re
import shutil
from pathlib import Path

import numpy as np
import pytest
import seisbench
from obspy import Catalog, UTCDateTime, read_events
from obspy.io.nordic.core import read_nordic

from seisbench_datasets.check import check_dataset, print_report
from seisbench_datasets.data.bohemia import BohemiaSaxony, in_swarm, load_swarms
from seisbench_datasets.data.dlf import DeepLowFrequency

pytestmark = pytest.mark.e2e

# Captured at import, before the fixtures point the cache to a temporary folder
REAL_DATA_ROOT = seisbench.cache_data_root

EIDA_TOKEN = os.environ.get("EIDA_TOKEN")


class TinyBohemia(BohemiaSaxony):
    """
    Three events of 2018, spread over the year, so that there are quiet periods
    between them.
    """

    @classmethod
    def get_catalog(cls):
        path = cls._path_internal()
        cls._download_catalog_colm(path)
        text = (path / "final2" / "cll_2018.txt").read_text().replace("\r", "")
        events = [e for e in re.split(r"\n[ \t]*\n", text) if e.strip()]
        subset = [events[0], events[len(events) // 2], events[-1]]
        file = path / "tiny_catalog.txt"
        file.write_text("\n\n".join(subset) + "\n\n")
        return read_nordic(str(file))


def _copy_bohemia_catalog():
    catalog = REAL_DATA_ROOT / "bohemiasaxony" / "final2"
    if catalog.is_dir():
        shutil.copytree(catalog, TinyBohemia._path_internal() / "final2")


def test_bohemia():
    _copy_bohemia_catalog()
    dataset = TinyBohemia(eida_token=EIDA_TOKEN, noise_fraction=1.0)
    metadata = dataset.metadata
    print(metadata["trace_category"].value_counts())

    assert set(metadata["trace_category"]) == {"earthquake", "noise"}
    events = metadata[metadata["trace_category"] == "earthquake"]
    noise = metadata[metadata["trace_category"] == "noise"]
    assert len(events) >= 5
    assert len(noise) >= 0.5 * len(events)
    assert events["trace_Pg_arrival_sample"].notna().any()
    assert noise["trace_Pg_arrival_sample"].isna().all()
    assert set(noise["split"]) <= {"train", "dev", "test"}

    swarms = load_swarms()
    origin_times = [UTCDateTime(t) for t in events["source_origin_time"]]
    for i in np.flatnonzero(metadata["trace_category"] == "noise"):
        row = metadata.iloc[i]
        start = UTCDateTime(row["trace_start_time"])
        end = start + 120.0
        assert not in_swarm(swarms, start.timestamp, end.timestamp)
        assert min(origin_times) <= start and end <= max(origin_times)
        waveforms = dataset.get_waveforms(i)
        assert waveforms.shape[0] == 3
        assert abs(waveforms.shape[1] - 12000) <= 1  # Sample alignment
        assert np.ptp(waveforms, axis=1).max() > 0

    # Export for pyrocko, as with --export
    pytest.importorskip("pyrocko")
    from pyrocko.model import load_events

    from seisbench_datasets.cli import _export

    export = TinyBohemia._path_internal().parent / "export"
    _export(TinyBohemia, export)
    assert list((export / "mseed").glob("*.mseed"))
    assert (export / "stations.yaml").is_file()
    assert (
        len(load_events(str(export / "events.yaml"))) == events["source_id"].nunique()
    )


def test_bohemia_fill_up():
    """
    A dataset compiled without noise, as before the noise traces, is checked and
    filled up with noise.
    """
    _copy_bohemia_catalog()
    TinyBohemia(eida_token=EIDA_TOKEN, noise_fraction=0.0)
    path = TinyBohemia._path_internal()

    before = check_dataset(path, 0.5, TinyBohemia.planned_trace_names())
    print_report(before)
    assert before.ok
    assert all(split.noise == 0 for split in before.splits)
    # GEOFON has no WB.VAC data for the events: recorded, and skipped from now on
    assert before.missing and before.no_data == before.missing

    TinyBohemia(eida_token=EIDA_TOKEN, noise_fraction=0.5, update=True)

    after = check_dataset(path, 0.5, TinyBohemia.planned_trace_names())
    print_report(after)
    assert after.ok
    assert TinyBohemia.is_complete()
    # The event traces are kept
    assert [s.events for s in after.splits] == [s.events for s in before.splits]
    assert after.missing == before.missing
    for split in after.splits:
        assert split.noise == round(0.5 * split.events)


DLF_REGIONS = ("hawaii", "eifel_qseek", "japan")


def _tiny_catalog(catalog: Catalog, n_events: int = 2, n_stations: int = 3):
    """
    The latest events, with the picks of their first stations.
    """
    events = sorted(
        (e for e in catalog if e.preferred_origin() is not None and e.picks),
        key=lambda e: e.preferred_origin().time,
    )[-n_events:]
    for event in events:
        stations = []
        for pick in event.picks:
            if pick.waveform_id.station_code not in stations:
                stations.append(pick.waveform_id.station_code)
        keep = set(stations[:n_stations])
        event.picks = [p for p in event.picks if p.waveform_id.station_code in keep]
    return Catalog(events)


@pytest.fixture(scope="module")
def dlf_catalog_dir(tmp_path_factory) -> Path:
    source = os.environ.get("DLF_CATALOG_DIR")
    if not source:
        pytest.skip("DLF_CATALOG_DIR is not set.")
    folder = tmp_path_factory.mktemp("quakeml")
    for region in DLF_REGIONS:
        catalog = read_events(str(Path(source) / f"{region}.xml"), format="QUAKEML")
        _tiny_catalog(catalog).write(str(folder / f"{region}.xml"), format="QUAKEML")
    return folder


class TinyDLF(DeepLowFrequency):
    pass


def test_dlf(dlf_catalog_dir):
    user = os.environ.get("HINET_USER")
    password = os.environ.get("HINET_PASSWORD")
    if not (user and password):
        pytest.skip("HINET_USER and HINET_PASSWORD are not set.")

    dataset = TinyDLF(
        catalog_dir=dlf_catalog_dir,
        eida_token=EIDA_TOKEN,
        hinet_user=user,
        hinet_password=password,
        noise_fraction=1.0,
    )
    metadata = dataset.metadata
    print(metadata.groupby(["source_region", "trace_category"]).size())
    print((dataset.path / "download_stats.json").read_text())

    events = metadata[metadata["trace_category"] == "event"]
    # FDSN (EarthScope, BGR / GFZ) and Hi-net
    assert set(events["source_region"]) == set(DLF_REGIONS)
    assert (metadata["trace_category"] == "noise").any()
    for i in range(len(metadata)):
        assert np.ptp(dataset.get_waveforms(i), axis=1).max() > 0
