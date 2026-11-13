import asyncio
from collections import Counter

import numpy as np
import pandas as pd
import pytest
from obspy import UTCDateTime
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.core.event import WaveformStreamID

from seisbench_datasets.data.bohemia import (
    NOISE_CANDIDATES,
    Anchor,
    BohemiaSaxony,
    NoiseJob,
    anchors_from_metadata,
    draw_noise_starttimes,
    in_swarm,
    load_swarms,
)
from seisbench_datasets.data.noise import (
    NOISE_GAP_AFTER_EVENT,
    NOISE_GAP_BEFORE_EVENT,
)

DAY = 86400.0


def test_load_swarms():
    swarms = load_swarms()
    assert len(swarms) > 0
    assert swarms == sorted(swarms)
    for (_, end), (start, _) in zip(swarms[:-1], swarms[1:], strict=True):
        assert end <= start  # Not overlapping
    assert all(start < end for start, end in swarms)


def test_in_swarm():
    swarms = [(100.0, 200.0), (300.0, 400.0)]
    assert not in_swarm(swarms, 0, 100)  # Ends where the swarm starts
    assert in_swarm(swarms, 0, 101)
    assert in_swarm(swarms, 150, 160)
    assert in_swarm(swarms, 199, 250)
    assert not in_swarm(swarms, 200, 300)  # Between the swarms
    assert in_swarm(swarms, 250, 350)
    assert not in_swarm(swarms, 400, 500)
    assert not in_swarm([], 0, 1)


def test_draw_noise_starttimes():
    t0 = UTCDateTime(2020, 1, 1).timestamp
    event = UTCDateTime(t0 + 50 * DAY)
    # The event is in a swarm, the catalog covers 100 days
    swarms = [(t0 + 40 * DAY, t0 + 60 * DAY)]
    origin_times = sorted([t0, event.timestamp, t0 + 70 * DAY, t0 + 100 * DAY])

    starttimes = draw_noise_starttimes(
        np.random.default_rng(0), event, 120.0, swarms, origin_times
    )
    assert len(starttimes) == NOISE_CANDIDATES
    for start in starttimes:
        end = start + 120.0
        assert abs(start - event) <= 30 * DAY
        assert not in_swarm(swarms, start.timestamp, end.timestamp)
        for origin in origin_times:
            assert end + NOISE_GAP_BEFORE_EVENT <= origin or (
                start - NOISE_GAP_AFTER_EVENT >= origin
            )

    # Deterministic
    again = draw_noise_starttimes(
        np.random.default_rng(0), event, 120.0, swarms, origin_times
    )
    assert again == starttimes

    # A swarm covers the whole span: no quiet period
    long_swarm = [(t0 - 100 * DAY, t0 + 200 * DAY)]
    rng = np.random.default_rng(0)
    assert draw_noise_starttimes(rng, event, 120.0, long_swarm, origin_times) == []


def _anchor(i, split, network="WB"):
    return Anchor(
        trace_name=f"bs_{i}_{network}.S{i:02d}..SH",
        split=split,
        origin_time=str(UTCDateTime(2020, 1, 1) + i * DAY),
        network=network,
        station=f"S{i:02d}",
        location="",
        channel="SH",
    )


def _dataset(noise_fraction=0.3, noise_length=120.0):
    dataset = BohemiaSaxony.__new__(BohemiaSaxony)
    dataset._noise_fraction = noise_fraction
    dataset._noise_length = noise_length
    dataset._client = None
    return dataset


class _Client:
    def get_source(self, network):
        if network == "XX":
            raise KeyError(f"No source serves network {network}.")
        return "source"


def test_noise_jobs_by_proportion(monkeypatch):
    monkeypatch.setattr("seisbench_datasets.data.bohemia.load_swarms", lambda: [])
    anchors = [_anchor(i, "train") for i in range(20)]
    anchors += [_anchor(i, "test") for i in range(20, 30)]
    anchors += [_anchor(30, "test", network="XX")]  # No source
    origin_times = sorted(UTCDateTime(a.origin_time).timestamp for a in anchors)
    noise = Counter(train=2)
    tried = {"bs_0_WB.S00..SH"}  # Has a noise trace already

    def plan(seed=0):
        return _dataset()._noise_jobs(
            _Client(),
            anchors,
            noise,
            set(tried),
            origin_times,
            np.random.default_rng(seed),
        )

    jobs = plan()
    # 30% of 20 train traces minus the 2 there, 30% of 11 test traces
    assert Counter(job.split for _, job in jobs) == {"train": 4, "test": 3}
    by_name = {a.trace_name: a for a in anchors}
    for source, job in jobs:
        assert source == "source"
        anchor = by_name[job.trace_name.removeprefix("noise_")]
        assert anchor.split == job.split
        assert anchor.network != "XX"
        assert anchor.trace_name not in tried
        assert job.waveform_id.station_code == anchor.station
        assert len(job.starttimes) == NOISE_CANDIDATES

    # Seeded
    assert plan() == jobs
    # Only the proportions count, not which anchors
    assert Counter(j.split for _, j in plan(seed=1)) == {"train": 4, "test": 3}

    # Enough noise
    full = _dataset()._noise_jobs(
        _Client(),
        anchors,
        Counter(train=6, test=3),
        set(),
        origin_times,
        np.random.default_rng(0),
    )
    assert full == []


def test_anchors_from_metadata():
    old = pd.DataFrame(
        {
            "trace_name": ["bucket0$0,:3,:100", "bucket0$1,:3,:100"],
            "trace_name_original": ["bs_1_WB.A..SH", "bs_2_SX.B..HH"],
            "split": ["train", "dev"],
            "source_origin_time": ["2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z"],
            "station_network_code": ["WB", "SX"],
            "station_code": ["A", "B"],
            "station_location_code": ["", ""],
            "trace_channel": ["SH", "HH"],
        }
    )
    anchors, noise = anchors_from_metadata(old)
    # Without a trace category, all traces are event traces
    assert [a.trace_name for a in anchors] == ["bs_1_WB.A..SH", "bs_2_SX.B..HH"]
    assert anchors[1].split == "dev" and anchors[1].channel == "HH"
    assert noise == {}

    new = old.assign(trace_category=["earthquake", "noise"])
    anchors, noise = anchors_from_metadata(new)
    assert [a.trace_name for a in anchors] == ["bs_1_WB.A..SH"]
    assert noise == {"dev": 1}
    assert anchors_from_metadata(pd.DataFrame()) == ([], Counter())


class _Inventory:
    def get_coordinates(self, seed_id, datetime):
        # A real inventory only knows full channel codes
        if len(seed_id.split(".")[-1]) != 3:
            raise Exception(f"No matching coordinates found for {seed_id}")
        return {"latitude": 50.2, "longitude": 12.4, "elevation": 500.0}


def test_noise_data_takes_first_complete_window(monkeypatch):
    starts = [UTCDateTime(2020, 1, 1, h) for h in range(4)]
    requested = []

    async def get_waveform_array(client, inventory, request, sampling_rate, **kw):
        requested.append(request.starttime)
        params = {
            "trace_start_time": str(request.starttime),
            "trace_sampling_rate_hz": 100.0,
            "trace_completeness": 1.0,
        }
        data = np.random.default_rng(0).normal(size=(3, 12000))
        if request.starttime == starts[0]:
            raise FDSNNoDataException("No data")
        if request.starttime == starts[1]:
            params["trace_completeness"] = 0.5  # Gaps
        return "WB.S01..SHZ", params, data

    monkeypatch.setattr(
        "seisbench_datasets.data.bohemia.get_waveform_array", get_waveform_array
    )
    job = NoiseJob(
        trace_name="noise_a",
        waveform_id=WaveformStreamID("WB", "S01", "", "SH"),  # As from an anchor
        split="dev",
        starttimes=tuple(starts),
    )
    params, data = asyncio.run(_dataset().get_noise_data(job, _Inventory()))

    assert requested == starts[:3]
    assert params["trace_start_time"] == str(starts[2])
    assert params["trace_category"] == "noise"
    assert params["trace_name"] == "noise_a"
    assert params["station_code"] == "S01"
    assert data.shape == (3, 12000)

    # No complete candidate
    with pytest.raises(ValueError, match="No candidate window"):
        asyncio.run(
            _dataset().get_noise_data(
                job._replace(starttimes=tuple(starts[:2])), _Inventory()
            )
        )
