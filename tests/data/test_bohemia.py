import asyncio

import numpy as np
import pytest
from obspy import UTCDateTime
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.core.event import Pick, WaveformStreamID

from seisbench_datasets.data.bohemia import (
    NOISE_CANDIDATES,
    BohemiaSaxony,
    NoiseJob,
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


def _event_job(i, split):
    waveform_id = WaveformStreamID("WB", f"S{i:02d}", "", "SHZ")
    event_params = {
        "source_id": f"bs_{i}",
        "source_origin_time": str(UTCDateTime(2020, 1, 1) + i * DAY),
        "split": split,
    }
    pick = Pick(time=UTCDateTime(2020, 1, 1) + i * DAY, waveform_id=waveform_id)
    return "source", (None, event_params, [pick])


def _dataset(noise_fraction=0.3, noise_length=120.0):
    dataset = BohemiaSaxony.__new__(BohemiaSaxony)
    dataset._noise_fraction = noise_fraction
    dataset._noise_length = noise_length
    dataset._client = None
    return dataset


def test_noise_jobs(monkeypatch):
    monkeypatch.setattr("seisbench_datasets.data.bohemia.load_swarms", lambda: [])
    event_jobs = [_event_job(i, "test" if i % 2 else "train") for i in range(20)]
    origin_times = [(UTCDateTime(2020, 1, 1) + i * DAY).timestamp for i in range(20)]

    jobs = _dataset()._noise_jobs(event_jobs, origin_times)
    assert len(jobs) == 6
    for source, job in jobs:
        assert source == "source"
        i = int(job.waveform_id.station_code[1:])
        assert job.trace_name == f"noise_bs_{i}_WB.S{i:02d}..SH"
        assert job.split == event_jobs[i][1][1]["split"]
        assert len(job.starttimes) == NOISE_CANDIDATES

    # Deterministic, so that a resumed download plans the same noise traces
    assert _dataset()._noise_jobs(event_jobs, origin_times) == jobs
    assert _dataset(noise_fraction=0)._noise_jobs(event_jobs, origin_times) == []


class _Inventory:
    def get_coordinates(self, seed_id, datetime):
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
        waveform_id=WaveformStreamID("WB", "S01", "", "SHZ"),
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
