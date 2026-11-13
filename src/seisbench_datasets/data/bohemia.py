from __future__ import annotations

import asyncio
import bisect
import csv
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from importlib import resources
from itertools import groupby
from pathlib import Path
from typing import Literal, NamedTuple, cast
from urllib.parse import urljoin
from zipfile import ZipFile

import numpy as np
import pandas as pd
import seisbench
from obspy import Inventory, UTCDateTime, read_inventory
from obspy.clients.fdsn.header import URL_MAPPINGS, FDSNNoDataException
from obspy.core.event.base import QuantityError, WaveformStreamID
from obspy.core.event.catalog import Catalog, Event
from obspy.core.event.origin import Pick
from obspy.geodetics import degrees2kilometers, gps2dist_azimuth
from obspy.io.nordic.core import read_nordic
from seisbench import logger
from seisbench.data.base import (
    EventParameters,
    TraceParameters,
)
from seisbench.util import download_http

from seisbench_datasets.data.multiclient import (
    FDSNSource,
    MultiClient,
    Source,
    WaveformRequest,
    download_parallel,
    get_cached_inventory,
    get_waveform_array,
)
from seisbench_datasets.data.noise import check_complete, is_away_from_events
from seisbench_datasets.data.resume import ResumableBenchmarkDataset

REMOVE_CHANNELS = {
    "HHT",  # Stray channel or typo
    "BHN",  # Low sampling rate channel BH
    "BHE",
    "BHZ",
    "SHN",  # Low sampling rate channel SH
    "SHE",
    "SHZ",
}

STATION_MAPPING = {
    "LACW": "LAC",
    "STCW": "STC",
    "VACW": "VAC",
    "KVCW": "KVC",
    "KRCW": "KRC",
    "KOCW": "KOC",
    "POCW": "POC",
    "LBCW": "LBC",
    "SKCW": "SKC",
    "ZHCW": "ZHC",
    "ZEIT": "ZEITZ",
    "LEIB": "LEIB1",
}

COMPONENT_ORDER = "ZNE"

# Networks the catalog picks come from (see the catalog README), used to resolve
# station codes that exist in several networks
PREFERRED_NETWORKS = ("SX", "WB", "TH", "BW", "CZ", "GR")
RNG_SEED = 31337

PICK_WHITELIST = {
    "Pg",
    "Sg",
}

WEBNET_SWITCH_SH_CH_CHANNEL = datetime(2015, 5, 26)

SAMPLING_RATE = 100.0

# Noise windows are drawn at random times in quiet periods: outside the swarm
# windows in bohemia_swarms.csv (see scripts/bohemia_swarms.py), away from catalog
# events, and within NOISE_SPAN of the event trace they are drawn for, so that the
# station records with the same channels and the catalog covers the time.
NOISE_SPAN = 30 * 86400.0
# Candidate windows per noise trace, the first one without gaps is used
NOISE_CANDIDATES = 5
NOISE_MAX_DRAWS = 1000
# Rounds of noise downloads, each tops up the noise that failed in the round before
NOISE_MAX_ROUNDS = 3

EventJob = tuple[Event, EventParameters, list[Pick]]


class Anchor(NamedTuple):
    """
    An event trace of the dataset, at whose station and near whose time a noise
    trace can be drawn.
    """

    trace_name: str
    split: str
    origin_time: str
    network: str
    station: str
    location: str
    channel: str


class NoiseJob(NamedTuple):
    """
    A noise trace at the station of an event trace, with candidate start times.
    """

    trace_name: str
    waveform_id: WaveformStreamID
    split: str
    starttimes: tuple[UTCDateTime, ...]


class BohemiaSaxony(ResumableBenchmarkDataset):
    """
    Regional benchmark dataset of waveform data and metadata
    for the North-West Bohemia and Saxony region in Germany/Czech Republic.

    .. warning ::

        This dataset contains restricted data from the West Bohemia Local Seismic Network (WEBNET).
        To compile the full dataset, you will need to provide an EIDA token.
        Please see the `WEBNET site <https://doi.org/10.7914/SN/WB>`_ for more information.

    .. note ::

        The dataset is compiled from FDSN web services on first use. An interrupted
        download resumes where it stopped when the dataset is created again.

    Noise traces (``trace_category = "noise"``) make up ``noise_fraction`` of the
    event traces (``trace_category = "earthquake"``) of each split. Each is drawn at
    the station of a random event trace of the split, in a quiet period, not selected
    by STA/LTA: at a random time within 30 days of the event, outside the swarm
    windows of the catalog (``bohemia_swarms.csv``), at least 120 s before and 180 s
    after every catalog event. Of 5 candidate windows, the first without gaps is
    used. The noise is planned after the event traces, from the event traces in the
    dataset, so a resumed or filled-up dataset reaches the same proportions.

    :param eida_token: EIDA token or path to a token file, for the restricted WB data
    :param noise_fraction: Fraction of the event traces that get a noise trace. 0
        disables noise traces.
    :param noise_length: Length of the noise traces in seconds
    :param connections: Number of parallel connections per FDSN data center
    """

    _client: MultiClient

    def __init__(
        self,
        eida_token: str | None = None,
        noise_fraction: float = 0.3,
        noise_length: float = 120.0,
        connections: int = 4,
        compile_from_source: bool = True,
        **kwargs,
    ):
        citation = """
The data is compiled from data of the networks from Saxony Seismic Network (SX),
West Bohemia Local Seismic Network (WEBNET; WB), Thuringia Seismological Network (TH),
and Czech Regional Seismic Network (CZ) and is provided by the BGR (Bundesanstalt für
Geowissenschaften und Rohstoffe) and GFZ GEOFON.

The dataset includes restricted data from the WB network, which requires an EIDA token
for access. Please provide the path to your EIDA token file when initializing the
dataset via the `eida_token` argument.

Catalog and Picks:
* Earthquakes in Saxony (Germany) and surroundings
  from 2006 to 2024 -- onsets and locations,
  https://opara.zih.tu-dresden.de/items/5387886f-25f2-4faf-8dca-33981d898ab9

Seismic Networks:
* SXNET Saxon Seismic Network, https://doi.org/10.7914/SN/SX
* West Bohemia Local Seismic Network (WEBNET), https://doi.org/10.7914/SN/WB
* Thüringer Seismologisches Netz, https://doi.org/10.7914/SN/TH
* Czech Regional Seismic Network, https://doi.org/10.7914/SN/CZ
"""
        self._eida_token = eida_token
        self._noise_fraction = noise_fraction
        self._noise_length = noise_length
        self._connections = connections

        super().__init__(
            citation=citation,
            license="CC0-1.0",
            name="BohemiaSaxony",
            repository_lookup=False,
            # There is no precompiled version, the dataset is always built from FDSN
            compile_from_source=compile_from_source,
            **kwargs,
        )

    def _init_client(self):
        self._client = MultiClient(
            [
                FDSNSource(URL_MAPPINGS["BGR"], connections=self._connections),
                FDSNSource(URL_MAPPINGS["LMU"], connections=self._connections),
                FDSNSource(
                    URL_MAPPINGS["GEOFON"],
                    connections=self._connections,
                    eida_token=self._eida_token,
                ),
            ]
        )

    @staticmethod
    def _download_catalog_colm(path: Path) -> Path:
        files = list((path / "final2").glob("cll_*.txt"))
        if files:
            logger.debug("Catalog files already exist, skipping download.")
            return path
        logger.info("Downloading Bohemia Saxony catalog files.")
        catalog_url = urljoin(seisbench.remote_root, "auxiliary/collm-catalog.zip")
        catalog_zip = path / "collm-catalog.zip"
        try:
            download_http(
                catalog_url,
                catalog_zip,
                desc="Downloading Bohemia Saxony catalog",
            )
            with ZipFile(catalog_zip, "r") as zip_file:
                zip_file.extractall(path)
        finally:
            catalog_zip.unlink(missing_ok=True)
        # Fixup
        bad_file = path / "final2" / "cll_2019.txt"
        bad_file.write_text(bad_file.read_text().replace("GRZ!", "GRZ1"))

        return path

    async def get_inventory(
        self,
        catalog: Catalog,
        force_download: bool = False,
    ) -> Inventory:
        """
        Downloads the inventory of all stations in the catalog, or loads it from
        the dataset folder. Must be called within the client's context.
        """
        return await get_cached_inventory(
            self._client,
            self.path / "inventory.xml",
            stations=get_stations(catalog),
            # starttime=starttime,
            # endtime=endtime,
            force_download=force_download,
        )

    @classmethod
    def get_catalog(cls) -> Catalog:
        path = cls._path_internal()
        cls._download_catalog_colm(path)
        nordic_files = sorted((path / "final2").glob("cll_*.txt"))

        catalog = Catalog()
        for file in nordic_files:
            logger.info(f"Reading NORDIC file {file}")
            with warnings.catch_warnings():
                # obspy has no field for fixed depths, get_event_params handles them
                warnings.filterwarnings(
                    "ignore", message="Depth indicator F has not been mapped"
                )
                catalog += read_nordic(str(file))

        logger.info("Loaded catalog with %d events.", len(catalog))
        return catalog

    async def get_station_waveform_data(
        self,
        event_params: EventParameters,
        picks: list[Pick],
        inventory: Inventory,
        sampling_rate: float = 100.0,
        time_before: float = 60.0,
        time_after: float = 60.0,
    ) -> tuple[TraceParameters, np.ndarray]:
        waveform_id = picks[0].waveform_id

        trace_params = get_trace_params(
            waveform_id,
            inventory,
            event_params,
        )

        tmin = min(p.time for p in picks) - time_before
        tmax = max(p.time for p in picks) + time_after

        request = WaveformRequest(
            network=waveform_id.network_code,
            station=waveform_id.station_code,
            location=waveform_id.location_code or "",
            channel=waveform_id.channel_code[:2] + "*",
            starttime=tmin,
            endtime=tmax,
        )
        _, waveform_params, data = await get_waveform_array(
            self._client,
            inventory,
            request,
            sampling_rate=sampling_rate,
            component_order=COMPONENT_ORDER,
        )
        trace_params.update(waveform_params)
        actual_t_start = UTCDateTime(trace_params["trace_start_time"])
        sampling_rate = trace_params["trace_sampling_rate_hz"]

        trace_params["trace_name"] = get_trace_name(event_params, waveform_id)
        trace_params["trace_category"] = "earthquake"

        for pick in picks:
            sample = round((pick.time - actual_t_start) * sampling_rate)
            if not 0 <= sample < data.shape[1]:
                # E.g., the data end early: the label would be outside the waveform
                raise ValueError(f"The {pick.phase_hint} pick is outside the data.")
            trace_params[f"trace_{pick.phase_hint}_arrival_sample"] = sample
            trace_params[f"trace_{pick.phase_hint}_status"] = pick.evaluation_mode
            if pick.polarity is None:
                trace_params[f"trace_{pick.phase_hint}_polarity"] = "undecidable"
            else:
                trace_params[f"trace_{pick.phase_hint}_polarity"] = pick.polarity

        return trace_params, data

    async def get_noise_data(
        self, job: NoiseJob, inventory: Inventory
    ) -> tuple[TraceParameters, np.ndarray]:
        """
        Downloads the first candidate window of the job that has data and no gaps.

        :raises ValueError: If no candidate window has data without gaps
        """
        waveform_id = job.waveform_id
        reasons = []
        for starttime in job.starttimes:
            request = WaveformRequest(
                network=waveform_id.network_code,
                station=waveform_id.station_code,
                location=waveform_id.location_code or "",
                channel=waveform_id.channel_code[:2] + "*",
                starttime=starttime,
                endtime=starttime + self._noise_length,
            )
            try:
                seed_id, waveform_params, data = await get_waveform_array(
                    self._client,
                    inventory,
                    request,
                    sampling_rate=SAMPLING_RATE,
                    component_order=COMPONENT_ORDER,
                )
                check_complete(data, waveform_params["trace_completeness"])
            except (FDSNNoDataException, ValueError) as exc:
                reasons.append(exc)
                continue
            # The seed id has the full channel code, the job only the band and instrument
            trace_params = get_station_params(
                WaveformStreamID(seed_string=seed_id), inventory, starttime
            )
            trace_params.update(waveform_params)
            trace_params["trace_name"] = job.trace_name
            trace_params["trace_category"] = "noise"
            return trace_params, data
        raise ValueError(
            f"No candidate window with data and without gaps, last: {reasons[-1]}"
        )

    @classmethod
    def planned_trace_names(cls) -> set[str]:
        """
        The names of the event traces the catalog plans, offline, with the inventory
        in the dataset folder. Reads the whole catalog, which takes about a minute.
        """
        inventory = read_inventory(str(cls._path_internal() / "inventory.xml"))
        catalog = fixup_catalog(cls.get_catalog(), inventory)
        return {_job_name(job) for job in plan_event_jobs(catalog)}

    def _noise_jobs(
        self,
        client: MultiClient,
        anchors: list[Anchor],
        noise: Counter,
        tried: set[str],
        origin_times: list[float],
        rng: np.random.Generator,
    ) -> list[tuple[Source, NoiseJob]]:
        """
        Noise jobs for the splits with less noise than ``noise_fraction`` of their
        event traces, as many as are missing. Each is drawn at a random anchor of the
        split that wasn't tried before.

        :param noise: Noise traces in the dataset per split
        :param tried: Names of the anchors already used, extended by this call
        :param origin_times: Sorted origin timestamps of all catalog events
        """
        swarms = load_swarms()
        events = Counter(anchor.split for anchor in anchors)
        jobs = []
        n_no_quiet = 0
        for split in sorted(events):
            missing = round(self._noise_fraction * events[split]) - noise[split]
            if missing <= 0:
                continue
            pool = sorted(
                anchor
                for anchor in anchors
                if anchor.split == split and anchor.trace_name not in tried
            )
            for i in rng.permutation(len(pool)):
                if missing == 0:
                    break
                anchor = pool[i]
                tried.add(anchor.trace_name)
                try:
                    source = client.get_source(anchor.network)
                except KeyError:
                    continue
                starttimes = draw_noise_starttimes(
                    rng,
                    UTCDateTime(anchor.origin_time),
                    self._noise_length,
                    swarms,
                    origin_times,
                )
                if not starttimes:
                    n_no_quiet += 1
                    continue
                noise_job = NoiseJob(
                    trace_name=f"noise_{anchor.trace_name}",
                    waveform_id=WaveformStreamID(
                        anchor.network, anchor.station, anchor.location, anchor.channel
                    ),
                    split=split,
                    starttimes=tuple(starttimes),
                )
                jobs.append((source, noise_job))
                missing -= 1
        if n_no_quiet:
            logger.info("%d anchors have no quiet period nearby.", n_no_quiet)
        return jobs

    def _download_dataset(
        self,
        writer,
        time_before: float = 60.0,
        time_after: float = 60.0,
        **kwargs,
    ):
        logger.info(
            "No pre-downloaded dataset available, downloading from BGR and Geofon. "
            "This may take a while.",
        )
        if not self._eida_token:
            logger.warning(
                "No EIDA token provided. "
                "Restricted WB network data will not be accessible.",
            )
        self._init_client()
        writer.data_format = {
            "dimension_order": "CW",
            "component_order": COMPONENT_ORDER,
            "measurement": "velocity",
            "unit": "counts",
            "instrument_response": "not restituted",
        }

        cat = self.get_catalog()
        asyncio.run(
            self._download_waveforms(
                writer, cat, time_before=time_before, time_after=time_after
            )
        )

    async def _download_waveforms(
        self,
        writer,
        catalog: Catalog,
        time_before: float,
        time_after: float,
    ):
        """
        Downloads the waveforms of all stations in the catalog, see
        :py:func:`~seisbench_datasets.data.multiclient.download_parallel`.

        When resuming, the traces already written are skipped. Source ids and splits
        are deterministic, so they match the earlier run.
        """
        done = writer.resumed_trace_names
        if done:
            logger.info("Resuming: %d traces are already downloaded.", len(done))
        # All catalog events, also those fixup_catalog removes, delimit quiet periods
        origin_times = sorted(
            event.preferred_origin().time.timestamp
            for event in catalog
            if event.preferred_origin() is not None
        )
        # Datasets compiled before the noise traces have no trace category
        writer.set_default("trace_category", "earthquake")
        anchors, noise = anchors_from_metadata(writer.resumed_metadata)
        tried = {
            name.removeprefix("noise_") for name in done if name.startswith("noise_")
        }

        async with self._client as client:
            # A resumed download keeps the inventory, so stations map to the same
            # networks as before
            inv = await self.get_inventory(catalog, force_download=not done)
            catalog = fixup_catalog(catalog, inv)

            jobs = []
            for job in plan_event_jobs(catalog):
                if _job_name(job) in done:
                    continue
                network = job[2][0].waveform_id.network_code
                try:
                    jobs.append((client.get_source(network), job))
                except KeyError as exc:
                    logger.warning("Skipping station: %s", exc)

            async def download(job):
                if isinstance(job, NoiseJob):
                    return await self.get_noise_data(job, inv)
                _, event_params, picks = job
                return await self.get_station_waveform_data(
                    event_params=event_params,
                    picks=picks,
                    inventory=inv,
                    time_before=time_before,
                    time_after=time_after,
                )

            n_noise_failed = 0

            def write(job, result, exc):
                nonlocal n_noise_failed
                if isinstance(job, NoiseJob):
                    if exc is not None:
                        logger.debug("No noise trace %s: %s", job.trace_name, exc)
                        n_noise_failed += 1
                        return
                    trace_params, data = result
                    writer.add_trace({"split": job.split, **trace_params}, data)
                    noise[job.split] += 1
                    return
                event, event_params, picks = job
                if exc is not None:
                    logger.warning(
                        "Error processing station %s of event %s: %s",
                        picks[0].waveform_id.get_seed_string(),
                        event.short_str(),
                        exc,
                    )
                    return
                trace_params, data = result
                writer.add_trace({**event_params, **trace_params}, data)
                waveform_id = picks[0].waveform_id
                anchors.append(
                    Anchor(
                        trace_name=trace_params["trace_name"],
                        split=event_params["split"],
                        origin_time=event_params["source_origin_time"],
                        network=waveform_id.network_code,
                        station=waveform_id.station_code,
                        location=waveform_id.location_code or "",
                        channel=waveform_id.channel_code[:2],
                    )
                )

            await download_parallel(
                jobs, download, write, checkpoint=writer.checkpoint, title="Events"
            )

            # Not the generator of the splits, so that the event splits don't change
            rng = np.random.default_rng(RNG_SEED + 1)
            for i_round in range(NOISE_MAX_ROUNDS):
                jobs = self._noise_jobs(
                    client, anchors, noise, tried, origin_times, rng
                )
                if not jobs:
                    break
                logger.info(
                    "Noise round %d: %d noise traces to reach the noise fraction.",
                    i_round + 1,
                    len(jobs),
                )
                await download_parallel(
                    jobs,
                    download,
                    write,
                    checkpoint=writer.checkpoint,
                    title=f"Noise round {i_round + 1}",
                )
            events = Counter(anchor.split for anchor in anchors)
            logger.info(
                "Noise traces per split: %s. %d without a window with data and "
                "without gaps.",
                ", ".join(
                    f"{split} {noise[split]} ({noise[split] / events[split]:.1%})"
                    for split in sorted(events)
                ),
                n_noise_failed,
            )


def plan_event_jobs(catalog: Catalog) -> list[EventJob]:
    """
    One job per event and station of the catalog, after :py:func:`fixup_catalog`.
    Source ids and splits are deterministic, so a resumed download plans the same.
    """
    # A new generator for every plan, so the splits are the same in a resumed download
    rng = np.random.default_rng(RNG_SEED)
    jobs = []
    source_ids = Counter()
    for event in catalog:
        split = get_split(rng)
        try:
            event_params = get_event_params(event)
        except ValueError as exc:
            logger.warning("Skipping event %s: %s", event.short_str(), exc)
            continue
        event_params["split"] = split
        # The catalog lists a few events twice
        source_ids[event_params["source_id"]] += 1
        if source_ids[event_params["source_id"]] > 1:
            event_params["source_id"] += f"_{source_ids[event_params['source_id']]}"

        # groupby only groups adjacent picks, so sort them by station first
        picks = sorted(event.picks, key=_station_key)
        for _, pick_group in groupby(picks, key=_station_key):
            jobs.append((event, event_params, list(pick_group)))
    return jobs


def anchors_from_metadata(metadata: pd.DataFrame) -> tuple[list[Anchor], Counter]:
    """
    The event traces of a dataset as anchors for noise traces, and the number of
    noise traces per split. Traces without a category are event traces.

    :param metadata: Metadata as text, e.g., ``ResumableWriter.resumed_metadata``
    """
    if metadata.empty:
        return [], Counter()
    names = metadata["trace_name"]
    if "trace_name_original" in metadata:
        original = metadata["trace_name_original"]
        names = original.where(original != "", names)
    category = metadata.get("trace_category", pd.Series("", index=metadata.index))
    is_noise = category == "noise"
    noise = Counter(metadata.loc[is_noise, "split"])
    events = metadata[~is_noise]
    anchors = [
        Anchor(*row)
        for row in zip(
            names[~is_noise],
            events["split"],
            events["source_origin_time"],
            events["station_network_code"],
            events["station_code"],
            events["station_location_code"],
            events["trace_channel"],
            strict=True,
        )
    ]
    return anchors, noise


def get_trace_name(event_params: EventParameters, waveform_id: WaveformStreamID) -> str:
    return (
        f"{event_params['source_id']}_{waveform_id.network_code}."
        f"{waveform_id.station_code}.{waveform_id.location_code or ''}."
        f"{waveform_id.channel_code[:2]}"
    )


def _job_name(job) -> str:
    if isinstance(job, NoiseJob):
        return job.trace_name
    _, event_params, picks = job
    return get_trace_name(event_params, picks[0].waveform_id)


def load_swarms() -> list[tuple[float, float]]:
    """
    The swarm windows of the catalog, see ``scripts/bohemia_swarms.py``, as sorted
    ``(start, end)`` timestamps.
    """
    text = resources.files(__package__).joinpath("bohemia_swarms.csv").read_text()
    rows = csv.DictReader(
        line for line in text.splitlines() if not line.startswith("#")
    )
    return sorted(
        (UTCDateTime(row["start"]).timestamp, UTCDateTime(row["end"]).timestamp)
        for row in rows
    )


def in_swarm(swarms: list[tuple[float, float]], start: float, end: float) -> bool:
    """
    Whether the window from ``start`` to ``end`` overlaps a swarm window. ``swarms``
    are sorted and don't overlap.
    """
    i = bisect.bisect_left(swarms, (end,))
    return i > 0 and swarms[i - 1][1] > start


def draw_noise_starttimes(
    rng: np.random.Generator,
    time: UTCDateTime,
    length: float,
    swarms: list[tuple[float, float]],
    origin_times: list[float],
    n: int = NOISE_CANDIDATES,
    span: float = NOISE_SPAN,
) -> list[UTCDateTime]:
    """
    Random start times of noise windows within ``span`` of ``time``, in quiet
    periods: outside the swarm windows, away from catalog events, and between the
    first and the last catalog event. Draws a fixed number of random numbers, so
    that the times of later windows don't depend on this one.

    :param origin_times: Sorted origin timestamps of all catalog events
    :return: Up to ``n`` start times, fewer if the quiet periods are too short
    """
    draws = rng.uniform(size=NOISE_MAX_DRAWS)
    low = max(time.timestamp - span, origin_times[0])
    high = min(time.timestamp + span, origin_times[-1]) - length
    if high <= low:
        return []
    starttimes = []
    for start in low + draws * (high - low):
        start = UTCDateTime(round(start, 2))
        end = start + length
        if in_swarm(swarms, start.timestamp, end.timestamp):
            continue
        if not is_away_from_events(origin_times, start, end):
            continue
        starttimes.append(start)
        if len(starttimes) == n:
            break
    return starttimes


def _station_key(pick: Pick) -> tuple[str, str, str]:
    waveform_id = pick.waveform_id
    return (
        waveform_id.network_code or "",
        waveform_id.station_code or "",
        waveform_id.location_code or "",
    )


def get_catalog_timerange(catalog: Catalog) -> tuple[datetime, datetime]:
    """
    Get the time range of the catalog.

    :param catalog: Catalog to get the time range from
    :return: Start and end time of the catalog
    """
    tmin = datetime.max
    tmax = datetime.min

    for ev in catalog:
        ev = cast(Event, ev)
        origin = ev.preferred_origin()
        tmin = min(tmin, origin.time.datetime)
        tmax = max(tmax, origin.time.datetime)

    return tmin, tmax


def get_stations(catalog: Catalog) -> set[str]:
    """
    Get the networks and stations from the catalog.

    :param catalog: Catalog to get the networks and stations from
    :return: Dictionary with networks as keys and sets of stations as values
    """
    stations = set()

    for ev in catalog:
        ev = cast(Event, ev)
        for pick in ev.picks:
            if pick.waveform_id is not None:
                station = pick.waveform_id.station_code
                # Request the FDSN station names that fixup_catalog maps to
                stations.add(STATION_MAPPING.get(station, station))

    return stations


def _uncertainty(errors: QuantityError | None) -> float:
    if errors is None or errors.uncertainty is None:
        return np.nan
    return errors.uncertainty


def get_event_params(event: Event) -> EventParameters:
    origin = event.preferred_origin()
    magnitude = event.preferred_magnitude()

    if origin is None:
        raise ValueError("Event has no preferred origin.")

    # The Nordic reader gives every event a new random resource id. The origin time
    # is deterministic, so that an interrupted download can resume.
    sb_id = "bs_" + origin.time.strftime("%Y%m%dT%H%M%S.%f")[:-4]

    # The Nordic reader stores the horizontal errors in degrees
    latitude_uncertainty_km = degrees2kilometers(_uncertainty(origin.latitude_errors))
    longitude_uncertainty_km = degrees2kilometers(
        _uncertainty(origin.longitude_errors)
    ) * np.cos(np.radians(origin.latitude))

    # A depth error of 0 marks a depth fixed during location (Nordic depth
    # indicator F), so the uncertainty is unknown, not zero
    depth_uncertainty_km = _uncertainty(origin.depth_errors) / 1e3
    if depth_uncertainty_km == 0:
        depth_uncertainty_km = np.nan

    params: EventParameters = EventParameters(
        split="train",
        source_id=sb_id,
        source_origin_time=str(origin.time),
        source_origin_uncertainty_sec=_uncertainty(origin.time_errors),
        source_latitude_deg=origin.latitude,
        source_latitude_uncertainty_km=latitude_uncertainty_km,
        source_longitude_deg=origin.longitude,
        source_longitude_uncertainty_km=longitude_uncertainty_km,
        source_depth_km=np.nan if origin.depth is None else origin.depth / 1e3,
        source_depth_uncertainty_km=depth_uncertainty_km,
    )
    if magnitude is not None:
        params["source_magnitude"] = magnitude.mag
        params["source_magnitude_uncertainty"] = _uncertainty(magnitude.mag_errors)
        params["source_magnitude_type"] = magnitude.magnitude_type
        if magnitude.creation_info is not None:
            params["source_magnitude_author"] = magnitude.creation_info.agency_id

    return params


def get_station_params(
    waveform_id: WaveformStreamID, inventory: Inventory, time: UTCDateTime
) -> TraceParameters:
    try:
        coordinates = inventory.get_coordinates(
            waveform_id.get_seed_string(), datetime=time
        )
    except Exception as r:
        raise ValueError(
            f"Could not get coordinates for {waveform_id.get_seed_string()}: {r}"
        ) from r

    return TraceParameters(
        trace_name="foo",
        station_network_code=waveform_id.network_code,
        station_code=waveform_id.station_code,
        trace_channel=waveform_id.channel_code[:2],
        station_location_code=waveform_id.location_code,
        station_latitude_deg=coordinates["latitude"],
        station_longitude_deg=coordinates["longitude"],
        station_elevation_m=coordinates["elevation"],
    )


def get_trace_params(
    waveform_id: WaveformStreamID,
    inventory: Inventory,
    event_params: EventParameters,
) -> TraceParameters:
    trace_params = get_station_params(
        waveform_id, inventory, UTCDateTime(event_params["source_origin_time"])
    )
    latitude = trace_params["station_latitude_deg"]
    longitude = trace_params["station_longitude_deg"]
    if not np.isnan(latitude * longitude):
        trace_params["path_back_azimuth_deg"] = gps2dist_azimuth(
            event_params["source_latitude_deg"],
            event_params["source_longitude_deg"],
            latitude,
            longitude,
        )[2]
    else:
        trace_params["path_back_azimuth_deg"] = np.nan
    return trace_params


def fixup_catalog(catalog: Catalog, inventory: Inventory) -> Catalog:
    """
    Fixes the channel codes of the picks.
    """
    station_index = get_station_index(inventory)
    for event in catalog.events.copy():
        origin = event.preferred_origin()
        if origin is None:
            logger.debug("Removing event %s without origin.", event.resource_id)
            catalog.events.remove(event)
            continue
        origin_time = origin.time
        for pick in event.picks.copy():
            if pick.phase_hint not in PICK_WHITELIST:
                event.picks.remove(pick)
                continue

            waveform_id = pick.waveform_id

            try:
                waveform_id.channel_code = "%sH%s" % tuple(waveform_id.channel_code)  # noqa: UP031, raises TypeError unless 2 characters
            except TypeError:
                # This happens if the channel code is already in the correct format.
                event.picks.remove(pick)
                continue

            # Fix station names to FDSN
            waveform_id.station_code = STATION_MAPPING.get(
                waveform_id.station_code,
                waveform_id.station_code,
            )

            try:
                waveform_id.network_code = get_network_code(
                    waveform_id.station_code, station_index, origin_time
                )
            except KeyError:
                event.picks.remove(pick)
                continue

            if waveform_id.network_code == "WB":
                if origin_time.datetime < WEBNET_SWITCH_SH_CH_CHANNEL:
                    waveform_id.channel_code = f"SH{waveform_id.channel_code[-1]}"
                else:
                    waveform_id.channel_code = f"CH{waveform_id.channel_code[-1]}"

            if waveform_id.channel_code in REMOVE_CHANNELS:
                event.picks.remove(pick)

        if not event.picks:
            logger.debug("Removing event %s with no picks.", event.resource_id)
            catalog.events.remove(event)

    return catalog


StationIndex = dict[str, list[tuple[str, UTCDateTime | None, UTCDateTime | None]]]


def get_station_index(inventory: Inventory) -> StationIndex:
    """
    Maps each station code to the networks and epochs it appears in.
    """
    index: StationIndex = defaultdict(list)
    for network in inventory:
        for station in network:
            index[station.code].append(
                (network.code, station.start_date, station.end_date)
            )
    return index


def is_temporary_network(network_code: str) -> bool:
    # FDSN convention: temporary network codes start with a digit or X, Y, Z
    return network_code[0].isdigit() or network_code[0] in "XYZ"


def get_network_code(
    station_code: str,
    station_index: StationIndex,
    time: UTCDateTime | None = None,
) -> str:
    """
    Resolves the network of a station code. Some codes exist in several networks,
    e.g., LAC in WB and ZB. Only networks where the station operates at the given
    time are considered. Of those, the dataset's networks are preferred, then other
    permanent networks, then temporary ones.
    """
    epochs = station_index.get(station_code)
    if not epochs:
        raise KeyError(f"Unknown network code for station {station_code}.")

    networks = {
        network
        for network, start, end in epochs
        if time is None
        or ((start is None or start <= time) and (end is None or time <= end))
    }
    if not networks:
        raise KeyError(f"Station {station_code} is not operating at {time}.")

    def priority(network: str) -> tuple[int, int, str]:
        if network in PREFERRED_NETWORKS:
            return (0, PREFERRED_NETWORKS.index(network), network)
        return (1 + is_temporary_network(network), 0, network)

    return min(networks, key=priority)


def get_split(
    rng: np.random.Generator, test: float = 0.1, dev: float = 0.1
) -> Literal["train", "dev", "test"]:
    """
    Returns a random split for the dataset.
    """
    r = rng.random()
    train = 1.0 - test - dev
    if r < train:
        return "train"
    elif r < train + dev:
        return "dev"
    else:
        return "test"
