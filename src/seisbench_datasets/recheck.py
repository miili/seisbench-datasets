"""
Rechecks the noise traces of a compiled dataset: removes those with an earthquake,
e.g., noise compiled before the earthquake detection, and those above the noise
fraction, from the metadata. The next compilation tops the noise up again.
"""

from __future__ import annotations

import csv
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import h5py
import numpy as np
import pandas as pd
from rich.progress import track

from seisbench_datasets.data.noise import EarthquakeDetector
from seisbench_datasets.data.resume import trace_names, try_lock
from seisbench_datasets.progress import console

NOISE_REJECTED_FILE = "noise_rejected.csv"


EARTHQUAKE = "earthquake"
ABOVE_FRACTION = "above noise fraction"


class RejectedNoise(NamedTuple):
    trace_name: str
    split: str
    reason: str
    #: P or S probability, NaN if not checked
    probability: float


class RecheckResult(NamedTuple):
    #: Noise traces checked
    checked: int
    #: Noise traces removed, with an earthquake
    removed: list[RejectedNoise]
    #: Copy of the metadata before the removal, None if nothing was removed
    backup: Path | None


def read_waveforms(file: h5py.File, trace_name: str) -> np.ndarray:
    """
    The waveforms of a trace, by its name in the metadata: a dataset of its own,
    or ``bucket<i>$<row>,:<components>,:<samples>`` in a bucket.
    """
    name, _, location = trace_name.partition("$")
    if not location:
        return file["data"][name][()]
    row, *shape = location.split(",")
    index = (int(row), *(slice(None, int(size.lstrip(":"))) for size in shape))
    return file["data"][name][index]


def recheck_noise(
    files: list[Path],
    detector: EarthquakeDetector,
    threshold: float,
    noise_fraction: float | None = None,
    group: str = "split",
    seed: int = 0,
) -> RecheckResult:
    """
    Runs ``detector`` over the noise traces and removes those with a P or S
    probability above ``threshold`` from the metadata. Then removes random noise
    traces of the groups with more noise than ``noise_fraction`` of their event
    traces. The waveforms stay in the HDF5 file, traces in buckets can't be removed
    alone. The metadata before is kept as ``metadata.csv.bak-<time>``, the removed
    traces are added to ``noise_rejected.csv``.

    :param files: Metadata and waveforms file of the dataset
    :param noise_fraction: Noise traces per event trace in each group, None to keep
        any amount
    :param group: Metadata column of the groups, e.g., ``split`` or
        ``source_region``
    :param seed: Seed of the choice of the noise traces above the fraction
    :raises RuntimeError: If a compilation into the dataset is running
    """
    metadata_path, waveforms_path = files
    lock = try_lock(metadata_path)
    if lock is None:
        raise RuntimeError(f"A compilation into {metadata_path.parent} is running.")
    try:
        metadata = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
        if "trace_category" not in metadata:
            return RecheckResult(0, [], None)
        names = trace_names(metadata)
        noise = np.flatnonzero(metadata["trace_category"] == "noise")

        removed = []
        drop = []
        with h5py.File(waveforms_path, "r") as file:
            for i in track(noise, description="Rechecking noise", console=console):
                data = read_waveforms(file, metadata["trace_name"].iloc[i])
                probability = detector.max_probability(data)
                if probability > threshold:
                    drop.append(i)
                    removed.append(
                        RejectedNoise(
                            names.iloc[i],
                            metadata["split"].iloc[i],
                            EARTHQUAKE,
                            probability,
                        )
                    )

        if noise_fraction is not None:
            is_noise = metadata["trace_category"] == "noise"
            events = Counter(metadata.loc[~is_noise, group])
            dropped = set(drop)
            kept = defaultdict(list)
            for i in noise:
                if i not in dropped:
                    kept[metadata[group].iloc[i]].append(i)
            rng = np.random.default_rng(seed)
            for key in sorted(kept):
                excess = len(kept[key]) - round(noise_fraction * events[key])
                if excess <= 0:
                    continue
                for i in np.sort(rng.choice(kept[key], size=excess, replace=False)):
                    drop.append(i)
                    removed.append(
                        RejectedNoise(
                            names.iloc[i],
                            metadata["split"].iloc[i],
                            ABOVE_FRACTION,
                            float("nan"),
                        )
                    )

        backup = None
        if drop:
            backup = metadata_path.with_name(
                f"{metadata_path.name}.bak-{datetime.now():%Y%m%dT%H%M%S}"
            )
            shutil.copy2(metadata_path, backup)
            tmp_path = metadata_path.with_name(metadata_path.name + ".tmp")
            metadata.drop(index=metadata.index[drop]).to_csv(tmp_path, index=False)
            tmp_path.replace(metadata_path)

            log = metadata_path.parent / NOISE_REJECTED_FILE
            new = not log.is_file()
            with open(log, "a", newline="") as f:
                writer = csv.writer(f)
                if new:
                    writer.writerow([*RejectedNoise._fields, "weights"])
                for rejected in removed:
                    writer.writerow(
                        [
                            rejected.trace_name,
                            rejected.split,
                            rejected.reason,
                            ""
                            if np.isnan(rejected.probability)
                            else f"{rejected.probability:.3f}",
                            detector.weights,
                        ]
                    )
        return RecheckResult(len(noise), removed, backup)
    finally:
        lock.close()
