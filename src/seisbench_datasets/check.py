"""
Checks of a compiled dataset: the files, the traces per split, the noise fraction,
and, if the dataset can plan its traces, the traces that are missing.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import NamedTuple

import h5py
import pandas as pd
from rich.table import Table

from seisbench_datasets.progress import console

COLUMNS = {"trace_name", "trace_name_original", "trace_category", "split"}


class SplitCounts(NamedTuple):
    split: str
    events: int
    noise: int

    @property
    def noise_fraction(self) -> float:
        return self.noise / self.events if self.events else float("nan")


class CheckReport(NamedTuple):
    path: Path
    traces: int
    #: Traces whose waveforms are not in the HDF5 file
    unreadable: list[str]
    splits: list[SplitCounts]
    #: The noise fraction the dataset aims for
    noise_fraction: float
    #: Planned event traces that are not in the dataset, None if not checked
    missing: list[str] | None = None
    #: Event traces in the dataset that are not planned, None if not checked
    extra: list[str] | None = None

    @property
    def ok(self) -> bool:
        return not self.unreadable


def trace_names(metadata: pd.DataFrame) -> pd.Series:
    """
    The original names of the traces, also of those written into buckets.
    """
    names = metadata["trace_name"]
    if "trace_name_original" in metadata:
        original = metadata["trace_name_original"]
        names = original.where(original != "", names)
    return names


def check_dataset(
    path: Path,
    noise_fraction: float,
    planned: set[str] | None = None,
    partial: bool = False,
) -> CheckReport:
    """
    :param path: Dataset folder, with ``metadata.csv`` and ``waveforms.hdf5``
    :param planned: Names of the event traces the dataset should contain
    :param partial: Check the ``.partial`` files of an interrupted compilation, up
        to its last checkpoint
    """
    suffix = ".partial" if partial else ""
    metadata = pd.read_csv(
        path / f"metadata.csv{suffix}",
        dtype=str,
        keep_default_na=False,
        usecols=lambda column: column in COLUMNS,
    )
    names = trace_names(metadata)

    with h5py.File(path / f"waveforms.hdf5{suffix}", "r") as file:
        datasets = set(file["data"])
    # Traces in buckets are named bucket<i>$<location>
    unreadable = names[~metadata["trace_name"].str.split("$").str[0].isin(datasets)]

    category = metadata.get("trace_category", pd.Series("", index=metadata.index))
    is_noise = category == "noise"
    events = Counter(metadata.loc[~is_noise, "split"])
    noise = Counter(metadata.loc[is_noise, "split"])
    splits = [
        SplitCounts(split, events[split], noise[split])
        for split in sorted(events | noise)
    ]

    missing = extra = None
    if planned is not None:
        present = set(names[~is_noise])
        missing = sorted(planned - present)
        extra = sorted(present - planned)

    return CheckReport(
        path=path,
        traces=len(metadata),
        unreadable=list(unreadable),
        splits=splits,
        noise_fraction=noise_fraction,
        missing=missing,
        extra=extra,
    )


def _station(trace_name: str) -> str:
    # <source id>_<network>.<station>.<location>.<channel>
    return ".".join(trace_name.rsplit("_", 1)[-1].split(".")[:2])


def print_report(report: CheckReport) -> None:
    table = Table(title=f"{report.path}: {report.traces} traces")
    table.add_column("Split")
    table.add_column("Events", justify="right")
    table.add_column("Noise", justify="right")
    table.add_column(
        f"Noise fraction (target {report.noise_fraction:.0%})", justify="right"
    )
    total = SplitCounts(
        "total",
        sum(s.events for s in report.splits),
        sum(s.noise for s in report.splits),
    )
    for counts in (*report.splits, total):
        style = "bold" if counts is total else None
        table.add_row(
            counts.split,
            str(counts.events),
            str(counts.noise),
            f"{counts.noise_fraction:.1%}",
            style=style,
        )
    console.print(table)

    if report.unreadable:
        console.print(
            f"[red]{len(report.unreadable)} traces have no waveforms in the HDF5 "
            f"file[/], e.g., {', '.join(report.unreadable[:3])}"
        )
    else:
        console.print("[green]All traces have waveforms in the HDF5 file.")

    if report.missing is None:
        console.print("Missing traces: not checked, the dataset has no plan.")
        return
    if report.missing:
        stations = Counter(_station(name) for name in report.missing)
        console.print(
            f"[yellow]{len(report.missing)} planned event traces are missing[/], "
            "most at: "
            + ", ".join(f"{station} {n}" for station, n in stations.most_common(10))
        )
    else:
        console.print("[green]No planned event traces are missing.")
    if report.extra:
        console.print(
            f"[yellow]{len(report.extra)} event traces are not in the plan[/], e.g., "
            + ", ".join(report.extra[:3])
        )
