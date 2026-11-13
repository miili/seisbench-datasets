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

from seisbench_datasets.data.resume import NO_DATA_FILE, NoDataLog, trace_names
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
    #: The missing traces the data centers have no data for, see ``no_data.csv``
    no_data: list[str] | None = None

    @property
    def ok(self) -> bool:
        return not self.unreadable


def check_dataset(
    path: Path,
    noise_fraction: float,
    planned: set[str] | None = None,
) -> CheckReport:
    """
    :param path: Dataset folder, with ``metadata.csv`` and ``waveforms.hdf5``
    :param planned: Names of the event traces the dataset should contain
    """
    metadata = pd.read_csv(
        path / "metadata.csv",
        dtype=str,
        keep_default_na=False,
        usecols=lambda column: column in COLUMNS,
    )
    names = trace_names(metadata)

    with h5py.File(path / "waveforms.hdf5", "r") as file:
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

    missing = extra = no_data = None
    if planned is not None:
        present = set(names[~is_noise])
        missing = sorted(planned - present)
        extra = sorted(present - planned)
        log = NoDataLog(path / NO_DATA_FILE)
        no_data = [name for name in missing if name in log]

    return CheckReport(
        path=path,
        traces=len(metadata),
        unreadable=list(unreadable),
        splits=splits,
        noise_fraction=noise_fraction,
        missing=missing,
        extra=extra,
        no_data=no_data,
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
        no_data = set(report.no_data or ())
        other = [name for name in report.missing if name not in no_data]
        console.print(
            f"[yellow]{len(report.missing)} planned event traces are missing:[/] "
            f"{len(no_data)} without data at the data centers ({NO_DATA_FILE}, "
            f"skipped), {len(other)} failed otherwise or not tried yet."
        )
        for label, names in (("Without data", no_data), ("Other", other)):
            if names:
                stations = Counter(_station(name) for name in names)
                console.print(
                    f"  {label}, most at: "
                    + ", ".join(f"{s} {n}" for s, n in stations.most_common(10))
                )
    else:
        console.print("[green]No planned event traces are missing.")
    if report.extra:
        console.print(
            f"[yellow]{len(report.extra)} event traces are not in the plan[/], e.g., "
            + ", ".join(report.extra[:3])
        )
