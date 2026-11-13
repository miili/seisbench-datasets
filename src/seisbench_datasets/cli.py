"""
Command line interface, ``seisbench-datasets``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer

from seisbench_datasets.progress import console, setup_logging

app = typer.Typer(
    help="Compile SeisBench training datasets from catalogs and FDSN / Hi-net "
    "waveforms. A command continues the dataset from what is there: it downloads "
    "the planned traces that are missing and tops up the noise. Traces the data "
    "centers have no data for (HTTP 204 / 404) are recorded in no_data.csv and "
    "skipped; delete the file to try them again.",
    no_args_is_help=True,
)

EidaToken = Annotated[
    Path | None,
    typer.Option(
        exists=True,
        dir_okay=False,
        resolve_path=True,
        help="EIDA token file, for restricted EIDA data.",
    ),
]
Force = Annotated[
    bool,
    typer.Option(
        help="Compile anew. An existing dataset is moved to a backup folder next "
        "to it first."
    ),
]
Check = Annotated[
    bool,
    typer.Option(
        "--check",
        "--validate",
        help="Check the compiled dataset and download nothing: the waveforms of "
        "every trace, the noise fraction per split, and the planned traces that are "
        "missing.",
    ),
]
NoiseFraction = Annotated[
    float,
    typer.Option(min=0.0, help="Noise traces per event trace, in each split."),
]


@app.callback()
def main() -> None:
    setup_logging()


def _prepare(dataset: type, force: bool, check: bool, noise_fraction) -> bool:
    """
    Runs --check, or prepares --force. Returns True if the dataset is to be compiled.
    """
    from seisbench import logger

    from seisbench_datasets.data.resume import move_to_backup

    if force and check:
        raise typer.BadParameter("Use only one of --force and --check.")
    try:
        if check:
            _check(dataset, noise_fraction)
            return False
        if force:
            backup = move_to_backup(dataset._path_internal())
            if backup is not None:
                logger.info("Moved the existing dataset to %s", backup)
    except RuntimeError as exc:
        console.print(f"[red]{exc}")
        raise typer.Exit(1) from exc
    return True


def _check(dataset: type, noise_fraction: float) -> None:
    from seisbench_datasets.check import check_dataset, print_report
    from seisbench_datasets.data.resume import incomplete_marker, try_lock

    path = dataset._path_internal()
    files = dataset.files()
    # Files of an interrupted compilation of an earlier version
    partial = not all(file.is_file() for file in files) and all(
        (file.parent / (file.name + ".partial")).is_file() for file in files
    )
    if not partial and not all(file.is_file() for file in files):
        console.print(f"[red]{path} has no compiled dataset.")
        raise typer.Exit(1)
    lock_existed = Path(f"{files[0]}.lock").exists()
    lock = try_lock(files[0])
    if lock is None:
        console.print(f"[red]A compilation into {path} is running.")
        raise typer.Exit(1)
    lock.close()
    if not lock_existed:  # Read-only: leave no lock file behind
        Path(lock.name).unlink(missing_ok=True)
    if partial or incomplete_marker(files).exists():
        console.print(
            "[yellow]The compilation was interrupted, checking up to its last "
            "checkpoint. Run the command without --check to continue it."
        )
    with console.status("Planning the traces from the catalog"):
        planned = dataset.planned_trace_names()
    report = check_dataset(path, noise_fraction, planned, partial=partial)
    print_report(report)
    if not report.ok:
        raise typer.Exit(1)


@app.command()
def bohemia(
    eida_token: EidaToken = None,
    noise_fraction: NoiseFraction = 0.3,
    force: Force = False,
    check: Check = False,
) -> None:
    """
    BohemiaSaxony: local earthquakes in NW Bohemia / Vogtland and Saxony.
    """
    from seisbench_datasets.data.bohemia import BohemiaSaxony

    if not _prepare(BohemiaSaxony, force, check, noise_fraction):
        return
    dataset = BohemiaSaxony(
        eida_token=str(eida_token) if eida_token else None,
        noise_fraction=noise_fraction,
        update=True,
        component_order="ZNE",
    )
    console.print(f"{dataset}\nin {dataset.path}")


@app.command()
def dlf(
    catalog_dir: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            file_okay=False,
            resolve_path=True,
            help="Folder with the QuakeML catalogs, <region>.xml. Defaults to "
            "quakeml in the dataset folder.",
        ),
    ] = None,
    eida_token: EidaToken = None,
    hinet_user: Annotated[
        str | None,
        typer.Option(
            help="Hi-net user name, for the Japan region. The password is read "
            "from the HINET_PASSWORD environment variable, or asked for."
        ),
    ] = None,
    noise_fraction: NoiseFraction = 0.3,
    force: Force = False,
    check: Check = False,
) -> None:
    """
    DeepLowFrequency: deep low-frequency events below volcanoes, plus noise.
    """
    from seisbench_datasets.data.dlf import DeepLowFrequency

    if not _prepare(DeepLowFrequency, force, check, noise_fraction):
        return
    hinet_password = None
    if hinet_user is not None:
        hinet_password = os.environ.get("HINET_PASSWORD") or typer.prompt(
            f"Hi-net password for {hinet_user}", hide_input=True
        )
    dataset = DeepLowFrequency(
        catalog_dir=catalog_dir,
        eida_token=str(eida_token) if eida_token else None,
        hinet_user=hinet_user,
        hinet_password=hinet_password,
        noise_fraction=noise_fraction,
        update=True,
        component_order="ZNE",
    )
    console.print(f"{dataset}\nin {dataset.path}")
