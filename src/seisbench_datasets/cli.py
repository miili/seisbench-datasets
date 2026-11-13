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
Export = Annotated[
    Path | None,
    typer.Option(
        file_okay=False,
        resolve_path=True,
        help="Export the compiled dataset to this folder, with "
        "seisbench.data.inspection: MiniSEED per day, picks, events and stations for "
        "pyrocko. Needs pyrocko, pip install seisbench-datasets[export]. Downloads "
        "nothing.",
    ),
]
Report = Annotated[
    Path | None,
    typer.Option(
        file_okay=False,
        resolve_path=True,
        help="Write an interactive HTML report of the compiled dataset to this "
        "folder, index.html: events, stations, picks, splits and noise. Needs "
        "plotly, pip install seisbench-datasets[report]. Downloads nothing.",
    ),
]
RecheckNoise = Annotated[
    bool,
    typer.Option(
        help="Run the earthquake detection (PhaseNet) over the noise traces already in "
        "the dataset, and remove those with an earthquake from the metadata, so that "
        "the next run replaces them. The metadata before is kept as a backup, the "
        "removed traces are listed in noise_rejected.csv. Downloads nothing.",
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


def _prepare(
    dataset: type,
    module,
    force: bool,
    check: bool,
    recheck_noise: bool,
    noise_fraction: float,
    export: Path | None = None,
    report: Path | None = None,
) -> bool:
    """
    Runs --check, --recheck-noise, --export or --report, or prepares --force.
    Returns True if the dataset is to be compiled.

    :param module: The module of the dataset, with its noise detection settings
    """
    from seisbench import logger

    from seisbench_datasets.data.resume import move_to_backup

    if force + check + recheck_noise + (export is not None) + (report is not None) > 1:
        raise typer.BadParameter(
            "Use only one of --force, --check, --recheck-noise, --export and --report."
        )
    try:
        if report is not None:
            _report(dataset, module, report, noise_fraction)
            return False
        if export is not None:
            _export(dataset, export)
            return False
        if check:
            _check(dataset, noise_fraction)
            return False
        if recheck_noise:
            _recheck_noise(dataset, module, noise_fraction)
            return False
        if force:
            backup = move_to_backup(dataset._path_internal())
            if backup is not None:
                logger.info("Moved the existing dataset to %s", backup)
    except RuntimeError as exc:
        console.print(f"[red]{exc}")
        raise typer.Exit(1) from exc
    return True


def _compiled_files(dataset: type) -> list[Path]:
    """
    The files of the dataset, exits if it isn't compiled.
    """
    files = dataset.files()
    if not all(file.is_file() for file in files):
        console.print(f"[red]{dataset._path_internal()} has no compiled dataset.")
        raise typer.Exit(1)
    return files


def _status(files: list[Path]) -> str:
    """
    ``compiling``, ``incomplete`` (interrupted) or ``complete``. Read-only: leaves no
    lock file behind.
    """
    from seisbench_datasets.data.resume import incomplete_marker, try_lock

    lock_existed = Path(f"{files[0]}.lock").exists()
    lock = try_lock(files[0])
    if lock is None:
        return "compiling"
    lock.close()
    if not lock_existed:
        Path(lock.name).unlink(missing_ok=True)
    return "incomplete" if incomplete_marker(files).exists() else "complete"


def _export(dataset: type, directory: Path) -> None:
    from seisbench_datasets.data.resume import try_lock

    try:
        from seisbench.data.inspection import DatasetInspection
    except ImportError as exc:
        console.print(
            "[red]Exporting needs pyrocko: pip install seisbench-datasets[export]"
        )
        raise typer.Exit(1) from exc

    files = _compiled_files(dataset)
    lock = try_lock(files[0])
    if lock is None:
        console.print(f"[red]A compilation into {dataset._path_internal()} is running.")
        raise typer.Exit(1)
    try:
        loaded = dataset(compile_from_source=False, component_order="ZNE")
        DatasetInspection(loaded).export(directory)
    finally:
        lock.close()
    console.print(f"Exported {dataset.__name__} to {directory}")


def _report(dataset: type, module, directory: Path, noise_fraction: float) -> None:
    try:
        from seisbench_datasets.report import write_report
    except ImportError as exc:
        console.print(
            "[red]The report needs plotly: pip install seisbench-datasets[report]"
        )
        raise typer.Exit(1) from exc

    files = dataset.files()
    path = dataset._path_internal()
    if not files[0].is_file():  # The report reads the metadata only
        console.print(f"[red]{path} has no compiled dataset.")
        raise typer.Exit(1)
    # A running compilation replaces metadata.csv at each checkpoint, atomically,
    # so the report shows it up to the last one
    status = _status(files)
    with console.status("Writing the report"):
        index = write_report(
            path,
            directory,
            title=dataset.__name__,
            group=module.NOISE_FRACTION_GROUP,
            noise_fraction=noise_fraction,
            status=status,
        )
    console.print(f"Wrote the report of {dataset.__name__} to {index}")


def _recheck_noise(dataset: type, module, noise_fraction: float) -> None:
    from collections import Counter

    from seisbench_datasets.check import check_dataset, print_report
    from seisbench_datasets.data.noise import EarthquakeDetector
    from seisbench_datasets.recheck import (
        ABOVE_FRACTION,
        EARTHQUAKE,
        NOISE_REJECTED_FILE,
        recheck_noise,
    )

    files = _compiled_files(dataset)
    weights = module.NOISE_PHASENET_WEIGHTS
    detector = EarthquakeDetector(weights, module.COMPONENT_ORDER)
    group = module.NOISE_FRACTION_GROUP
    result = recheck_noise(
        files,
        detector,
        module.NOISE_MAX_PHASE_PROBABILITY,
        noise_fraction=noise_fraction,
        group=group,
    )

    def per_split(reason):
        counts = Counter(r.split for r in result.removed if r.reason == reason)
        n = sum(counts.values())
        return f"{n}" + (
            " (" + ", ".join(f"{s} {c}" for s, c in sorted(counts.items())) + ")"
            if n
            else ""
        )

    console.print(
        f"Checked {result.checked} noise traces with PhaseNet {weights}. Removed: "
        f"{per_split(EARTHQUAKE)} with an earthquake (P or S above "
        f"{module.NOISE_MAX_PHASE_PROBABILITY}), {per_split(ABOVE_FRACTION)} above "
        f"the noise fraction {noise_fraction:.0%} per {group}."
    )
    if result.removed:
        console.print(
            f"The metadata before is in {result.backup.name}, the removed traces are "
            f"listed in {NOISE_REJECTED_FILE}. Run the command without "
            "--recheck-noise to replace them."
        )
    print_report(check_dataset(dataset._path_internal(), noise_fraction))


def _check(dataset: type, noise_fraction: float) -> None:
    from seisbench_datasets.check import check_dataset, print_report

    path = dataset._path_internal()
    status = _status(_compiled_files(dataset))
    if status == "compiling":
        console.print(f"[red]A compilation into {path} is running.")
        raise typer.Exit(1)
    if status == "incomplete":
        console.print(
            "[yellow]The compilation was interrupted, checking up to its last "
            "checkpoint. Run the command without --check to continue it."
        )
    with console.status("Planning the traces from the catalog"):
        planned = dataset.planned_trace_names()
    report = check_dataset(path, noise_fraction, planned)
    print_report(report)
    if not report.ok:
        raise typer.Exit(1)


@app.command()
def bohemia(
    eida_token: EidaToken = None,
    noise_fraction: NoiseFraction = 0.3,
    force: Force = False,
    check: Check = False,
    recheck_noise: RecheckNoise = False,
    export: Export = None,
    report: Report = None,
) -> None:
    """
    BohemiaSaxony: local earthquakes in NW Bohemia / Vogtland and Saxony.
    """
    from seisbench_datasets.data import bohemia
    from seisbench_datasets.data.bohemia import BohemiaSaxony

    if not _prepare(
        BohemiaSaxony,
        bohemia,
        force,
        check,
        recheck_noise,
        noise_fraction,
        export,
        report,
    ):
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
    recheck_noise: RecheckNoise = False,
    export: Export = None,
    report: Report = None,
) -> None:
    """
    DeepLowFrequency: deep low-frequency events below volcanoes, plus noise.
    """
    from seisbench_datasets.data import dlf as dlf_module
    from seisbench_datasets.data.dlf import DeepLowFrequency

    if not _prepare(
        DeepLowFrequency,
        dlf_module,
        force,
        check,
        recheck_noise,
        noise_fraction,
        export,
        report,
    ):
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
