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
    "waveforms. An interrupted compilation resumes when it is started again.",
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


@app.callback()
def main() -> None:
    setup_logging()


def _move_to_backup(dataset: type) -> None:
    from seisbench import logger

    from seisbench_datasets.data.resume import move_to_backup

    try:
        backup = move_to_backup(dataset._path_internal())
    except RuntimeError as exc:
        console.print(f"[red]{exc}")
        raise typer.Exit(1) from exc
    if backup is not None:
        logger.info("Moved the existing dataset to %s", backup)


@app.command()
def bohemia(eida_token: EidaToken = None, force: Force = False) -> None:
    """
    BohemiaSaxony: local earthquakes in NW Bohemia / Vogtland and Saxony.
    """
    from seisbench_datasets.data.bohemia import BohemiaSaxony

    if force:
        _move_to_backup(BohemiaSaxony)
    dataset = BohemiaSaxony(
        eida_token=str(eida_token) if eida_token else None, component_order="ZNE"
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
    force: Force = False,
) -> None:
    """
    DeepLowFrequency: deep low-frequency events below volcanoes, plus noise.
    """
    from seisbench_datasets.data.dlf import DeepLowFrequency

    hinet_password = None
    if hinet_user is not None:
        hinet_password = os.environ.get("HINET_PASSWORD") or typer.prompt(
            f"Hi-net password for {hinet_user}", hide_input=True
        )
    if force:
        _move_to_backup(DeepLowFrequency)
    dataset = DeepLowFrequency(
        catalog_dir=catalog_dir,
        eida_token=str(eida_token) if eida_token else None,
        hinet_user=hinet_user,
        hinet_password=hinet_password,
        component_order="ZNE",
    )
    console.print(f"{dataset}\nin {dataset.path}")
