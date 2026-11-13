"""
Terminal output: logs through rich, and a live progress table of the downloads.

The live table is only shown after :py:func:`setup_logging`, so that the logs run
above it instead of through it.
"""

from __future__ import annotations

import logging
from collections.abc import Hashable

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from seisbench import logger

console = Console(stderr=True)


def setup_logging(level: int | str = logging.INFO) -> None:
    """
    Sends the SeisBench logs, which include the logs of this package, through rich,
    and enables the live progress table.
    """
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    logger.addHandler(RichHandler(console=console, show_path=False))
    logger.setLevel(level)


def live_enabled() -> bool:
    return console.is_terminal and any(
        isinstance(handler, RichHandler) and handler.console is console
        for handler in logger.handlers
    )


class DownloadProgress:
    """
    Live table of the downloads, one row per source and a total row.

    :param jobs_per_source: Number of jobs per source. Rows are labeled ``str(source)``.
    :param title: Label of the total row
    """

    def __init__(self, jobs_per_source: dict[Hashable, int], title: str = "Total"):
        self.live = live_enabled()
        self._progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[red]{task.fields[failed]} failed"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            disable=not self.live,
        )
        self._tasks: dict[Hashable, TaskID] = {
            source: self._progress.add_task(str(source), total=n, failed=0)
            for source, n in jobs_per_source.items()
        }
        self._failed = dict.fromkeys(jobs_per_source, 0)
        self._total = self._progress.add_task(
            f"[bold]{title}", total=sum(jobs_per_source.values()), failed=0
        )

    def __enter__(self) -> DownloadProgress:
        self._progress.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self._progress.stop()

    def advance(self, source: Hashable, failed: bool) -> None:
        self._failed[source] += failed
        self._progress.update(
            self._tasks[source], advance=1, failed=self._failed[source]
        )
        self._progress.update(self._total, advance=1, failed=sum(self._failed.values()))
