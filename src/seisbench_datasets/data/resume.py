"""
Resumable dataset compilation on top of SeisBench, without changes to SeisBench.

Compiling a dataset from FDSN web services takes hours to days. These helpers let
an interrupted compilation continue from its partial files:

* :py:class:`ResumableWriter` keeps the traces of earlier runs and writes
  checkpoints.
* :py:func:`compile_if_uncached` runs the compilation under a lock, so that a
  second process does not write into the same files, and resumes the partial
  files of a process that stopped.
* :py:class:`ResumableBenchmarkDataset` wires both into a SeisBench
  :py:class:`~seisbench.data.base.WaveformBenchmarkDataset`.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import IO

import h5py
import pandas as pd
from seisbench import logger
from seisbench.data.base import WaveformBenchmarkDataset, WaveformDataWriter
from tqdm import tqdm


class ResumableWriter(WaveformDataWriter):
    """
    A :py:class:`~seisbench.data.base.WaveformDataWriter` that can continue
    existing files.

    :param resume: If true, existing files are continued: their traces are kept,
        see :py:attr:`resumed_metadata`, and new traces are appended. Use
        :py:meth:`checkpoint` to make progress resumable.
    """

    def __init__(self, metadata_path, waveforms_path, resume: bool = False):
        super().__init__(metadata_path, waveforms_path)
        # Progress is shown by download_parallel, hide SeisBench's tqdm bar
        self._pbar = tqdm(disable=True)
        #: Metadata of the traces in the files when resuming, as text
        self.resumed_metadata = pd.DataFrame()
        #: Original names of the traces in the files when resuming
        self.resumed_trace_names: set[str] = set()
        if resume:
            self._resume()

    def _resume(self) -> None:
        if self.metadata_path.is_file() and self.metadata_path.stat().st_size > 0:
            # Read as text, so that values such as location code "00" are written
            # back unchanged
            self.resumed_metadata = pd.read_csv(
                self.metadata_path, dtype=str, keep_default_na=False
            )
            self._metadata = self.resumed_metadata.to_dict("records")
            names = self.resumed_metadata["trace_name"]
            if "trace_name_original" in self.resumed_metadata:
                # Traces in buckets keep their name in trace_name_original
                original = self.resumed_metadata["trace_name_original"]
                names = original.where(original != "", names)
            self.resumed_trace_names = set(names)
        if self.waveforms_path.is_file() and self.waveforms_path.stat().st_size > 0:
            self._waveform_file = h5py.File(self.waveforms_path, "a")
            buckets = [
                int(name[6:])
                for name in self._waveform_file["data"]
                if name.startswith("bucket") and name[6:].isdigit()
            ]
            self._bucket_counter = max(buckets, default=-1) + 1

    def _write_bucket(self, bucket):
        if len(bucket) == 1 and self._waveform_file is not None:
            metadata = bucket[0][0]
            if "trace_name" in metadata:
                trace_name = str(metadata["trace_name"]).replace("$", "_")
                if trace_name in self._waveform_file["data"]:
                    # Written after the last checkpoint of an interrupted run
                    del self._waveform_file["data"][trace_name]
        super()._write_bucket(bucket)

    def _finalize(self):
        self.flush_hdf5()

        if len(self._metadata) == 0:
            return

        if len(self.data_format) > 0:
            # Resumed from a partial file that was finalized after an error
            if "data_format" in self._waveform_file:
                del self._waveform_file["data_format"]
            g_data_format = self._waveform_file.create_group("data_format")
            for key in self.data_format.keys():
                g_data_format.create_dataset(key, data=self.data_format[key])
        else:
            logger.warning("No data format options specified.")

        self._write_metadata()

    def _write_metadata(self) -> None:
        metadata = pd.DataFrame(self._metadata)
        if self.metadata_dict is not None:
            metadata = metadata.rename(columns=self.metadata_dict)

        tmp_path = self.metadata_path.with_name(self.metadata_path.name + ".tmp")
        metadata.to_csv(tmp_path, index=False)
        tmp_path.replace(self.metadata_path)

    def set_default(self, column: str, value) -> None:
        """
        Sets ``column`` of the resumed traces that have no value, e.g., a column
        added after the files were written.
        """
        for row in self._metadata:
            if row.get(column, "") == "":
                row[column] = value

    def checkpoint(self) -> None:
        """
        Writes all traces added so far to disk, waveforms and metadata, so that an
        interrupted compilation can resume from here.
        """
        self.flush_hdf5()
        if self._waveform_file is not None:
            self._waveform_file.flush()
        if self._metadata:
            self._write_metadata()


def try_lock(file: Path) -> IO | None:
    """
    Takes an exclusive lock on ``file.lock``. Returns the open lock file, which
    holds the lock until it is closed, or None if another process holds the lock.
    Without ``fcntl`` (Windows), nothing is locked.
    """
    lock_path = file.parent / (file.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "w")
    try:
        import fcntl
    except ImportError:
        return handle
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def move_to_backup(path: Path) -> Path | None:
    """
    Moves a dataset folder to ``<name>.bak-<time>`` next to it, e.g., before
    compiling the dataset anew.

    :return: The backup folder, or None if the folder doesn't exist
    :raises RuntimeError: If a compilation into the folder is running, i.e.,
        holds the lock of its metadata file, see :py:func:`compile_if_uncached`
    """
    if not path.exists():
        return None
    lock = try_lock(path / "metadata.csv")
    if lock is None:
        raise RuntimeError(f"A download into {path} is running. Stop it first.")
    lock.close()
    backup = path.with_name(f"{path.name}.bak-{datetime.now():%Y%m%dT%H%M%S}")
    path.rename(backup)
    return backup


def compile_if_uncached(
    files: Sequence[Path],
    callback: Callable[[list[Path]], None],
    force: bool = False,
    wait_for_file: bool = False,
    test_interval: float = 60,
) -> None:
    """
    Runs ``callback`` with the ``.partial`` versions of ``files`` if a file is
    missing, and moves the partial files into place afterwards.

    Like :py:func:`seisbench.util.callback_if_uncached`, but the callback holds a
    lock while it runs. Partial files of an earlier callback that no longer holds
    the lock, e.g., after a crash, are kept and given to the callback to continue.

    :param force: Remove partial files instead of resuming from them
    :param wait_for_file: If another process is compiling, wait for it instead of
        raising an error
    :param test_interval: Seconds between checks while waiting
    """
    files = [Path(file) for file in files]
    partial_files = [file.parent / (file.name + ".partial") for file in files]

    def exist():
        return all(file.is_file() for file in files)

    if exist():
        return

    lock = try_lock(files[0])
    while lock is None:
        if not wait_for_file:
            raise ValueError(
                f"Another process is compiling {files[0].parent}. "
                "To wait for it to finish, use wait_for_file=True."
            )
        logger.warning(
            f"Another process is compiling. Rechecking in {test_interval} seconds."
        )
        time.sleep(test_interval)
        if exist():
            return
        lock = try_lock(files[0])

    try:
        if exist():  # Finished by the process we waited for
            lock.close()
            return

        if force:
            for partial_file in partial_files:
                partial_file.unlink(missing_ok=True)
        elif any(file.is_file() for file in partial_files):
            logger.info("Found partial files of an interrupted download. Resuming.")

        for partial_file in partial_files:
            partial_file.parent.mkdir(parents=True, exist_ok=True)
            open(partial_file, "a").close()

        try:
            callback(partial_files)
        finally:
            for partial_file in partial_files:
                if partial_file.stat().st_size == 0:
                    os.remove(partial_file)

        logger.info("Moving partial files to target")
        for partial_file, file in zip(partial_files, files, strict=True):
            if partial_file.is_file():
                partial_file.rename(file)
    except BaseException:
        lock.close()
        raise

    # Released only now, so that no other process resumes a finished compilation
    lock.close()
    try:
        Path(lock.name).unlink(missing_ok=True)
    except PermissionError:
        pass  # Windows: another process still has the lock file open


class ResumableBenchmarkDataset(WaveformBenchmarkDataset):
    """
    A :py:class:`~seisbench.data.base.WaveformBenchmarkDataset` whose compilation
    from source resumes after an interruption.

    The compilation runs before SeisBench's own cache check, which then finds the
    finished files. ``_download_dataset`` receives a :py:class:`ResumableWriter`
    and must skip the traces in ``writer.resumed_trace_names``. Only unchunked
    datasets are supported. The datasets live in the SeisBench cache, so that
    SeisBench finds them.
    """

    def __init__(
        self,
        force: bool = False,
        wait_for_file: bool = False,
        compile_from_source: bool = True,
        download_kwargs: dict | None = None,
        **kwargs,
    ):
        download_kwargs = download_kwargs or {}
        if compile_from_source:
            files = self.files()
            compile_if_uncached(
                files,
                lambda partial_files: self._compile(partial_files, **download_kwargs),
                force=force,
                wait_for_file=wait_for_file,
            )
            if not all(file.is_file() for file in files):
                raise RuntimeError(
                    f"Compiling {type(self).__name__} wrote no traces, see the log."
                )

        # SeisBench must not compile itself, its writer can't resume
        super().__init__(
            wait_for_file=wait_for_file,
            compile_from_source=False,
            download_kwargs=download_kwargs,
            **kwargs,
        )

    @classmethod
    def planned_trace_names(cls) -> set[str] | None:
        """
        The names of the event traces the dataset should contain, to find missing
        traces, or None if the dataset can't plan them without downloading.
        """
        return None

    @classmethod
    def files(cls) -> list[Path]:
        return [
            cls._path_internal() / file.replace("$CHUNK", "") for file in cls._files
        ]

    @classmethod
    def reopen(cls) -> None:
        """
        Turns the files of a compiled dataset back into partial files, so that
        creating the dataset resumes from them: it downloads the traces that are
        missing, e.g., to fill up a dataset with traces that failed or with noise.
        Until the compilation finishes, SeisBench can't load the dataset.

        :raises RuntimeError: If a compilation into the folder is running
        """
        files = cls.files()
        lock = try_lock(files[0])
        if lock is None:
            raise RuntimeError(f"A download into {files[0].parent} is running.")
        try:
            for file in files:
                if file.is_file():
                    file.rename(file.parent / (file.name + ".partial"))
        finally:
            lock.close()

    def _compile(self, files: list[Path], **kwargs) -> None:
        logger.warning(
            f"Dataset {type(self).__name__} not in cache. Compiling from source."
        )
        download_args = self.add_chunk_to_download_args("", kwargs)
        with ResumableWriter(*files, resume=True) as writer:
            self._download_dataset(writer, **download_args)
