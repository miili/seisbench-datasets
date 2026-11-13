import sys

import numpy as np
import pandas as pd
import pytest
import seisbench.data as sbd

from seisbench_datasets.data.resume import (
    ResumableBenchmarkDataset,
    ResumableWriter,
    compile_if_uncached,
    move_to_backup,
    try_lock,
)


def _add(writer, names):
    for i, name in enumerate(names):
        writer.add_trace(
            {"trace_name": name, "station_location_code": "00", "split": "train"},
            np.full((3, 100 + i), float(i)),
        )


def _writer(path, resume=False):
    writer = ResumableWriter(
        path / "metadata.csv", path / "waveforms.hdf5", resume=resume
    )
    writer.data_format = {"dimension_order": "CW", "component_order": "ZNE"}
    return writer


def _kill(writer):
    # A hard kill: no finalize, the files are just left behind
    writer._waveform_file.close()


@pytest.mark.parametrize("bucket_size", [1024, 1])
def test_writer_resume(tmp_path, bucket_size):
    writer = _writer(tmp_path)
    writer.bucket_size = bucket_size
    _add(writer, ["a", "b", "c"])
    writer.checkpoint()
    _add(writer, ["d"])  # Not checkpointed: lost
    _kill(writer)

    with _writer(tmp_path, resume=True) as writer:
        writer.bucket_size = bucket_size
        assert writer.resumed_trace_names == {"a", "b", "c"}
        _add(writer, ["d", "e"])

    metadata = pd.read_csv(tmp_path / "metadata.csv", dtype=str, keep_default_na=False)
    names = metadata["trace_name"]
    if "trace_name_original" in metadata:
        names = metadata["trace_name_original"].where(
            metadata["trace_name_original"] != "", names
        )
    assert sorted(names) == ["a", "b", "c", "d", "e"]
    # Values are written back unchanged, e.g., location codes are not read as numbers
    assert set(metadata["station_location_code"]) == {"00"}

    dataset = sbd.WaveformDataset(tmp_path)
    assert len(dataset) == 5
    for i in range(len(dataset)):
        assert dataset.get_waveforms(i).shape[0] == 3


def test_writer_resume_after_error(tmp_path):
    # An exception finalizes the files, which can then be resumed, too
    with pytest.raises(RuntimeError), _writer(tmp_path) as writer:
        _add(writer, ["a", "b"])
        raise RuntimeError("download failed")

    with _writer(tmp_path, resume=True) as writer:
        assert writer.resumed_trace_names == {"a", "b"}
        _add(writer, ["c"])

    assert len(sbd.WaveformDataset(tmp_path)) == 3


def test_compile_resume(tmp_path):
    file = tmp_path / "metadata.csv"
    partial = tmp_path / "metadata.csv.partial"
    partial.write_text("progress")
    calls = []

    def callback(paths):
        (path,) = paths
        calls.append(path.read_text())
        path.write_text(path.read_text() + " done")

    if sys.platform != "win32":  # Without fcntl, nothing is locked
        # Another process holds the lock: still running
        lock = try_lock(file)
        with pytest.raises(ValueError, match="Another process"):
            compile_if_uncached([file], callback)
        lock.close()

    # The lock is free: the earlier download stopped, so resume from its files
    compile_if_uncached([file], callback)
    assert calls == ["progress"]
    assert file.read_text() == "progress done"
    assert not (tmp_path / "metadata.csv.lock").exists()

    # Finished: the callback doesn't run again
    compile_if_uncached([file], callback)
    assert calls == ["progress"]


def test_compile_force(tmp_path):
    file = tmp_path / "metadata.csv"
    (tmp_path / "metadata.csv.partial").write_text("progress")

    def callback(paths):
        (path,) = paths
        assert path.read_text() == ""
        path.write_text("fresh")

    compile_if_uncached([file], callback, force=True)
    assert file.read_text() == "fresh"


def test_compile_error_keeps_partial(tmp_path):
    file = tmp_path / "metadata.csv"

    def callback(paths):
        paths[0].write_text("progress")
        raise RuntimeError("download failed")

    with pytest.raises(RuntimeError):
        compile_if_uncached([file], callback)
    assert not file.exists()
    assert (tmp_path / "metadata.csv.partial").read_text() == "progress"


def test_move_to_backup(tmp_path):
    dataset = tmp_path / "dataset"
    assert move_to_backup(dataset) is None

    dataset.mkdir()
    (dataset / "metadata.csv.partial").write_text("progress")
    if sys.platform != "win32":  # Without fcntl, nothing is locked
        # A running download holds the lock: its folder must not be moved
        lock = try_lock(dataset / "metadata.csv")
        with pytest.raises(RuntimeError, match="running"):
            move_to_backup(dataset)
        lock.close()
        assert dataset.is_dir()

    backup = move_to_backup(dataset)
    assert not dataset.exists()
    assert (backup / "metadata.csv.partial").read_text() == "progress"


class _ToyDataset(ResumableBenchmarkDataset):
    fail_at = None
    added = []

    def __init__(self, **kwargs):
        super().__init__(repository_lookup=False, **kwargs)

    def _download_dataset(self, writer, **kwargs):
        writer.data_format = {"dimension_order": "CW", "component_order": "ZNE"}
        for name in "abcd":
            if name in writer.resumed_trace_names:
                continue
            if name == self.fail_at:
                raise RuntimeError("download failed")
            writer.add_trace({"trace_name": name, "split": "train"}, np.zeros((3, 100)))
            self.added.append(name)
            writer.checkpoint()


def test_dataset_resume(tmp_path):
    path = _ToyDataset._path_internal()
    assert path == tmp_path / "datasets" / "_toydataset"

    _ToyDataset.fail_at = "c"
    with pytest.raises(RuntimeError):
        _ToyDataset()
    assert not (path / "metadata.csv").exists()
    assert (path / "metadata.csv.partial").exists()

    _ToyDataset.fail_at = None
    dataset = _ToyDataset()
    assert _ToyDataset.added == ["a", "b", "c", "d"]
    assert len(dataset) == 4
    assert not (path / "metadata.csv.lock").exists()


class _GrowingDataset(ResumableBenchmarkDataset):
    names = "ab"
    added = []

    def __init__(self, **kwargs):
        super().__init__(repository_lookup=False, **kwargs)

    def _download_dataset(self, writer, **kwargs):
        writer.data_format = {"dimension_order": "CW", "component_order": "ZNE"}
        writer.set_default("trace_category", "earthquake")
        for name in self.names:
            if name not in writer.resumed_trace_names:
                writer.add_trace(
                    {
                        "trace_name": name,
                        "split": "train",
                        "trace_category": "earthquake",
                    },
                    np.zeros((3, 9)),
                )
                self.added.append(name)


def test_reopen_fills_up():
    _GrowingDataset()
    assert _GrowingDataset.added == ["a", "b"]

    # A finished dataset is not compiled again
    _GrowingDataset.names = "abcd"
    assert len(_GrowingDataset()) == 2

    _GrowingDataset.reopen()
    assert not any(file.is_file() for file in _GrowingDataset.files())
    dataset = _GrowingDataset()
    assert _GrowingDataset.added == ["a", "b", "c", "d"]
    assert len(dataset) == 4
    # Traces written before the column existed get the default
    assert set(dataset.metadata["trace_category"]) == {"earthquake"}


def test_reopen_running(tmp_path):
    _GrowingDataset.names = "a"
    _GrowingDataset()
    if sys.platform != "win32":  # Without fcntl, nothing is locked
        lock = try_lock(_GrowingDataset.files()[0])
        with pytest.raises(RuntimeError, match="running"):
            _GrowingDataset.reopen()
        lock.close()
        assert all(file.is_file() for file in _GrowingDataset.files())
