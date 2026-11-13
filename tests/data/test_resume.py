import sys

import numpy as np
import pandas as pd
import pytest
import seisbench.data as sbd

from seisbench_datasets.data.resume import (
    NoDataLog,
    ResumableBenchmarkDataset,
    ResumableWriter,
    compile_in_place,
    move_to_backup,
    try_lock,
)

LOCKING = sys.platform != "win32"  # Without fcntl, nothing is locked


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


def test_compile_in_place(tmp_path):
    file = tmp_path / "metadata.csv"
    marker = tmp_path / "metadata.csv.incomplete"
    calls = []

    def fail(paths):
        paths[0].write_text("progress")
        raise RuntimeError("download failed")

    def finish(paths):
        calls.append(paths[0].read_text())
        paths[0].write_text(paths[0].read_text() + " done")

    # Interrupted: the files stay in place, marked as incomplete
    with pytest.raises(RuntimeError):
        compile_in_place([file], fail)
    assert file.read_text() == "progress"
    assert marker.exists()

    if LOCKING:
        # Another process holds the lock: still running
        lock = try_lock(file)
        with pytest.raises(ValueError, match="Another process"):
            compile_in_place([file], finish)
        lock.close()

    # Continued from what is there
    compile_in_place([file], finish)
    assert calls == ["progress"]
    assert file.read_text() == "progress done"
    assert not marker.exists()
    assert not (tmp_path / "metadata.csv.lock").exists()

    # Complete: not compiled again, unless updated
    compile_in_place([file], finish)
    assert calls == ["progress"]
    compile_in_place([file], finish, update=True)
    assert calls == ["progress", "progress done"]
    assert not marker.exists()


def test_compile_force(tmp_path):
    file = tmp_path / "metadata.csv"
    file.write_text("progress")
    (tmp_path / "metadata.csv.incomplete").touch()
    (tmp_path / "no_data.csv").write_text("trace_name,reason\na,no data\n")

    def callback(paths):
        assert not paths[0].exists()
        paths[0].write_text("fresh")

    # An incomplete dataset is compiled anew, also the traces without data
    compile_in_place([file], callback, force=True)
    assert file.read_text() == "fresh"
    assert not (tmp_path / "no_data.csv").exists()

    # A complete dataset is kept
    compile_in_place([file], callback, force=True)
    assert file.read_text() == "fresh"


def test_no_data_log(tmp_path):
    path = tmp_path / "no_data.csv"
    log = NoDataLog(path)
    assert "a" not in log
    log.add("a", "no data")
    log.add("a", "again")  # Recorded once
    log.add("b", ValueError("no data either"))
    assert "a" in log and len(log) == 2

    again = NoDataLog(path)
    assert again.names == {"a", "b"}
    assert len(pd.read_csv(path)) == 2


def test_move_to_backup(tmp_path):
    dataset = tmp_path / "dataset"
    assert move_to_backup(dataset) is None

    dataset.mkdir()
    (dataset / "metadata.csv").write_text("progress")
    if LOCKING:
        # A running download holds the lock: its folder must not be moved
        lock = try_lock(dataset / "metadata.csv")
        with pytest.raises(RuntimeError, match="running"):
            move_to_backup(dataset)
        lock.close()
        assert dataset.is_dir()

    backup = move_to_backup(dataset)
    assert not dataset.exists()
    assert (backup / "metadata.csv").read_text() == "progress"


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
    # The checkpointed traces are in place, marked as incomplete
    assert (path / "metadata.csv").exists()
    assert not _ToyDataset.is_complete()

    _ToyDataset.fail_at = None
    dataset = _ToyDataset()
    assert _ToyDataset.added == ["a", "b", "c", "d"]
    assert len(dataset) == 4
    assert _ToyDataset.is_complete()
    assert not (path / "metadata.csv.lock").exists()


class _GrowingDataset(ResumableBenchmarkDataset):
    names = "ab"
    #: The data centers have no data for these
    unavailable = set()
    tried = []

    def __init__(self, **kwargs):
        super().__init__(repository_lookup=False, **kwargs)

    def _download_dataset(self, writer, **kwargs):
        writer.data_format = {"dimension_order": "CW", "component_order": "ZNE"}
        writer.set_default("trace_category", "earthquake")
        for name in self.names:
            if name in writer.resumed_trace_names or name in writer.no_data:
                continue
            self.tried.append(name)
            if name in self.unavailable:
                writer.no_data.add(name, "no data")
                continue
            writer.add_trace(
                {"trace_name": name, "split": "train", "trace_category": "earthquake"},
                np.zeros((3, 9)),
            )


def test_update_fills_up():
    _GrowingDataset.unavailable = {"c"}
    _GrowingDataset()
    assert _GrowingDataset.tried == ["a", "b"]

    # A complete dataset is loaded, not compiled again
    _GrowingDataset.names = "abcd"
    assert len(_GrowingDataset()) == 2
    assert _GrowingDataset.tried == ["a", "b"]

    # Updated: only the missing traces are downloaded
    dataset = _GrowingDataset(update=True)
    assert _GrowingDataset.tried == ["a", "b", "c", "d"]
    assert len(dataset) == 3
    # Traces written before the column existed get the default
    assert set(dataset.metadata["trace_category"]) == {"earthquake"}

    # The trace without data is skipped from now on
    _GrowingDataset(update=True)
    assert _GrowingDataset.tried == ["a", "b", "c", "d"]


@pytest.mark.skipif(not LOCKING, reason="Without fcntl, nothing is locked")
def test_update_running():
    _GrowingDataset.names = "a"
    _GrowingDataset()
    lock = try_lock(_GrowingDataset.files()[0])
    with pytest.raises(ValueError, match="Another process"):
        _GrowingDataset(update=True)
    lock.close()
    assert _GrowingDataset.is_complete()
