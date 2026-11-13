import h5py
import numpy as np

from seisbench_datasets.check import check_dataset
from seisbench_datasets.data.resume import ResumableWriter


def _write(path, bucket_size):
    with ResumableWriter(path / "metadata.csv", path / "waveforms.hdf5") as writer:
        writer.bucket_size = bucket_size
        writer.data_format = {"dimension_order": "CW", "component_order": "ZNE"}
        for i in range(10):
            writer.add_trace(
                {
                    "trace_name": f"e{i}",
                    "split": "train" if i < 8 else "test",
                    "trace_category": "earthquake",
                },
                np.zeros((3, 100)),
            )
        for i in range(3):
            writer.add_trace(
                {
                    "trace_name": f"noise_e{i}",
                    "split": "train",
                    "trace_category": "noise",
                },
                np.zeros((3, 100)),
            )


def test_check_dataset(tmp_path):
    _write(tmp_path, bucket_size=4)
    planned = {f"e{i}" for i in range(9)} | {"e42"}
    report = check_dataset(tmp_path, 0.3, planned)

    assert report.ok
    assert report.traces == 13
    assert [tuple(s[:3]) for s in report.splits] == [("test", 2, 0), ("train", 8, 3)]
    assert report.splits[1].noise_fraction == 3 / 8
    assert report.missing == ["e42"]
    assert report.extra == ["e9"]

    assert check_dataset(tmp_path, 0.3).missing is None


def test_check_finds_missing_waveforms(tmp_path):
    _write(tmp_path, bucket_size=1)  # One dataset per trace
    with h5py.File(tmp_path / "waveforms.hdf5", "a") as file:
        del file["data"]["e3"]
    report = check_dataset(tmp_path, 0.3)
    assert not report.ok
    assert report.unreadable == ["e3"]
