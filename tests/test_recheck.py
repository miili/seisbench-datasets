import sys

import h5py
import numpy as np
import pandas as pd
import pytest
import seisbench.data as sbd

from seisbench_datasets.data.resume import ResumableWriter, try_lock
from seisbench_datasets.recheck import read_waveforms, recheck_noise


class _Detector:
    """An "earthquake" is a window with values of 5 or more."""

    weights = "stub"

    def __init__(self):
        self.calls = 0

    def max_probability(self, data, start=0):
        self.calls += 1
        return 0.9 if data.max() >= 5 else 0.1


def _write(path, bucket_size):
    files = [path / "metadata.csv", path / "waveforms.hdf5"]
    with ResumableWriter(*files) as writer:
        writer.bucket_size = bucket_size
        writer.data_format = {"dimension_order": "CW", "component_order": "ZNE"}
        for i in range(4):
            writer.add_trace(
                {
                    "trace_name": f"e{i}",
                    "split": "train",
                    "trace_category": "earthquake",
                },
                np.full((3, 100), 9.0),  # Events are not checked
            )
        for i in range(6):
            writer.add_trace(
                {
                    "trace_name": f"noise_e{i}",
                    "split": "dev" if i % 2 else "train",
                    "trace_category": "noise",
                },
                np.full((3, 100 + i), float(i)),
            )
    return files


@pytest.mark.parametrize("bucket_size", [1024, 1])
def test_recheck_noise(tmp_path, bucket_size):
    files = _write(tmp_path, bucket_size)
    detector = _Detector()
    result = recheck_noise(files, detector, threshold=0.5)

    assert result.checked == 6 and detector.calls == 6
    assert [(r.trace_name, r.split, r.reason) for r in result.removed] == [
        ("noise_e5", "dev", "earthquake")
    ]
    assert len(pd.read_csv(result.backup)) == 10

    metadata = pd.read_csv(files[0], dtype=str, keep_default_na=False)
    names = (
        metadata["trace_name_original"] if bucket_size > 1 else metadata["trace_name"]
    )
    assert "noise_e5" not in set(names) and len(metadata) == 9
    rejected = pd.read_csv(tmp_path / "noise_rejected.csv")
    assert list(rejected["trace_name"]) == ["noise_e5"]
    assert list(rejected["weights"]) == ["stub"]

    # The remaining traces load with their waveforms
    dataset = sbd.WaveformDataset(tmp_path, component_order="ZNE")
    assert len(dataset) == 9
    for i in range(len(dataset)):
        assert dataset.get_waveforms(i).shape[0] == 3

    # Nothing left to remove
    again = recheck_noise(files, _Detector(), threshold=0.5)
    assert again.checked == 5 and again.removed == [] and again.backup is None


def test_read_waveforms(tmp_path):
    files = _write(tmp_path, bucket_size=1024)
    metadata = pd.read_csv(files[0], dtype=str, keep_default_na=False)
    with h5py.File(files[1], "r") as file:
        for name, original in zip(
            metadata["trace_name"], metadata["trace_name_original"], strict=True
        ):
            data = read_waveforms(file, name)
            if original.startswith("noise_e"):
                i = int(original[-1])
                assert data.shape == (3, 100 + i)  # Without the bucket padding
                assert (data == i).all()


@pytest.mark.skipif(sys.platform == "win32", reason="Without fcntl, nothing is locked")
def test_recheck_running(tmp_path):
    files = _write(tmp_path, bucket_size=1024)
    lock = try_lock(files[0])
    with pytest.raises(RuntimeError, match="running"):
        recheck_noise(files, _Detector(), threshold=0.5)
    lock.close()


def test_recheck_trims_above_fraction(tmp_path):
    # 4 event traces in train and 0 in dev, 3 noise traces in each
    files = _write(tmp_path, bucket_size=1024)
    result = recheck_noise(files, _Detector(), 0.5, noise_fraction=0.5, group="split")

    reasons = [(r.split, r.reason) for r in result.removed]
    # dev: noise_e5 has an earthquake, the other 2 are above 0.5 of 0 event traces;
    # train: 3 noise traces, 2 allowed
    assert reasons.count(("dev", "earthquake")) == 1
    assert reasons.count(("dev", "above noise fraction")) == 2
    assert reasons.count(("train", "above noise fraction")) == 1

    metadata = pd.read_csv(files[0], dtype=str, keep_default_na=False)
    noise = metadata[metadata["trace_category"] == "noise"]
    assert list(noise["split"]) == ["train", "train"]
    rejected = pd.read_csv(tmp_path / "noise_rejected.csv", keep_default_na=False)
    assert set(rejected["reason"]) == {"earthquake", "above noise fraction"}
    assert (rejected["probability"] == "").sum() == 3

    # Seeded, and nothing more to remove
    again = recheck_noise(files, _Detector(), 0.5, noise_fraction=0.5, group="split")
    assert again.removed == []
