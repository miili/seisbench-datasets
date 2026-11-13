import pytest
import seisbench


@pytest.fixture(autouse=True)
def cache_root(tmp_path, monkeypatch):
    # Never touch the real caches, neither ours nor the SeisBench datasets
    root = tmp_path / "cache"
    monkeypatch.setenv("SEISBENCH_DATASETS_CACHE", str(root))
    monkeypatch.setattr(seisbench, "cache_data_root", tmp_path / "datasets")
    return root
