import pytest


@pytest.fixture(autouse=True)
def cache_root(tmp_path, monkeypatch):
    # Never touch the real cache
    root = tmp_path / "cache"
    monkeypatch.setenv("SEISBENCH_DATASETS_CACHE", str(root))
    return root
