import os
from pathlib import Path

import platformdirs


def cache_root() -> Path:
    """
    Cache of seisbench-datasets: the compiled datasets and the Hi-net downloads.
    ``$SEISBENCH_DATASETS_CACHE``, or ``seisbench-datasets`` in the user cache
    directory, e.g., ``~/.cache/seisbench-datasets``.
    """
    root = os.environ.get("SEISBENCH_DATASETS_CACHE")
    if root:
        return Path(root).expanduser()
    return platformdirs.user_cache_path("seisbench-datasets")
