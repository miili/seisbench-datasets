import os
from pathlib import Path

import platformdirs


def cache_root() -> Path:
    """
    Cache of downloads that are not part of a dataset, e.g., the Hi-net blocks.
    ``$SEISBENCH_DATASETS_CACHE``, or ``seisbench-datasets`` in the user cache
    directory, e.g., ``~/.cache/seisbench-datasets``. The compiled datasets live in
    the SeisBench cache, ``seisbench.cache_data_root``.
    """
    root = os.environ.get("SEISBENCH_DATASETS_CACHE")
    if root:
        return Path(root).expanduser()
    return platformdirs.user_cache_path("seisbench-datasets")
