# Notes for agents

## What this repo is

Dataset compilers for SeisBench: catalog + FDSN / Hi-net waveforms → SeisBench format
(`metadata.csv` + `waveforms.hdf5`) in the SeisBench cache (`~/.seisbench/datasets/`),
so that SeisBench finds them. Downloads that aren't part of a dataset (Hi-net blocks)
go to our own cache, `cache.py` (`~/.cache/seisbench-datasets/`). Split out of
`seisbench` (branch `feature/dlf-dataset`, 167b7bb) because the changes were too
large for SeisBench itself.

- `src/seisbench_datasets/data/multiclient.py`: async FDSN client (aiohttp) and
  `HinetSource`, routing of streams to data centers, resampling, `download_parallel`.
- `src/seisbench_datasets/data/resume.py`: resumable compilation. `ResumableWriter`
  subclasses SeisBench's `WaveformDataWriter`, `compile_if_uncached` holds a lock
  while compiling, `ResumableBenchmarkDataset` compiles before SeisBench's cache
  check. Dataset `_download_dataset` methods must skip
  `writer.resumed_trace_names` and pass `writer.checkpoint` to `download_parallel`.
- `src/seisbench_datasets/data/bohemia.py`, `dlf.py`: the datasets.
- `scripts/bohemia_swarms.py` writes `src/seisbench_datasets/data/bohemia_swarms.csv`, the
  swarm windows of the Bohemia catalog. Edit the parameters there, not the CSV by hand.
- `src/seisbench_datasets/cli.py`: typer CLI `seisbench-datasets`, one command per dataset.
- `src/seisbench_datasets/progress.py`: rich logging and the live download table.
  `download_parallel` drives it; the table only shows after `setup_logging()`, so logs
  don't tear through it. SeisBench's tqdm bar is disabled in `ResumableWriter`.

## Working in this repo

- `uv sync`, then `uv run pytest`. Hooks: `prek install` once, `prek run --all-files`
  to check everything (ruff lint + format, uv lock, basic file checks).
- Depend on released SeisBench (`seisbench>=0.12.7`), and don't change SeisBench
  for this repo. `resume.py` overrides private `WaveformDataWriter` methods
  (`_write_bucket`, `_finalize`) and attributes (`_metadata`, `_waveform_file`,
  `_bucket_counter`); check them when raising the SeisBench version.
- Tests must not touch the network. Mock the client, see `tests/data/test_dlf.py`.
- Hi-net data may not be redistributed, WEBNET (WB) data are restricted: never
  commit or upload compiled data.
- Dataset class names set the folder in the SeisBench cache
  (`~/.seisbench/datasets/<classname lower>`). Tests point both caches to a temp
  folder (`tests/conftest.py`).
- Planned features are in `ROADMAP.md`.
