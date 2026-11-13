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
- `src/seisbench_datasets/data/resume.py`: compilation in place. `ResumableWriter`
  subclasses SeisBench's `WaveformDataWriter`; `compile_in_place` holds a lock and
  marks the dataset incomplete (`metadata.csv.incomplete`) while compiling;
  `ResumableBenchmarkDataset` compiles before SeisBench's cache check, continues an
  incomplete dataset, and with `update=True` a complete one. Dataset
  `_download_dataset` methods must skip `writer.resumed_trace_names` and
  `writer.no_data`, record `FDSNNoDataException` (HTTP 204 / 404 only) with
  `writer.no_data.add`, and pass `writer.checkpoint` to `download_parallel`.
- `src/seisbench_datasets/data/bohemia.py`, `dlf.py`: the datasets.
- `scripts/bohemia_swarms.py` writes `src/seisbench_datasets/data/bohemia_swarms.csv`, the
  swarm windows of the Bohemia catalog. Edit the parameters there, not the CSV by hand.
- `src/seisbench_datasets/cli.py`: typer CLI `seisbench-datasets`, one command per dataset.
  A command always continues the dataset (`update=True`); `--check` (`check.py`) is
  the read-only report; `--recheck-noise` (`recheck.py`) removes noise traces with
  an earthquake or above the noise fraction from the metadata (waveforms stay in
  the HDF5); `--export` uses `seisbench.data.inspection` (pyrocko, extra `export`). `--report` (`report.py`, extra `report`) writes an interactive HTML page from
  `metadata.csv` alone; figures are plotly.js specs with colors as CSS custom
  properties (`"--series-1"`) that `report.html` resolves per light / dark theme.
  Splits keep their colors (train, dev, test = series 1–3) in every figure. Maps
  are plotly geo, not tile maps: MapLibre's worker is blocked on `file://` pages. The Python constructor only compiles a missing or
  incomplete dataset, so loading a complete one stays offline.
- Noise is planned by proportion, not 1:1: after the event traces, each split gets
  `round(noise_fraction * event traces) - noise traces` more, drawn at random event
  traces (anchors) with a seeded generator. Which anchors doesn't matter. DLF plans
  its first noise windows per event trace (for combined Hi-net requests), capped by
  `cap_noise_jobs` to the noise fraction of each region, so that a resumed
  compilation doesn't download windows that were replaced before.
- Noise windows are rejected only for earthquakes: `EarthquakeDetector` (`noise.py`) runs
  PhaseNet and rejects P/S probabilities above 0.5, with `volpick` for DLF (also
  detects long-period events) and `original` for Bohemia (local earthquakes). Data are clipped at 20 MAD first: peak-normalized `volpick` otherwise
  calls a single spike a phase. Not STA/LTA, which also
  rejects spikes, traffic and wind and left Hawaii and Japan without noise. A few
  earthquakes in the noise are fine.
- `src/seisbench_datasets/progress.py`: rich logging and the live download table.
  `download_parallel` drives it; the table only shows after `setup_logging()`, so logs
  don't tear through it. SeisBench's tqdm bar is disabled in `ResumableWriter`.

## Working in this repo

- `uv sync`, then `uv run pytest`. Hooks: `prek install` once, `prek run --all-files`
  to check everything (ruff lint + format, uv lock, basic file checks).
- End-to-end tests compile tiny subsets from the real data centers. They are
  deselected by default; run them with `uv sync --extra hinet`, then
  `EIDA_TOKEN=<file> DLF_CATALOG_DIR=<quakeml dir> HINET_USER=<user>
  HINET_PASSWORD=<password> uv run pytest -m e2e tests/e2e` (~2 min). Never write
  credentials into files.
- Depend on released SeisBench (`seisbench>=0.12.7`), and don't change SeisBench
  for this repo. `resume.py` overrides private `WaveformDataWriter` methods
  (`_write_bucket`, `_finalize`) and attributes (`_metadata`, `_waveform_file`,
  `_bucket_counter`); check them when raising the SeisBench version.
- Unit tests must not touch the network. Mock the client, see `tests/data/test_dlf.py`.
- Hi-net data may not be redistributed, WEBNET (WB) data are restricted: never
  commit or upload compiled data.
- Dataset class names set the folder in the SeisBench cache
  (`~/.seisbench/datasets/<classname lower>`). Tests point both caches to a temp
  folder (`tests/conftest.py`).
- Planned features are in `ROADMAP.md`.
