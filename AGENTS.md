# Notes for agents

## What this repo is

Dataset compilers for SeisBench: catalog + FDSN / Hi-net waveforms → SeisBench format
(`metadata.csv` + `waveforms.hdf5`) in the SeisBench cache. Split out of
`seisbench` (branch `feature/dlf-dataset`, 167b7bb) because the changes were too
large for SeisBench itself.

- `src/seisbench_models/data/multiclient.py`: async FDSN client (aiohttp) and
  `HinetSource`, routing of streams to data centers, resampling, `download_parallel`.
- `src/seisbench_models/data/resume.py`: resumable compilation. `ResumableWriter`
  subclasses SeisBench's `WaveformDataWriter`, `compile_if_uncached` holds a lock
  while compiling, `ResumableBenchmarkDataset` compiles before SeisBench's cache
  check. Dataset `_download_dataset` methods must skip
  `writer.resumed_trace_names` and pass `writer.checkpoint` to `download_parallel`.
- `src/seisbench_models/data/bohemia.py`, `dlf.py`: the datasets.

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
- Dataset class names set the cache folder (`<cache>/datasets/<classname lower>`).
  `BohemiaSaxony` shares its folder with `seisbench.data.BohemiaSaxony`.
- Planned features are in `ROADMAP.md`.
