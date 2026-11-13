# seisbench-datasets

Compile [SeisBench](https://github.com/seisbench/seisbench) training datasets from
earthquake catalogs and waveforms from FDSN web services and NIED Hi-net.

The datasets are ordinary SeisBench `WaveformBenchmarkDataset`s. They are compiled
on first use into the SeisBench cache (`~/.seisbench/datasets/`), where SeisBench
finds them, and an interrupted download resumes where it stopped.

| Dataset | Content | Sources |
| --- | --- | --- |
| `BohemiaSaxony` | Local earthquakes in NW Bohemia / Vogtland and Saxony, 2006–2024, manual P and S picks, plus noise from quiet periods | BGR, GFZ (SX, WB, TH, CZ) |
| `DeepLowFrequency` | Deep low-frequency (DLF / DLP) events below volcanoes in Alaska, Japan, Hawaii, N. California, the Cascades and the Eifel, plus noise windows | EarthScope, NCEDC, BGR, GFZ, NIED Hi-net |

## Install

```sh
uv sync                    # or: pip install .
uv sync --extra hinet      # Japan region of DeepLowFrequency
```

The Hi-net download also needs a Hi-net account and the Hi-net win32tools
(`catwin32` and `win2sac_32` on the `PATH`).

## Compile

```sh
seisbench-datasets bohemia --eida-token ~/.eidatoken
seisbench-datasets dlf --catalog-dir quakeml/ [--hinet-user USER]
```

The dataset is compiled in place, and a command always continues it from what is
there: it downloads the planned traces that are missing and tops up the noise to
`--noise-fraction` in each split. So the same command compiles a new dataset,
continues an interrupted one (Ctrl-C is fine), and fills up an older one. Traces the
data centers have no data for (HTTP 204 / 404) are recorded in `no_data.csv` in the
dataset folder and skipped; delete the file to try them again. A live table shows
the downloads per data center (done, failed, time left), the logs run above it.
`--force` moves an existing dataset to a backup folder and compiles it anew.

```sh
seisbench-datasets bohemia --check            # or --validate; downloads nothing
seisbench-datasets bohemia --recheck-noise    # PhaseNet over the existing noise
seisbench-datasets bohemia --export <dir>     # MiniSEED, picks, events for pyrocko
```

`--check` reports the event and noise traces per split against `--noise-fraction`,
traces without waveforms in the HDF5 file, and, for BohemiaSaxony, the event traces
the catalog plans that are missing, with and without data at the data centers (it
reads the whole catalog, about a minute). `--recheck-noise` runs the earthquake
detection over the noise traces already in the dataset, e.g., noise compiled
before it, and removes the noise traces with an earthquake and those above the noise
fraction from the metadata; the next run tops the noise up again. The metadata
before is kept as `metadata.csv.bak-<time>`, the removed traces are listed in
`noise_rejected.csv`. `--export` needs pyrocko: `uv sync --extra export`.

## Use

```python
from seisbench_datasets.data import BohemiaSaxony, DeepLowFrequency

data = BohemiaSaxony(component_order="ZNE")
dlf = DeepLowFrequency(catalog_dir="quakeml/")  # one QuakeML file per region
```

Compiling from Python works the same. Call `seisbench_datasets.progress.setup_logging()`
first to get the live table there, too.

### Notes

- **BohemiaSaxony**: the WEBNET (WB) data are restricted and need an
  [EIDA token](https://doi.org/10.7914/SN/WB). No precompiled version is distributed.
  Noise traces (`trace_category = "noise"`, 30% of the event traces by default) are
  taken at random times in quiet periods, not selected by STA/LTA: outside the swarm
  windows in `src/seisbench_datasets/data/bohemia_swarms.csv` (from
  `scripts/bohemia_swarms.py`), away from catalog events, within 30 days of an event
  at the same station. Windows in which PhaseNet (`original` weights) detects an
  earthquake are skipped. DeepLowFrequency rejects its noise windows the same way,
  with the `volpick` weights, which also detect long-period events.
- **DeepLowFrequency**: Hi-net data may not be redistributed. Hi-net downloads are
  cached in blocks of 5 minutes in `~/.cache/seisbench-datasets/hinet/` (or
  `$SEISBENCH_DATASETS_CACHE`), so compiling again with other windows reuses them.
  The compilation writes `download_stats.json` and `download_failures.csv` to the
  dataset folder.

See [ROADMAP.md](ROADMAP.md) for planned features.

## License

GPL-3.0-or-later, as SeisBench, from which the code originates.
