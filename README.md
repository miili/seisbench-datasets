# seisbench-datasets

Compile [SeisBench](https://github.com/seisbench/seisbench) training datasets from
earthquake catalogs and waveforms from FDSN web services and NIED Hi-net.

The datasets are ordinary SeisBench `WaveformBenchmarkDataset`s. They are compiled
on first use into the SeisBench cache (`~/.seisbench/datasets/`), where SeisBench
finds them, and an interrupted download resumes where it stopped.

| Dataset | Content | Sources |
| --- | --- | --- |
| `BohemiaSaxony` | Local earthquakes in NW Bohemia / Vogtland and Saxony, 2009–2024, manual P and S picks, plus noise from quiet periods | BGR, GFZ (SX, WB, TH, CZ) |
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

A live table shows the downloads per data center (done, failed, time left), the
logs run above it. Stop with Ctrl-C and run the same command again to resume.
`--force` moves an existing dataset to a backup folder and compiles it anew.

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
  at the same station.
- **DeepLowFrequency**: Hi-net data may not be redistributed. Hi-net downloads are
  cached in blocks of 5 minutes in `~/.cache/seisbench-datasets/hinet/` (or
  `$SEISBENCH_DATASETS_CACHE`), so compiling again with other windows reuses them.
  The compilation writes `download_stats.json` and `download_failures.csv` to the
  dataset folder.

See [ROADMAP.md](ROADMAP.md) for planned features.

## License

GPL-3.0-or-later, as SeisBench, from which the code originates.
