# seisbench-models

Compile [SeisBench](https://github.com/seisbench/seisbench) training datasets from
earthquake catalogs and waveforms from FDSN web services and NIED Hi-net.

The datasets are ordinary SeisBench `WaveformBenchmarkDataset`s. They are compiled
into the SeisBench cache on first use, and an interrupted download resumes where it
stopped.

| Dataset | Content | Sources |
| --- | --- | --- |
| `BohemiaSaxony` | Local earthquakes in NW Bohemia / Vogtland and Saxony, 2009–2024, manual P and S picks | BGR, GFZ (SX, WB, TH, CZ) |
| `DeepLowFrequency` | Deep low-frequency (DLF / DLP) events below volcanoes in Alaska, Japan, Hawaii, N. California, the Cascades and the Eifel, plus noise windows | EarthScope, NCEDC, BGR, GFZ, NIED Hi-net |

## Install

```sh
uv sync                    # or: pip install .
uv sync --extra hinet      # Japan region of DeepLowFrequency
```

The Hi-net download also needs a Hi-net account and the Hi-net win32tools
(`catwin32` and `win2sac_32` on the `PATH`).

## Use

```python
from seisbench_models.data import BohemiaSaxony, DeepLowFrequency

data = BohemiaSaxony(eida_token="~/.eidatoken", component_order="ZNE")
dlf = DeepLowFrequency(catalog_dir="quakeml/")  # one QuakeML file per region
```

Or compile from the command line:

```sh
uv run python -m seisbench_models.data.bohemia --eida-token ~/.eidatoken
uv run python -m seisbench_models.data.dlf --catalog-dir quakeml/ [--hinet-user USER]
```

`--force` moves an existing dataset to a backup folder and compiles it anew.

### Notes

- **BohemiaSaxony**: the WEBNET (WB) data are restricted and need an
  [EIDA token](https://doi.org/10.7914/SN/WB). No precompiled version is distributed.
- **DeepLowFrequency**: Hi-net data may not be redistributed. Hi-net downloads are
  cached in blocks of 5 minutes, so compiling again with other windows reuses them.
  The compilation writes `download_stats.json` and `download_failures.csv` to the
  dataset folder.

See [ROADMAP.md](ROADMAP.md) for planned features.

## License

GPL-3.0-or-later, as SeisBench, from which the code originates.
