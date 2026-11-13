# Roadmap

## Fill up

Scan a compiled dataset against its catalog and download the traces that are
missing, e.g., after transient data center failures (`download_failures.csv`) or
when a data center has added data since. Builds on the resume support: the writer
already knows which traces exist (`ResumableWriter.resumed_trace_names`).

## Statistics plots

Basic plots of what matters for training:

- Events: map, magnitude over time, magnitude and depth histograms, events per
  split.
- Stations: map, traces per station, picks per station.
- Traces: P / S pick availability, S − P times and epicentral distances, trace
  completeness, sampling rates, event / noise ratio per split.

## BohemiaSaxony: noise traces

Add noise traces to the dataset, about 30% of the event traces (configurable, as
`noise_fraction` in `DeepLowFrequency`).

Unlike `DeepLowFrequency`, which rejects noise windows by STA/LTA, the noise is
taken from quiet periods: a list of busy (swarm) periods is defined, and noise
windows are drawn only outside of them. This keeps the noise realistic, without
an STA/LTA trigger biasing it towards the quietest windows.

The swarm windows are in `src/seisbench_datasets/data/bohemia_swarms.csv`,
extracted from the catalog with `scripts/bohemia_swarms.py` (busy day: ≥ 10
events; 63 windows, 701 days, about half of the events). The catalog has almost
no events before 2011 (none in 2009): there, no swarm means no monitoring, not
quiet, so noise should only be drawn from the period the catalog covers.
