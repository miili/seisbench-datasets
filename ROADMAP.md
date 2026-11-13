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
