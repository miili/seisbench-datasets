"""
Helpers shared by the noise windows of the datasets.
"""

from __future__ import annotations

import bisect
import threading

import numpy as np
from obspy import UTCDateTime

# Noise windows end this long before a catalog event and start this long after one
NOISE_GAP_BEFORE_EVENT = 120.0
NOISE_GAP_AFTER_EVENT = 180.0


def is_away_from_events(
    origin_times: list[float], starttime: UTCDateTime, endtime: UTCDateTime
) -> bool:
    """
    Whether a window keeps the noise gaps to all events. ``origin_times`` are sorted
    timestamps.
    """
    i = bisect.bisect_right(origin_times, starttime.timestamp - NOISE_GAP_AFTER_EVENT)
    return (
        i == len(origin_times)
        or origin_times[i] >= endtime.timestamp + NOISE_GAP_BEFORE_EVENT
    )


def check_complete(data: np.ndarray, completeness: float) -> None:
    """
    Raises ValueError if no component has data, or if the components with data have
    gaps. Components without data are allowed, as in the event traces.

    :param completeness: Fraction of the samples that contain data
    """
    has_data = np.ptp(data, axis=1) > 0
    if not has_data.any():
        raise ValueError("No component has data.")
    # Missing components count as incomplete, so compare with the components we have
    if completeness < has_data.sum() / len(data) - 1e-3:
        raise ValueError(f"The window has gaps, completeness {completeness:.3f}.")


class EarthquakeDetector:
    """
    Detects earthquake phases in a noise window with PhaseNet, a deep-learning
    picker, so that only windows with earthquakes are rejected, not those with other
    transients, e.g., spikes, traffic or wind.

    :param weights: PhaseNet weights, e.g., ``original`` (Zhu & Beroza, 2019,
        trained on Northern California) or ``volpick`` (volcanic events)
    :param component_order: Component order of the data
    """

    def __init__(self, weights: str = "original", component_order: str = "ZNE"):
        import seisbench.models as sbm

        self.weights = weights
        self._model = sbm.PhaseNet.from_pretrained(weights)
        self._model.eval()
        self._order = [component_order.index(c) for c in self._model.component_order]
        self._phases = [self._model.labels.index(phase) for phase in "PS"]
        self._blinding = tuple(self._model.default_args.get("blinding", (0, 0)))
        # One inference at a time, so that the download threads don't oversubscribe
        # the CPU
        self._lock = threading.Lock()

    @property
    def sampling_rate(self) -> float:
        return self._model.sampling_rate

    def max_probability(self, data: np.ndarray, start: int = 0) -> float:
        """
        The highest P or S probability from sample ``start`` on. The data before
        ``start`` give the first windows context.

        :param data: Waveforms (components, samples) at :py:attr:`sampling_rate`
        """
        import torch

        n = self._model.in_samples
        x = data[self._order].astype(np.float32)
        if x.shape[1] < n:
            x = np.pad(x, ((0, 0), (0, n - x.shape[1])))
        length = x.shape[1]
        starts = list(range(0, length - n + 1, n // 2))
        if starts[-1] != length - n:
            starts.append(length - n)
        windows = torch.from_numpy(np.stack([x[:, i : i + n] for i in starts]))

        with self._lock, torch.no_grad():
            batch = self._model.annotate_batch_pre(windows, {})
            probabilities = self._model(batch).numpy()

        best = 0.0
        before, after = self._blinding
        for i, window in zip(starts, probabilities, strict=True):
            # Window edges are unreliable, except at the ends of the data
            first = max(start - i, before if i > 0 else 0)
            last = n - after if i + n < length else n
            if last > first:
                best = max(best, float(window[self._phases, first:last].max()))
        return best
