"""
Helpers shared by the noise windows of the datasets.
"""

from __future__ import annotations

import bisect

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
