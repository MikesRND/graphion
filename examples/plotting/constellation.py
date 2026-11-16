"""An I/Q scatter panel fed by a ConstellationViewer block."""

from __future__ import annotations

from math import isfinite
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike

from graphion import SetupError
from graphion.dsp.blocks import ConstellationViewer

from .live import PlotPanel, PlotSeries


def constellation_panel(
    viewer: ConstellationViewer,
    *,
    channel: int = 0,
    reference: ArrayLike | None = None,
    limit: float | None = None,
    title: str = "",
    position: tuple[int, int] | None = None,
) -> PlotPanel:
    """Describe a square I/Q panel of the samples a set-up viewer retains.

    The panel reads the viewer's history, which is NaN until samples are
    captured, so it draws no points before the first run or after a reset.
    reference marks ideal points. limit fixes symmetric axes; by default they
    are 1.6 times the largest reference magnitude, or autoscaled without a
    reference.
    """
    try:
        channels = viewer.ctx.channels
    except SetupError as exc:
        raise ValueError("set up the viewer's model before describing its panel") from exc
    if isinstance(channel, bool) or not isinstance(channel, Integral) or not 0 <= channel < channels:
        raise ValueError(f"channel must be an integer in [0, {channels})")
    if limit is not None and not (isfinite(limit) and limit > 0):
        raise ValueError("limit must be positive and finite")

    series = [
        PlotSeries(
            lambda: viewer.history[channel].real, lambda: viewer.history[channel].imag, label="Received", marker="."
        )
    ]
    if reference is not None:
        ideal = np.asarray(reference, dtype=np.complex128).ravel()
        series.append(PlotSeries(ideal.real, lambda: ideal.imag, label="Ideal", color="black", marker="P"))
        extent = float(np.max(np.abs(ideal))) if ideal.size else 0.0
        if limit is None and extent > 0:
            limit = 1.6 * extent
    limits = None if limit is None else (-limit, limit)
    return PlotPanel(
        series,
        title=title,
        xlabel="In-phase",
        ylabel="Quadrature",
        xlim=limits,
        ylim=limits,
        position=position,
        aspect="equal",
    )
