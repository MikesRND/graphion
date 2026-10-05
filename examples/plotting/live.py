"""Retained line plots driven by caller-supplied data getters and a step callback."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from matplotlib.animation import FuncAnimation
    from matplotlib.axes import Axes
    from matplotlib.lines import Line2D


@dataclass(frozen=True)
class PlotSeries:
    x: ArrayLike
    y: Callable[[], ArrayLike]
    label: str | None = None
    color: str | None = None


@dataclass(frozen=True)
class PlotPanel:
    series: Sequence[PlotSeries]
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    position: tuple[int, int] | None = None
    span: tuple[int, int] = (1, 1)


class LivePlotter:
    """Own a figure and update existing lines after one optional model step.

    Panels default to a vertical stack. Explicit positions and spans are
    (row, column) pairs. Fixed limits are retained; omitted limits autoscale.
    A step receives the frame index and elapsed wall-clock seconds.
    """

    def __init__(
        self,
        panels: Sequence[PlotPanel],
        step: Callable[[int, float], None] | None = None,
        *,
        title: str = "",
        interval_ms: float = 50,
        figsize: tuple[float, float] = (12, 7),
        headless: bool = False,
    ) -> None:
        if not panels or not isfinite(interval_ms) or interval_ms <= 0:
            raise ValueError("provide at least one panel and a positive finite update interval")
        self.panels = tuple(panels)
        positions = [panel.position if panel.position is not None else (i, 0) for i, panel in enumerate(panels)]
        for panel, (row, column) in zip(panels, positions):
            if row < 0 or column < 0 or min(panel.span) < 1 or not panel.series:
                raise ValueError("panels need series, nonnegative positions and positive spans")
        self.step = step
        self.interval_ms = interval_ms
        self._started_at: float | None = None
        self._animation: FuncAnimation | None = None
        self._headless = headless

        if headless:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure

            self.figure = Figure(figsize=figsize, layout="constrained")
            FigureCanvasAgg(self.figure)
        else:
            import matplotlib.pyplot as plt

            self.figure = plt.figure(figsize=figsize, layout="constrained")
        self.figure.suptitle(title)
        rows = max(row + panel.span[0] for panel, (row, column) in zip(panels, positions))
        columns = max(column + panel.span[1] for panel, (row, column) in zip(panels, positions))
        grid = self.figure.add_gridspec(rows, columns)
        self.axes: list[Axes] = []
        self.lines: list[tuple[Line2D, ...]] = []
        for panel, (row, column) in zip(panels, positions):
            ax = self.figure.add_subplot(grid[row : row + panel.span[0], column : column + panel.span[1]])
            ax.set(title=panel.title, xlabel=panel.xlabel, ylabel=panel.ylabel)
            if panel.xlim is not None:
                ax.set_xlim(*panel.xlim)
            if panel.ylim is not None:
                ax.set_ylim(*panel.ylim)
            ax.grid(alpha=0.25)
            lines = tuple(
                ax.plot([], [], label=series.label, color=series.color, linewidth=1)[0] for series in panel.series
            )
            if any(series.label for series in panel.series):
                ax.legend()
            self.axes.append(ax)
            self.lines.append(lines)

    def refresh(self) -> tuple[Line2D, ...]:
        """Read the latest data without advancing the model."""
        for panel, ax, lines in zip(self.panels, self.axes, self.lines):
            for series, line in zip(panel.series, lines):
                line.set_data(series.x, series.y())
            if panel.xlim is None or panel.ylim is None:
                ax.relim()
                ax.autoscale_view(scalex=panel.xlim is None, scaley=panel.ylim is None)
        return tuple(line for lines in self.lines for line in lines)

    def update(self, frame: int, elapsed_s: float | None = None) -> tuple[Line2D, ...]:
        """Advance once, then read the resulting data; explicit time supports manual updates."""
        if elapsed_s is None:
            now = perf_counter()
            if self._started_at is None:
                self._started_at = now
            elapsed_s = now - self._started_at
        if self.step is not None:
            self.step(frame, elapsed_s)
        return self.refresh()

    def show(self) -> None:
        """Open the window and retain the animation until the caller closes it."""
        if self._headless:
            raise ValueError("a headless plotter can save images but cannot open a window")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        if plt.get_backend().lower() == "agg":
            raise RuntimeError(
                "Live plots need a GUI Matplotlib backend and a graphical desktop. "
                "Run uv sync to install PySide6, or run the example with --save."
            )
        self._started_at = perf_counter()
        self._animation = FuncAnimation(
            self.figure,
            self.update,
            init_func=self.refresh,
            interval=self.interval_ms,
            cache_frame_data=False,
        )
        plt.show(block=True)

    def save(self, path: str | Path, *, dpi: float = 150) -> None:
        """Save the latest data without advancing the model or changing the global backend."""
        self.refresh()
        self.figure.savefig(path, dpi=dpi)

    def close(self) -> None:
        if self._animation is not None and self._animation.event_source is not None:
            self._animation.event_source.stop()
        if not self._headless:
            import matplotlib.pyplot as plt

            plt.close(self.figure)
