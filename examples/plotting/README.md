# Example plotting library

`LivePlotter` creates subplot layouts, retains lines, runs an optional step
callback and refreshes data from getters. It supports spectra, time traces
and other x/y arrays. Importing the library requires no Matplotlib; constructing
a plotter requires the optional plotting dependency.

```python
from examples.plotting import LivePlotter, PlotPanel, PlotSeries

# Set up and run your block once so its output contains meaningful data.
plotter = LivePlotter(
    [
        PlotPanel(
            [PlotSeries(time_ms, lambda: block.y[0], label="Output")],
            xlabel="Time (ms)",
            ylabel="Amplitude",
            ylim=(-1, 1),
        )
    ],
    step=lambda frame, elapsed_s: block.run(),
    interval_ms=50,
)
plotter.show()
plotter.close()
```

Each series has fixed x values and a getter for its current y values. Multiple
series overlay on a panel and labeled series get a legend. Omitted axis limits
autoscale; explicit limits stay fixed. Panels default to a vertical stack, or
use `position=(row, column)` and `span=(rows, columns)` for a grid.

The step receives the frame index and elapsed wall-clock seconds. It runs once
before any data getters, so borrowed model buffers are read after processing.
Use elapsed seconds for controls whose timing should follow actual time.
The [DDC demo](../ddc/ddc_tb.py) adds its band shading and tuner readout using
the plotter's public `axes` and its own step callback.

For an image without a window, construct with `headless=True`, advance your
model as needed, then call `plotter.save("out.png")`. This uses an Agg canvas
without changing Matplotlib's global backend. `save()` and `refresh()` read
current values without advancing the model. `update(frame, elapsed_s=...)`
supports manual or deterministic updates using the same step callback.

The [time-domain demo](../time_domain.py) shows two overlaid series on each of
two panels. Both demos use this library; it is an example API rather than a
public Graphion API.
