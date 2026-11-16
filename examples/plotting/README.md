# Example plotting library

`LivePlotter` creates subplot layouts, retains lines, runs an optional step
callback and refreshes data from getters. It supports spectra, time traces,
constellations and other x/y arrays. Importing the library requires no Matplotlib; constructing
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

Each series has x values and a getter for its current y values. The x values
are fixed, or a getter when they change too. A series with `marker="."` draws
points without a connecting line. Multiple series overlay on a panel and
labeled series get a legend. Omitted axis limits autoscale; explicit limits
stay fixed. `aspect="equal"` gives a panel equal axis scales. Panels default to
a vertical stack, or use `position=(row, column)` and `span=(rows, columns)`
for a grid.

## Constellation panels

`constellation_panel` describes a square I/Q panel for a
[ConstellationViewer](../../docs/reference/dsp-library.md#block-contracts)
block. Add the block to a model, connect it to a complex symbol stream, and
pass it to the helper:

```python
from examples.plotting import LivePlotter, constellation_panel
from graphion.dsp.blocks import ConstellationViewer
from graphion.dsp.communications import qpsk

# In the model's build(), then set up and run it from the model as usual:
#     self.viewer = self.child("viewer", ConstellationViewer(depth=1024))
#     self.connect(self.rx, "y", self.viewer, "x")


def make_plot(model, ticks):
    panel = constellation_panel(model.viewer, reference=qpsk(), title="Received")
    return LivePlotter([panel], step=lambda frame, elapsed_s: model.run(), frames=ticks)
```

Set the model up before describing the panel. `reference` marks ideal points
and sets symmetric limits around them; `limit` fixes the limits directly, and
`channel` selects one of the viewer's channels. The panel reads the viewer's
retained history, which holds no samples until the first run and again after
a reset, so it starts empty without running the model first. The
[live BER runner](../ber/live.py) relies on this: only its ticks advance the model.

The block is part of the installed `graphion` package. This helper is part
of the example plotting library and is not installed with it.

The step receives the frame index and elapsed wall-clock seconds. It runs once
before any data getters, so borrowed model buffers are read after processing.
By default a shown plotter steps until its window is closed. With `frames=n`
it steps exactly n times, then stops and keeps the final state on screen.
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
