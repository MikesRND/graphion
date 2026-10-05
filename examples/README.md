# Runnable examples and example libraries

The demos share two small libraries that live here while their interfaces
develop: [blocks](blocks/README.md) for Graphion processing and
[plotting](plotting/README.md) for host-side figures and animation.
They are repository-local Python packages, separate from the installed
`graphion` package. Import them as `examples.blocks` and `examples.plotting`.

Set up the development environment following the root [README](../README.md).
It includes Matplotlib and PySide6 for live plotting. Run these from the repository root:

```bash
uv run python -m examples.ddc.ddc_tb                    # three-channel spectrum demo
uv run python -m examples.time_domain                  # input/filtered time traces and spectra
uv run python -m examples.ddc.ddc_tb --save ddc.png      # headless spectrum snapshot
uv run python -m examples.time_domain --save traces.png
```

The [DDC demo](ddc/README.md) shows real-to-complex conversion, exact rate
changes, input fan-out and nested composites. Its first tuner sweeps over 30
seconds, using an unaveraged live spectrum to keep the moving peak sharp.

The live demos use Matplotlib's QtAgg backend and need a graphical desktop.
`--save` works without a GUI.

The [time-domain demo](time_domain.py) drives Graphion's OnePoleLowpass filter with
the shared Tone source. One panel overlays input and filtered waveforms;
another overlays their spectra using the shared Spectrum block. Its live
callback varies source amplitude over ten seconds.

Each model can run without Matplotlib. Plot descriptions use data getters;
the generic plotter advances the model through the demo's callback, then
reads its latest arrays into retained lines. The same figure can save an
image without a GUI. Tuning and domain annotations remain in the demos.
