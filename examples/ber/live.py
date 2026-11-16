"""Live BPSK and QPSK received constellations at one fixed Es/N0.

Each tick advances both testbenches by one frame. The run stops after --ticks
and keeps the final constellations on screen; closing the window ends it early.
Nothing but a tick advances the models, and the panels are empty until the first.
--diagram also opens the model diagram of one of the two testbenches.
"""

import argparse
import math
from fractions import Fraction
from time import perf_counter

import numpy as np

from examples.plotting import LivePlotter, constellation_panel
from graphion.dsp.communications import bpsk, qpsk
from graphion.profile.numpy_frame import Ctx

from .ber_tb import BerTestbench


def make_models(esn0_db=10.0, *, depth=1024, frame_size=4096, seed=0, **kwargs):
    """Configured viewer-equipped testbenches by modulation name, at equal Es/N0.

    Eb/N0 = Es/N0 - 10 log10(bits per symbol), so QPSK runs 3.01 dB below BPSK.
    """
    models = {}
    for index, (name, table) in enumerate((("BPSK", bpsk()), ("QPSK", qpsk()))):
        k = len(table).bit_length() - 1
        model = BerTestbench(
            constellation=table,
            ebn0_db=esn0_db - 10 * math.log10(k),
            seed=np.random.SeedSequence([seed, index]),
            viewer_depth=depth,
            **kwargs,
        )
        model.setup(Ctx(n=frame_size, channels=1, rate=Fraction(k * 1_000_000)))
        models[name] = model
    return models


class LiveRun:
    """Advance every model one frame per tick and count completed ticks."""

    def __init__(self, models):
        self.models = models
        self.ticks = 0

    def advance(self):
        for model in self.models.values():
            model.run()
        self.ticks += 1


def make_plot(run, esn0_db, *, ticks, headless=False, interval_ms=50, on_tick=None):
    """One constellation panel per model; the plotter's step is the only thing that advances them.

    on_tick, if given, is called without arguments after each completed tick.
    """
    panels = [
        constellation_panel(model.viewer, reference=model.constellation, position=(0, column))
        for column, model in enumerate(run.models.values())
    ]

    def label():
        plotter.figure.suptitle(f"Received constellations · Es/N0 {esn0_db:g} dB · tick {run.ticks:,} / {ticks:,}")
        for ax, (name, model) in zip(plotter.axes, run.models.items()):
            ber = model.counter.rate[0] if run.ticks else math.nan
            measured = "BER —" if math.isnan(ber) else f"BER {ber:.2e}"
            ax.set_title(f"{name} · Eb/N0 {model.ebn0_db:.2f} dB · {measured}")

    def step(frame, elapsed_s):
        run.advance()
        label()
        if on_tick is not None:
            on_tick()

    plotter = LivePlotter(panels, step, interval_ms=interval_ms, figsize=(11, 5.8), headless=headless, frames=ticks)
    label()
    return plotter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--esn0-db", type=float, default=10.0, help="symbol energy to noise density, both modulations")
    parser.add_argument("--ticks", type=int, default=300, help="frames to run; closing the window ends the run early")
    parser.add_argument("--depth", type=int, default=1024, help="most recent symbols shown per constellation")
    parser.add_argument("--frame-size", type=int, default=4096, help="bits per frame")
    parser.add_argument("--seed", type=int, default=0)
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--save", metavar="PNG", help="run headlessly and save the final constellations")
    display.add_argument(
        "--diagram",
        nargs="?",
        const="qpsk",
        type=str.lower,
        choices=("bpsk", "qpsk"),
        metavar="MODULATION",
        help="also show the model diagram of the bpsk or qpsk testbench (qpsk if not named)",
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.esn0_db):
        parser.error("--esn0-db must be finite")
    if args.ticks < 1:
        parser.error("--ticks must be a positive integer")
    if args.depth < 1:
        parser.error("--depth must be a positive integer")
    if args.frame_size < 1 or args.frame_size % 2:
        parser.error("--frame-size must be a positive even number of bits")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    run = LiveRun(make_models(args.esn0_db, depth=args.depth, frame_size=args.frame_size, seed=args.seed))
    show_model = None
    closers = []
    try:
        if args.diagram:
            from .diagram import BerDiagram

            diagram = BerDiagram()
            closers.append(diagram.close)
            name = args.diagram.upper()
            next_refresh = 0.0

            def show_model():
                # Redrawing the diagram on every tick would slow the constellations, so
                # refresh it a few times a second and at the end. The plotter's event
                # loop is running, so the diagram only requests a redraw.
                nonlocal next_refresh
                if run.ticks == args.ticks or perf_counter() >= next_refresh:
                    diagram.update(run.models[name], name, run.ticks, args.ticks, service_events=False)
                    next_refresh = perf_counter() + 0.25

        plotter = make_plot(run, args.esn0_db, ticks=args.ticks, headless=bool(args.save), on_tick=show_model)
        closers.append(plotter.close)
        if show_model is not None:
            show_model()
            # Closing the constellation window ends the run, so it takes the diagram with it.
            plotter.figure.canvas.mpl_connect("close_event", lambda event: diagram.close())
        if args.save:
            for frame in range(args.ticks):
                plotter.update(frame, elapsed_s=frame * plotter.interval_ms / 1000)
            plotter.save(args.save)
        else:
            plotter.show()
    finally:
        for close in reversed(closers):
            close()
    print(f"Completed {run.ticks:,} of {args.ticks:,} ticks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
