"""BPSK and QPSK BER versus Eb/N0 through pulse shaping and an AWGN channel."""

import argparse
import csv
import math
from dataclasses import asdict, dataclass
from fractions import Fraction
from time import perf_counter

import numpy as np

from graphion.dsp.blocks import (
    Awgn,
    ErrorCounter,
    FirDecimate,
    FirInterpolate,
    Prbs,
    SampleDelay,
    SymbolDemapper,
    SymbolMapper,
)
from graphion.dsp.communications import bpsk, bpsk_qpsk_ber, ebn0_to_snr_db, qpsk
from graphion.dsp.filter_design import rrc
from graphion.dsp.streams import stream_ctx
from graphion.profile.numpy_frame import Array, Block, Ctx, Param

from .prediction import FiniteFilterPrediction


def _matched_alignment(ntaps, sps, bits_per_symbol):
    """Sample delay, phase, reference-bit delay and startup skip for matched taps."""
    sample_delay = ntaps - 1  # RX is the conjugate reverse of TX, with equal length.
    phase = sample_delay % sps
    bit_delay = ((sample_delay - phase) // sps) * bits_per_symbol
    return sample_delay, phase, bit_delay, 2 * bit_delay


def _frame_budget(target, bit_cap, frame_size, skip):
    """Round the target to frames while respecting the cap on compared bits."""
    frames = min(math.ceil((target + skip) / frame_size), (bit_cap + skip) // frame_size)
    if frames * frame_size <= skip:
        raise ValueError("bit cap cannot accommodate a full frame with compared bits")
    return frames


class BerTestbench(Block):
    """Bit-context testbench; optional tx_taps also exercises sampling phase."""

    constellation: Array = Param()
    ebn0_db: float = Param(default=0.0)
    snr_db: object = Param(default=None)
    sps: int = Param(default=8)
    span: int = Param(default=12)
    rolloff: float = Param(default=0.35)
    seed: object = Param(default=0)
    tx_taps: object = Param(default=None)

    def build(self):
        self.taps = rrc(self.span, self.sps, self.rolloff) if self.tx_taps is None else np.asarray(self.tx_taps)
        table = np.asarray(self.constellation)
        self.k = len(table).bit_length() - 1
        rx_taps = self.taps.conj()[::-1].copy()
        self.sample_delay, self.phase, self.bit_delay, self.skip = _matched_alignment(len(self.taps), self.sps, self.k)
        power = float(np.mean(np.abs(table) ** 2) * np.sum(np.abs(self.taps) ** 2) / self.sps)
        snr = float(ebn0_to_snr_db(self.ebn0_db, self.k, self.sps)) if self.snr_db is None else self.snr_db
        self.prbs = self.child("prbs", Prbs())
        self.mapper = self.child("mapper", SymbolMapper(constellation=table))
        self.tx = self.child("tx", FirInterpolate(taps=self.taps, factor=self.sps))
        self.chan = self.child("chan", Awgn(snr_db=snr, signal_power=power, seed=self.seed))
        self.rx = self.child("rx", FirDecimate(taps=rx_taps, factor=self.sps, phase=self.phase))
        self.demap = self.child("demap", SymbolDemapper(constellation=table))
        self.delay = self.child("delay", SampleDelay(delay=self.bit_delay))
        self.counter = self.child("counter", ErrorCounter(skip=self.skip))
        self._chain = (self.prbs, self.mapper, self.tx, self.chan, self.rx, self.demap)
        for source, destination in zip(self._chain, self._chain[1:]):
            self.connect(source, "y", destination, "x")
        self.connect(self.demap, "y", self.counter, "x")
        self.connect(self.prbs, "y", self.delay, "x")
        self.connect(self.delay, "y", self.counter, "ref")

    def setup(self, ctx):
        c = ctx
        for stage in self._chain:
            stage.setup(c)
            c = stream_ctx(stage, "y")
        self.counter.setup(c)
        self.delay.setup(stream_ctx(self.prbs, "y"))

    def process(self):
        for stage in self._chain:
            stage.run()
        self.delay.run()
        self.counter.run()


def agreement(errors: int, total: int, prediction: float) -> tuple[float, str]:
    """Return deviation in predicted standard errors and a five-sigma verdict."""
    if total <= 0 or not 0 <= errors <= total or not 0 <= prediction <= 1:
        raise ValueError("agreement needs 0 <= errors <= total, total > 0 and a probability")
    sigma = math.sqrt(prediction * (1 - prediction) / total)
    delta = errors / total - prediction
    deviation = delta / sigma if sigma else (0.0 if delta == 0 else math.copysign(math.inf, delta))
    if prediction == 0 and errors > 0:
        return deviation, "FAIL"
    if total * prediction <= 25:
        return deviation, "INCONCLUSIVE"
    return deviation, "PASS" if abs(delta) <= 5 * sigma else "FAIL"


def confidence_interval(errors: int, total: int) -> tuple[float, float]:
    """95% Wilson interval, or exact one-sided 95% upper bound at zero errors."""
    if total <= 0 or not 0 <= errors <= total:
        raise ValueError("confidence interval needs 0 <= errors <= total and total > 0")
    if errors == 0:
        return 0.0, -math.expm1(math.log(0.05) / total)
    z = 1.959963984540054
    p = errors / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


@dataclass(frozen=True)
class Result:
    modulation: str
    ebn0_db: float
    bits: int
    errors: int
    ber: float
    theory: float
    prediction: float
    penalty: float
    deviation: float
    status: str
    lower95: float
    upper95: float


def simulate(table, ebn0_db, *, frames, frame_size=4096, seed=0, **kwargs):
    """Use a budget fixed before simulation; return the configured testbench."""
    if frames < 1:
        raise ValueError("frames must be positive")
    tb = BerTestbench(constellation=table, ebn0_db=ebn0_db, seed=seed, **kwargs)
    k = len(table).bit_length() - 1
    tb.setup(Ctx(n=frame_size, channels=1, rate=Fraction(k * 1_000_000)))
    for _ in range(frames):
        tb.run()
    return tb


def sweep(points, *, error_target=1000, bit_cap=10_000_000, frame_size=4096, seed=0, progress=None, **kwargs):
    """Frame-rounded fixed budgets, capped in compared bits; no optional stopping."""
    if not np.isfinite(error_target) or error_target <= 0 or bit_cap < 1 or frame_size < 1 or frame_size % 2:
        raise ValueError("need positive target/cap and a positive even bit frame size")
    results = []
    for modulation_index, (name, table) in enumerate((("BPSK", bpsk()), ("QPSK", qpsk()))):
        predictor = None
        for point_index, db in enumerate(points):
            theory = float(bpsk_qpsk_ber(db))
            # Set up once to obtain the exact startup exclusion before rounding.
            tb = BerTestbench(
                constellation=table,
                ebn0_db=float(db),
                seed=np.random.SeedSequence([seed, modulation_index, point_index]),
                **kwargs,
            )
            tb.setup(Ctx(n=frame_size, channels=1, rate=Fraction((len(table).bit_length() - 1) * 1_000_000)))
            if predictor is None:
                predictor = FiniteFilterPrediction(table, tb.taps, tb.sps)
            prediction = predictor.ber(tb.chan.snr_db)
            penalty = prediction / theory if theory else (math.inf if prediction else math.nan)
            target = math.ceil(error_target / prediction) if prediction > error_target / bit_cap else bit_cap
            frames = _frame_budget(target, bit_cap, frame_size, tb.skip)
            if progress is not None:
                progress(tb, name, 0, frames)
            next_refresh = perf_counter() + 0.1
            for frame in range(1, frames + 1):
                tb.run()
                if progress is not None and (frame == frames or perf_counter() >= next_refresh):
                    progress(tb, name, frame, frames)
                    next_refresh = perf_counter() + 0.1
            total, errors = int(tb.counter.total[0]), int(tb.counter.errors[0])
            deviation, status = agreement(errors, total, prediction)
            lower, upper = confidence_interval(errors, total)
            results.append(
                Result(
                    name,
                    float(db),
                    total,
                    errors,
                    errors / total,
                    theory,
                    prediction,
                    penalty,
                    deviation,
                    status,
                    lower,
                    upper,
                )
            )
    return results


def plot_results(results, *, save=None):
    if save:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        fig = Figure(figsize=(8, 5), layout="constrained")
        FigureCanvasAgg(fig)
    else:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(8, 5), layout="constrained")
    ax = fig.subplots()
    for index, (name, marker) in enumerate((("BPSK", "o"), ("QPSK", "s"))):
        rows = [row for row in results if row.modulation == name]
        y = np.array([row.ber for row in rows])
        ax.errorbar(
            [row.ebn0_db for row in rows],
            y,
            yerr=[y - [row.lower95 for row in rows], [row.upper95 for row in rows] - y],
            fmt=marker,
            color=f"C{index}",
            capsize=3,
            label=name + " (95% intervals)",
        )
        ax.plot(
            [row.ebn0_db for row in rows],
            [row.prediction for row in rows],
            color=f"C{index}",
            label=name + " finite-filter prediction",
        )
    db = np.linspace(min(row.ebn0_db for row in results), max(row.ebn0_db for row in results), 201)
    ax.semilogy(db, bpsk_qpsk_ber(db), "k--", label="Coherent AWGN theory")
    ax.set(xlabel="Eb/N0 (dB)", ylabel="Bit error rate", title="Pulse-shaped BPSK and Gray QPSK in AWGN")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    if save:
        fig.savefig(save, dpi=160)
    else:
        plt.show()
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--save", metavar="PNG", help="save a plot headlessly")
    display.add_argument("--diagram", action="store_true", help="show the model diagram while the sweep runs")
    parser.add_argument("--csv", metavar="FILE")
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--stop", type=float, default=8)
    parser.add_argument("--step", type=float, default=1)
    parser.add_argument("--error-target", type=float, default=1000)
    parser.add_argument("--bit-cap", type=int, default=10_000_000)
    parser.add_argument("--frame-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rolloff", type=float, default=0.35)
    parser.add_argument("--span", type=int, default=12)
    parser.add_argument("--sps", type=int, default=8)
    args = parser.parse_args(argv)
    if (
        not all(math.isfinite(v) for v in (args.start, args.stop, args.step))
        or args.step <= 0
        or args.stop < args.start
    ):
        parser.error("sweep needs finite start <= stop and step > 0")
    if args.frame_size < 1 or args.frame_size % 2:
        parser.error("--frame-size must be a positive even number of bits")
    if not math.isfinite(args.error_target) or args.error_target <= 0:
        parser.error("--error-target must be finite and positive")
    if args.bit_cap < 1:
        parser.error("--bit-cap must be positive")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.span < 1 or args.sps < 1:
        parser.error("--span and --sps must be positive")
    if not math.isfinite(args.rolloff) or not 0 <= args.rolloff <= 1:
        parser.error("--rolloff must lie in [0, 1]")
    try:
        for k in (1, 2):
            _, _, _, skip = _matched_alignment(args.span * args.sps + 1, args.sps, k)
            _frame_budget(args.bit_cap, args.bit_cap, args.frame_size, skip)
    except ValueError as exc:
        parser.error(str(exc))
    points = np.arange(args.start, args.stop + args.step * 1e-9, args.step)
    diagram = None
    if args.diagram:
        from .diagram import BerDiagram

        diagram = BerDiagram()
    try:
        results = sweep(
            points,
            error_target=args.error_target,
            bit_cap=args.bit_cap,
            frame_size=args.frame_size,
            seed=args.seed,
            rolloff=args.rolloff,
            span=args.span,
            sps=args.sps,
            progress=diagram.update if diagram is not None else None,
        )
    except BaseException:
        if diagram is not None:
            diagram.close()
        raise
    if diagram is not None:
        diagram.finish()
    print(
        "Mod   Eb/N0       bits   errors      BER       theory    prediction  pred/theory  deviation  status         95% interval"
    )
    for row in results:
        print(
            f"{row.modulation:4}  {row.ebn0_db:5.1f}  {row.bits:9d}  {row.errors:7d}  {row.ber:.3e}  "
            f"{row.theory:.3e}  {row.prediction:.3e}  {row.penalty:10.5f}  {row.deviation:+7.2f}σ  {row.status:12}  [{row.lower95:.3e}, {row.upper95:.3e}]"
        )
    if args.csv:
        with open(args.csv, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(asdict(results[0])))
            writer.writeheader()
            writer.writerows(asdict(row) for row in results)
    try:
        plot_results(results, save=args.save)
    finally:
        if diagram is not None:
            diagram.close()
    return int(any(row.status == "FAIL" for row in results))


if __name__ == "__main__":
    raise SystemExit(main())
