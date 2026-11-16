import csv
import math
from fractions import Fraction

import numpy as np
import pytest

from examples.ber.ber_tb import BerTestbench, agreement, confidence_interval, main, plot_results, simulate, sweep
from examples.ber.prediction import FiniteFilterPrediction
from graphion.dsp.communications import bpsk, bpsk_qpsk_ber, ebn0_to_snr_db, qpsk
from graphion.dsp.filter_design import rrc
from graphion.profile.numpy_frame import Ctx


@pytest.mark.parametrize("table", [bpsk(), qpsk()])
@pytest.mark.parametrize("n", [2, 30, 128, 4096])
def test_noiseless(table, n):
    tb = simulate(table, 0, frames=max(4, 128 // n), frame_size=n, snr_db=np.inf)
    assert tb.counter.total[0] > 0
    assert tb.counter.errors[0] == 0
    assert tb.sample_delay == 96 and tb.phase == 0
    assert tb.bit_delay == 12 * tb.k
    tb.reset()
    for _ in range(max(4, 128 // n)):
        tb.run()
    assert tb.counter.errors[0] == 0


@pytest.mark.parametrize("table", [bpsk(), qpsk()])
def test_nondefault_span_and_nonzero_phase(table):
    for config, delay, phase in (({"span": 8}, 64, 0), ({"tx_taps": np.r_[0, rrc(12, 8, 0.35)]}, 97, 1)):
        tb = simulate(table, 0, frames=10, frame_size=256, snr_db=np.inf, **config)
        assert tb.sample_delay == delay and tb.phase == phase
        assert tb.counter.errors[0] == 0
        assert tb.bit_delay == ((delay - phase) // 8) * tb.k


@pytest.mark.parametrize("k", [1, 2])
def test_finite_filter_conditional_ber_over_complete_pn_cycle(k):
    # Independent scalar PRBS15, mapping and circular convolution; no testbench
    # filters, decisions, channel or counter enter this deterministic calculation.
    period = 32767
    reg = period
    bits = np.empty(k * period)
    for index in range(len(bits)):
        bits[index] = reg >> 14
        reg = ((reg << 1) | (((reg >> 14) ^ (reg >> 13)) & 1)) & period
    signs = 1 - 2 * bits.reshape(period, k)
    symbols = signs[:, 0] if k == 1 else (signs[:, 0] + 1j * signs[:, 1]) / np.sqrt(2)
    h = rrc(12, 8, 0.35)
    cascade = np.convolve(h, h[::-1])[::8]
    decision = sum(coefficient * np.roll(symbols, index - 12) for index, coefficient in enumerate(cascade))
    margins = (decision.real * signs[:, 0])[:, None]
    if k == 2:
        margins = np.stack([decision.real * signs[:, 0], decision.imag * signs[:, 1]], axis=1)
    predictor = FiniteFilterPrediction(bpsk() if k == 1 else qpsk(), h, 8)
    np.testing.assert_allclose(predictor.decisions, decision, rtol=1e-14, atol=1e-14)
    for db in range(9):
        scale = math.sqrt(k * 10 ** (db / 10))
        conditional = np.mean([0.5 * math.erfc(scale * margin) for margin in margins.flat])
        assert conditional == pytest.approx(bpsk_qpsk_ber(db), rel=0.001)
        assert predictor.ber(float(ebn0_to_snr_db(db, k, 8))) == pytest.approx(conditional, rel=1e-13)


def test_agreement():
    assert agreement(1000, 10000, 0.1)[1] == "PASS"
    assert agreement(2000, 10000, 0.1)[1] == "FAIL"
    assert agreement(0, 10000, 0.1)[1] == "FAIL"
    assert agreement(0, 2500, 0.01)[1] == "INCONCLUSIVE"
    assert agreement(0, 2501, 0.01)[1] == "FAIL"
    assert agreement(0, 100, 0.01)[0] == pytest.approx(-1 / math.sqrt(0.99))
    # No truncation allowance: the predictor already includes finite-filter ISI.
    n = 100_000_000
    p = 0.1
    sigma = math.sqrt(p * (1 - p) / n)
    assert agreement(round(n * (p + 4.99 * sigma)), n, p)[1] == "PASS"
    assert agreement(round(n * (p + 5.01 * sigma)), n, p)[1] == "FAIL"


@pytest.mark.parametrize(
    "errors,total,prediction,status",
    [
        (2, 1_000_000, 1e-9, "INCONCLUSIVE"),
        (3, 1_000_000, 1e-9, "FAIL"),
        (54, 2500, 0.01, "INCONCLUSIVE"),
        (55, 2500, 0.01, "FAIL"),
        (100, 10_000_000, 1e-12, "FAIL"),
        (100, 10_000_000, 1e-300, "FAIL"),
    ],
)
def test_low_count_agreement(errors, total, prediction, status):
    assert agreement(errors, total, prediction)[1] == status


@pytest.mark.parametrize("snr_db", [np.inf, 35.0])
def test_zero_prediction_fails_on_observed_errors(snr_db):
    predictor = FiniteFilterPrediction(bpsk(), rrc(12, 8, 0.35), 8)
    prediction = predictor.ber(snr_db)
    assert prediction == 0
    assert agreement(5, 1000, prediction) == (math.inf, "FAIL")
    assert agreement(0, 1000, prediction) == (0.0, "INCONCLUSIVE")


def test_cli_fails_when_observed_errors_contradict_zero_prediction(monkeypatch, capsys):
    from examples.ber import ber_tb

    monkeypatch.setattr(FiniteFilterPrediction, "ber", lambda self, snr_db: 0.0)
    monkeypatch.setattr(ber_tb, "plot_results", lambda *args, **kwargs: None)
    assert main(["--start", "0", "--stop", "0", "--bit-cap", "4096"]) == 1
    output = capsys.readouterr().out
    assert output.count("FAIL") == 2
    assert "INCONCLUSIVE" not in output


@pytest.mark.parametrize("table", [bpsk(), qpsk()])
@pytest.mark.parametrize("db,frames", [(0, 20), (4, 20), (8, 400)])
def test_seeded_ber(table, db, frames):
    tb = simulate(table, db, frames=frames, seed=np.random.SeedSequence([2026, len(table), db]))
    total, errors = int(tb.counter.total[0]), int(tb.counter.errors[0])
    prediction = FiniteFilterPrediction(table, tb.taps, tb.sps).ber(tb.chan.snr_db)
    assert agreement(errors, total, prediction)[1] == "PASS"


def test_budget_cap_and_seed_reproducibility():
    first = sweep([0, 8], error_target=100, bit_cap=10_000, frame_size=128)
    assert first == sweep([0, 8], error_target=100, bit_cap=10_000, frame_size=128)
    assert all(row.bits <= 10_000 for row in first)
    assert [row.status for row in first if row.ebn0_db == 8] == ["INCONCLUSIVE", "INCONCLUSIVE"]
    for name, skip in (("BPSK", 24), ("QPSK", 48)):
        assert all((row.bits + skip) % 128 == 0 for row in first if row.modulation == name)


def test_confidence_intervals():
    low, high = confidence_interval(50, 100)
    assert low == pytest.approx(0.40383153)
    assert high == pytest.approx(0.59616847)
    low, high = confidence_interval(0, 100)
    assert low == 0 and (1 - high) ** 100 == pytest.approx(0.05)


def test_csv_and_headless_png(tmp_path, capsys):
    png, csv_file = tmp_path / "ber.png", tmp_path / "ber.csv"
    assert (
        main(["--save", str(png), "--csv", str(csv_file), "--start", "0", "--stop", "0", "--error-target", "100"]) == 0
    )
    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    with csv_file.open() as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2 and all(row["status"] == "PASS" for row in rows)
    assert {row["modulation"] for row in rows} == {"BPSK", "QPSK"}
    for row in rows:
        assert float(row["penalty"]) == pytest.approx(float(row["prediction"]) / float(row["theory"]))
    assert "deviation" in capsys.readouterr().out


FILTER_CASES = [
    (rrc(12, 8, 0.35), 8),
    (rrc(4, 8, 0.35), 8),
    (rrc(5, 3, 0), 3),
    (rrc(12, 1, 0.35), 1),
    (np.r_[0, rrc(12, 8, 0.35)], 8),
    (np.multiply(3.2, rrc(8, 4, 0.5)), 4),
    (rrc(4, 3, 0.35) + np.multiply(0.4j, np.roll(rrc(4, 3, 0.35), 1)), 3),
]


@pytest.mark.parametrize("table", [bpsk(), qpsk()])
@pytest.mark.parametrize("taps,sps", FILTER_CASES)
def test_predictor_decisions_match_noiseless_testbench(table, taps, sps):
    tb = simulate(table, 6, frames=1, frame_size=128, snr_db=np.inf, tx_taps=taps, sps=sps)
    predictor = FiniteFilterPrediction(table, taps, sps)
    assert predictor.phase == tb.phase
    assert predictor.symbol_delay * tb.k == tb.bit_delay
    samples = [tb.rx.y[0].copy()]
    for _ in range(8):
        tb.run()
        samples.append(tb.rx.y[0].copy())
    actual = np.concatenate(samples)
    # Exclude the full cascade startup; compare aligned values, not hard bits.
    start = math.ceil((2 * len(taps) - 2 - tb.phase) / sps)
    indices = (np.arange(start, len(actual)) - predictor.symbol_delay) % predictor.period
    np.testing.assert_allclose(actual[start:], predictor.decisions[indices], rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("taps,sps", FILTER_CASES)
def test_nondefault_filters_agree_with_prediction(taps, sps):
    results = sweep([6], error_target=1000, tx_taps=taps, sps=sps)
    for row in results:
        assert row.status == "PASS"
        assert row.penalty == pytest.approx(row.prediction / row.theory)
    # This includes real physical degradation: agreement must use the prediction.
    if sps == 1:
        assert all(row.penalty > 5 for row in results)
        assert all(agreement(row.errors, row.bits, row.theory)[1] == "FAIL" for row in results)


def test_prediction_normalization_and_noise_variance():
    taps = rrc(12, 8, 0.35)
    model = FiniteFilterPrediction(qpsk(), taps, 8)
    scaled = FiniteFilterPrediction(qpsk(), taps * 3.2, 8)
    np.testing.assert_allclose(scaled.decisions, model.decisions * 3.2**2)
    assert model.axis_noise_variance(0) == pytest.approx(1 / 16)
    assert scaled.axis_noise_variance(0) == pytest.approx(model.axis_noise_variance(0) * 3.2**4)
    assert scaled.ber(0) == pytest.approx(model.ber(0), rel=1e-13)
    assert model.ber(np.inf) == 0
    with pytest.raises(ValueError, match="BPSK and QPSK"):
        FiniteFilterPrediction([1, -1, 1j, -1j], taps, 8)


def test_headless_plot_preserves_global_backend(tmp_path, monkeypatch):
    import matplotlib

    monkeypatch.setitem(matplotlib.rcParams, "backend", "svg")
    before = matplotlib.get_backend()
    plot_results(sweep([0], error_target=100), save=tmp_path / "ber.png")
    assert matplotlib.get_backend() == before


@pytest.mark.parametrize(
    "args",
    [
        ["--frame-size", "4095"],
        ["--frame-size", "0"],
        ["--bit-cap", "0"],
        ["--bit-cap", "1"],
        ["--error-target", "0"],
        ["--error-target", "nan"],
        ["--seed", "-1"],
        ["--span", "0"],
        ["--sps", "0"],
        ["--rolloff", "1.1"],
        ["--rolloff", "nan"],
        ["--step", "0"],
    ],
)
def test_invalid_cli_arguments_are_usage_errors(args, capsys):
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert "error:" in error and "Traceback" not in error


def test_cli_checks_qpsk_budget_before_starting_bpsk(monkeypatch, capsys):
    from examples.ber import ber_tb

    # One BPSK frame compares 8 bits. QPSK requires two frames and compares 16.
    bpsk_model = simulate(bpsk(), 0, frames=1, frame_size=32)
    assert bpsk_model.counter.total[0] == 8

    def unexpected_sweep(*args, **kwargs):
        pytest.fail("CLI must validate both modulations before starting the sweep")

    monkeypatch.setattr(ber_tb, "sweep", unexpected_sweep)
    with pytest.raises(SystemExit) as exc:
        main(["--frame-size", "32", "--bit-cap", "8"])
    assert exc.value.code == 2
    assert "bit cap" in capsys.readouterr().err


@pytest.mark.parametrize(
    "kwargs", [{"frame_size": 3}, {"bit_cap": 1}, {"error_target": -1}, {"frame_size": 32, "bit_cap": 8}]
)
def test_invalid_programmatic_sweep_remains_value_error(kwargs):
    with pytest.raises(ValueError):
        sweep([0], **kwargs)


def test_viewer_is_absent_by_default():
    tb = simulate(qpsk(), 4, frames=1)
    assert tb.viewer is None
    assert len(tb.describe()["blocks"]) == 9  # The root and the eight blocks of the curve model.
    assert len(tb.describe()["connections"]) == 8


@pytest.mark.parametrize("depth", [-1, 1.5, True])
def test_invalid_viewer_depth(depth):
    with pytest.raises(ValueError, match="viewer_depth"):
        BerTestbench(constellation=qpsk(), viewer_depth=depth)


@pytest.mark.parametrize("table", [bpsk(), qpsk()])
def test_viewer_retains_matched_filter_output(table):
    depth = 300  # Longer than a frame of either modulation's symbols.
    tb = BerTestbench(constellation=table, ebn0_db=4, viewer_depth=depth)
    tb.setup(Ctx(n=256, channels=1, rate=Fraction(tb.k * 1_000_000)))
    viewer = tb.viewer
    assert viewer is not None
    assert viewer.skip == tb.skip // tb.k == 24
    received = []
    for _ in range(6):
        tb.run()
        assert np.shares_memory(viewer.x, tb.demap.x)  # Fan-out borrows the decision input.
        received.append(tb.rx.y[0].copy())
        kept = np.concatenate(received)[viewer.skip :][-depth:]
        expected = np.full(depth, np.nan, dtype=np.complex128)
        expected[depth - len(kept) :] = kept
        np.testing.assert_array_equal(viewer.points[0], expected)
    assert not np.isnan(viewer.points).any()


@pytest.mark.parametrize("table", [bpsk(), qpsk()])
def test_noiseless_viewer_points_lie_near_ideal_symbols(table):
    tb = simulate(table, 0, frames=3, snr_db=np.inf, viewer_depth=4096)
    viewer = tb.viewer
    assert viewer is not None
    points = viewer.points[0]
    points = points[~np.isnan(points)]
    assert len(points) >= 2048
    distance = np.abs(points[:, None] - np.asarray(table)[None, :]).min(axis=1)
    # The truncated RRC pair leaves residual ISI: about 0.0049 for BPSK and
    # 0.0043 for QPSK at the default span. Points are near, not on, the ideal.
    assert 0 < distance.max() < 0.01


@pytest.mark.parametrize("table", [bpsk(), qpsk()])
def test_viewer_does_not_change_measurements(table):
    plain = simulate(table, 2, frames=4, seed=7)
    viewed = simulate(table, 2, frames=4, seed=7, viewer_depth=512)
    assert plain.counter.errors[0] > 0
    for name in ("total", "errors", "rate"):
        np.testing.assert_array_equal(getattr(viewed.counter, name), getattr(plain.counter, name))
    np.testing.assert_array_equal(viewed.prbs.register, plain.prbs.register)
    np.testing.assert_array_equal(viewed.rx.y, plain.rx.y)
