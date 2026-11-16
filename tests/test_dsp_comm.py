import math
from copy import deepcopy

import numpy as np
import pytest

from graphion.dsp.blocks import Awgn, ErrorCounter, Prbs, SampleDelay, SymbolDemapper, SymbolMapper
from graphion.dsp.communications import bpsk, bpsk_qpsk_ber, ebn0_to_snr_db, esn0_to_snr_db, qpsk
from graphion.dsp.streams import stream_ctx
from graphion.profile.numpy_frame import Block, Ctx, SetupError
from tests.allocation_helpers import check_storage_regression


def context(n=64, channels=1, dtype=np.float64):
    return Ctx(rate=1, n=n, channels=channels, dtype=dtype)


def pn(order, n, seed=None, channels=1, invert=False):
    block = Prbs(order=order, seed=seed, invert=invert)
    block.setup(context(n, channels))
    return block


def test_prbs_prefix():
    assert "".join(str(v) for v in pn(15, 30).run()[0]) == "111111111111111000000000000001"


@pytest.mark.parametrize("order", [7, 9, 15])
def test_prbs_maximal_period(order):
    period = 2**order - 1
    block = pn(order, period)
    first = block.run().copy()
    np.testing.assert_array_equal(block.run(), first)
    windows = np.lib.stride_tricks.sliding_window_view(np.concatenate([first[0], first[0, : order - 1]]), order)
    assert len(np.unique(windows, axis=0)) == period
    assert first.sum() == 2 ** (order - 1)


@pytest.mark.parametrize("order", [7, 9, 15, 23, 31])
@pytest.mark.parametrize("n", [1, 5, 64, 4096])
def test_prbs_continuity_and_replay(order, n):
    full = pn(order, n * 4, seed=[1, 19], channels=2)
    chunked = pn(order, n, seed=[1, 19], channels=2)
    expected = full.run().copy()
    actual = np.concatenate([chunked.run().copy() for _ in range(4)], axis=1)
    np.testing.assert_array_equal(actual, expected)
    # Independent scalar register recurrence establishes orientation for arbitrary seeds.
    q = {7: 6, 9: 5, 15: 14, 23: 18, 31: 28}[order]
    reg = 19
    reference = []
    for _ in range(n * 4):
        reference.append(reg >> (order - 1))
        reg = ((reg << 1) | (((reg >> (order - 1)) ^ (reg >> (q - 1))) & 1)) & ((1 << order) - 1)
    np.testing.assert_array_equal(actual[1], reference)
    chunked.reset()
    np.testing.assert_array_equal(chunked.run(), expected[:, :n])
    np.testing.assert_array_equal(pn(order, n * 4, seed=[1, 19], channels=2, invert=True).run(), expected ^ 1)
    broadcast = pn(order, n, seed=19, channels=2).run()
    np.testing.assert_array_equal(broadcast[0], broadcast[1])


@pytest.mark.parametrize(
    "kwargs",
    [{"seed": 0}, {"seed": -1}, {"seed": 32768}, {"seed": [1, 2]}, {"seed": 1.5}, {"order": 8}, {"order": 15.0}],
)
def test_prbs_invalid(kwargs):
    with pytest.raises(SetupError):
        Prbs(**kwargs).setup(context())


def tables():
    return [bpsk(), qpsk(), np.array([i + 1j * q for i in (-3, -1, 1, 3) for q in (-3, -1, 1, 3)])]


@pytest.mark.parametrize("table", tables())
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_mapping_labels_roundtrip(table, dtype):
    k = len(table).bit_length() - 1
    bits = ((np.arange(len(table))[:, None] >> np.arange(k - 1, -1, -1)) & 1).astype(np.uint8).reshape(1, -1)
    mapper = SymbolMapper(constellation=table)
    mapper.setup(context(bits.size))
    mapper.x = bits
    np.testing.assert_array_equal(mapper.run()[0], table)
    demapper = SymbolDemapper(constellation=table)
    demapper.setup(stream_ctx(mapper, "y").replace(dtype=dtype))
    demapper.x = mapper.y.astype(dtype)
    np.testing.assert_array_equal(demapper.run(), bits)
    demapper.reset()
    np.testing.assert_array_equal(demapper.run(), bits)


def test_factory_labels_and_energy():
    np.testing.assert_array_equal(bpsk(), [1, -1])
    np.testing.assert_allclose(qpsk() * np.sqrt(2), [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j])
    for table in (bpsk(), qpsk()):
        np.testing.assert_allclose(np.abs(table) ** 2, 1)


@pytest.mark.parametrize("table", [[], [1], [1, 2, 3], [1, 1], [1, np.inf], [[1, -1]]])
@pytest.mark.parametrize("cls", [SymbolMapper, SymbolDemapper])
def test_bad_constellation(cls, table):
    with pytest.raises(SetupError):
        cls(constellation=table).setup(context(dtype=np.complex128))


def test_mapper_frame_divisibility_and_demapper_dtype():
    with pytest.raises(SetupError):
        SymbolMapper(constellation=qpsk()).setup(context(3))
    with pytest.raises(SetupError):
        SymbolDemapper(constellation=qpsk()).setup(context())


@pytest.mark.parametrize("value", [2, 128, 255])
def test_invalid_bits_leave_output_unwritten_and_require_reset(value):
    class Composite(Block):
        def build(self):
            self.mapper = self.child("mapper", SymbolMapper(constellation=qpsk()))

        def process(self):
            self.mapper.run()

    root = Composite()
    root.setup(context(4))
    output = root.mapper.y
    root.mapper.x = np.zeros((1, 4), dtype=np.uint8)
    root.run()
    before = output.copy()
    root.mapper.x = np.array([[0, 1, 1, value]], dtype=np.uint8)
    with pytest.raises(ValueError, match="unpacked bits"):
        root.run()
    np.testing.assert_array_equal(output, before)
    for block in (root, root.mapper):
        with pytest.raises(Exception, match="reset"):
            block.run()
    root.reset()
    root.mapper.x = np.ones((1, 4), dtype=np.uint8)
    root.run()
    np.testing.assert_array_equal(root.mapper.y, [[qpsk()[3], qpsk()[3]]])


def test_demapper_ties():
    block = SymbolDemapper(constellation=np.array([1, -1, 1j, -1j]))
    block.setup(context(3, dtype=np.complex128))
    block.x = np.array([[0, 0.5 + 0.5j, -0.5 - 0.5j]])
    np.testing.assert_array_equal(block.run(), [[0, 0, 0, 0, 0, 1]])


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_awgn_statistics_and_replay(dtype):
    block = Awgn(snr_db=4, signal_power=2, seed=16)
    block.setup(context(200_000, 2, dtype))
    block.x = np.zeros((2, 200_000), dtype=dtype)
    first = block.run().copy()
    variance = 2 / 10**0.4
    assert np.abs(first.mean()) < 0.005
    assert np.mean(np.abs(first) ** 2) == pytest.approx(variance, rel=0.01)
    if np.iscomplexobj(first):
        assert first.real.var() == pytest.approx(variance / 2, rel=0.01)
        assert first.imag.var() == pytest.approx(variance / 2, rel=0.01)
        assert abs(np.corrcoef(first.real.ravel(), first.imag.ravel())[0, 1]) < 0.01
    block.run()
    block.reset()
    np.testing.assert_array_equal(block.run(), first)
    same = Awgn(snr_db=4, signal_power=2, seed=16)
    same.setup(block.ctx)
    same.x = block.x
    np.testing.assert_array_equal(same.run(), first)
    block.snr_db = np.inf
    np.testing.assert_array_equal(block.run(), block.x)


@pytest.mark.parametrize("bit_generator", [np.random.PCG64, np.random.MT19937, np.random.Philox])
@pytest.mark.parametrize("seed_form", ["generator", "bit_generator"])
def test_awgn_owns_rng_and_cannot_rewind_caller_or_sibling(bit_generator, seed_form):
    caller = np.random.Generator(bit_generator(71))
    caller.standard_normal(17)  # Clone the current state, not merely the original seed.
    original = deepcopy(caller)
    seed = caller if seed_form == "generator" else caller.bit_generator
    first = Awgn(snr_db=0, signal_power=1, seed=seed)
    second = Awgn(snr_db=0, signal_power=1, seed=seed)
    for block in (first, second):
        block.setup(context())
        block.x = np.zeros((1, 64))
        assert block._rng is not caller
        assert block._rng.bit_generator is not caller.bit_generator
    assert first._rng.bit_generator is not second._rng.bit_generator
    expected_first = original.standard_normal((1, 64))
    expected_second = original.standard_normal((1, 64))
    np.testing.assert_array_equal(first.run(), expected_first)
    np.testing.assert_array_equal(second.run(), expected_first)
    first.reset()
    np.testing.assert_array_equal(first.run(), expected_first)
    np.testing.assert_array_equal(second.run(), expected_second)
    # Setup, processing and reset must all leave the external RNG untouched.
    untouched = np.random.Generator(bit_generator(71))
    untouched.standard_normal(17)
    np.testing.assert_array_equal(caller.standard_normal(64), untouched.standard_normal(64))
    # Later caller draws and resets must not affect either block's private stream.
    caller.bit_generator.state = bit_generator(999).state
    np.testing.assert_array_equal(first.run(), expected_second)


def test_awgn_spawned_seed_sequences_give_separate_replayable_streams():
    blocks = [Awgn(snr_db=0, signal_power=1, seed=seed) for seed in np.random.SeedSequence(71).spawn(2)]
    outputs = []
    for block in blocks:
        block.setup(context())
        block.x = np.zeros((1, 64))
        outputs.append(block.run().copy())
        block.reset()
        np.testing.assert_array_equal(block.run(), outputs[-1])
    assert not np.array_equal(*outputs)


@pytest.mark.parametrize("real", [False, True])
def test_snr_conversions(real):
    db = np.arange(9.0)
    for k in (1, 2, 4):
        eb = ebn0_to_snr_db(db, k, 8, real=real)
        es = esn0_to_snr_db(db + 10 * np.log10(k), 8, real=real)
        np.testing.assert_allclose(eb, es, atol=2e-15)
        np.testing.assert_allclose(10 ** (eb / 10), 10 ** (db / 10) * k / 8 * (2 if real else 1))
    # Identical seeds and equivalent energy specifications produce the same samples.
    outputs = []
    for snr in (float(ebn0_to_snr_db(4, 2, 8, real=real)), float(esn0_to_snr_db(4 + 10 * np.log10(2), 8, real=real))):
        block = Awgn(snr_db=snr, signal_power=1 / 8)
        block.setup(context(dtype=np.float64 if real else np.complex128))
        block.x = np.zeros((1, 64), dtype=block.ctx.dtype)
        outputs.append(block.run())
    np.testing.assert_allclose(*outputs, rtol=1e-15, atol=1e-15)
    assert bpsk_qpsk_ber(0) == pytest.approx(0.5 * math.erfc(1))
    np.testing.assert_allclose(bpsk_qpsk_ber([0, 4, 8]), [0.0786496035, 0.0125008180, 0.0001909078], rtol=1e-6)


@pytest.mark.parametrize("delay", [0, 1, 4, 7, 21])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32, np.complex128])
def test_sample_delay(delay, dtype):
    x = np.arange(56).reshape(2, 28).astype(dtype)
    block = SampleDelay(delay=delay)
    block.setup(context(7, 2, dtype))
    frames = []
    for row in np.split(x, 4, axis=1):
        block.x = row
        frames.append(block.run().copy())
    expected = np.pad(x, ((0, 0), (delay, 0)))[:, :28]
    np.testing.assert_array_equal(np.concatenate(frames, axis=1), expected)
    block.reset()
    block.x = x[:, :7]
    np.testing.assert_array_equal(block.run(), expected[:, :7])


@pytest.mark.parametrize("skip", [0, 3, 9, 25])
def test_counter(skip):
    counter = ErrorCounter(skip=skip)
    counter.setup(context(5, 2, np.uint8))
    x = np.zeros((2, 20), dtype=np.uint8)
    ref = x.copy()
    ref[0, ::3] = 255  # generic inequality, not just bits
    ref[1, ::2] = 1
    for start in range(0, 20, 5):
        counter.x = x[:, start : start + 5]
        counter.ref = ref[:, start : start + 5]
        counter.run()
    np.testing.assert_array_equal(counter.errors, np.count_nonzero(ref[:, skip:], axis=1))
    np.testing.assert_array_equal(counter.total, [max(0, 20 - skip)] * 2)
    assert counter.total.dtype == counter.errors.dtype == np.uint64
    if skip < 20:
        np.testing.assert_allclose(counter.rate, np.count_nonzero(ref[:, skip:], axis=1) / (20 - skip))
    else:
        assert np.isnan(counter.rate).all()
    expected = tuple(value.copy() for value in (counter.total, counter.errors, counter.rate))
    counter.reset()
    for start in range(0, 20, 5):
        counter.x = x[:, start : start + 5]
        counter.ref = ref[:, start : start + 5]
        counter.run()
    for actual, replay in zip((counter.total, counter.errors, counter.rate), expected):
        np.testing.assert_array_equal(actual, replay)


def test_counter_reset_preserves_output_storage_and_skip_run_defines_outputs():
    counter = ErrorCounter(skip=10)
    supplied = {
        "total": np.full(2, 123, dtype=np.uint64),
        "errors": np.full(2, 123, dtype=np.uint64),
        "rate": np.full(2, 123.0),
    }
    for name, values in supplied.items():
        counter.bind_output(name, values)
    counter.setup(context(5, 2, np.uint8))
    counter.x = np.zeros((2, 5), dtype=np.uint8)
    counter.ref = counter.x
    for _ in range(2):
        counter.reset()
        # Inspect caller-owned storage to enforce the no-output-writes reset rule.
        for values in supplied.values():
            np.testing.assert_array_equal(values, [123, 123])
        for _ in range(2):
            counter.run()
            np.testing.assert_array_equal(counter.total, [0, 0])
            np.testing.assert_array_equal(counter.errors, [0, 0])
            assert np.isnan(counter.rate).all()
            for values in supplied.values():
                values.fill(123)


@pytest.mark.parametrize("snr", [np.nan, -np.inf])
def test_awgn_invalid_initial_and_updated_control(snr):
    with pytest.raises(SetupError, match="snr_db"):
        Awgn(snr_db=snr, signal_power=1).setup(context())
    block = Awgn(snr_db=0, signal_power=1)
    block.setup(context())
    block.x = np.zeros((1, 64))
    block.snr_db = snr
    with pytest.raises(ValueError, match="snr_db"):
        block.run()


@pytest.mark.parametrize(
    "block",
    [
        Prbs(),
        SymbolMapper(constellation=qpsk()),
        SymbolDemapper(constellation=qpsk()),
        Awgn(snr_db=0, signal_power=1),
        SampleDelay(delay=100),
        ErrorCounter(skip=100),
    ],
)
def test_comm_storage(block):
    dtype = np.uint8 if isinstance(block, (SymbolMapper, ErrorCounter, SampleDelay)) else np.complex128
    block.setup(context(65536, 2, dtype))
    if not isinstance(block, Prbs):
        block.x = np.zeros((2, 65536), dtype=dtype)
    if isinstance(block, ErrorCounter):
        block.ref = block.x
    check_storage_regression(block.run, frame_bytes=65536, repeats=2)
