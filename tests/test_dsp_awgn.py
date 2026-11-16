from copy import deepcopy

import numpy as np
import pytest

from graphion.dsp.blocks import Awgn
from graphion.dsp.communications import ebn0_to_snr_db, esn0_to_snr_db
from graphion.profile.numpy_frame import SetupError
from tests.allocation_helpers import check_storage_regression
from tests.dsp_helpers import context


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
def test_awgn_equivalent_energy_specifications_give_the_same_samples(real):
    outputs = []
    for snr in (float(ebn0_to_snr_db(4, 2, 8, real=real)), float(esn0_to_snr_db(4 + 10 * np.log10(2), 8, real=real))):
        block = Awgn(snr_db=snr, signal_power=1 / 8)
        block.setup(context(dtype=np.float64 if real else np.complex128))
        block.x = np.zeros((1, 64), dtype=block.ctx.dtype)
        outputs.append(block.run())
    np.testing.assert_allclose(*outputs, rtol=1e-15, atol=1e-15)


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


def test_awgn_storage():
    block = Awgn(snr_db=0, signal_power=1)
    block.setup(context(65536, 2, np.complex128))
    block.x = np.zeros((2, 65536), dtype=np.complex128)
    check_storage_regression(block.run, frame_bytes=65536, repeats=2)
