import math

import numpy as np
import pytest

from graphion.dsp.communications import bpsk, bpsk_qpsk_ber, ebn0_to_snr_db, esn0_to_snr_db, qpsk


def test_factory_labels_and_energy():
    np.testing.assert_array_equal(bpsk(), [1, -1])
    np.testing.assert_allclose(qpsk() * np.sqrt(2), [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j])
    for table in (bpsk(), qpsk()):
        np.testing.assert_allclose(np.abs(table) ** 2, 1)


@pytest.mark.parametrize("real", [False, True])
def test_snr_conversions(real):
    db = np.arange(9.0)
    for k in (1, 2, 4):
        eb = ebn0_to_snr_db(db, k, 8, real=real)
        es = esn0_to_snr_db(db + 10 * np.log10(k), 8, real=real)
        np.testing.assert_allclose(eb, es, atol=2e-15)
        np.testing.assert_allclose(10 ** (eb / 10), 10 ** (db / 10) * k / 8 * (2 if real else 1))


def test_bpsk_qpsk_ber():
    assert bpsk_qpsk_ber(0) == pytest.approx(0.5 * math.erfc(1))
    np.testing.assert_allclose(bpsk_qpsk_ber([0, 4, 8]), [0.0786496035, 0.0125008180, 0.0001909078], rtol=1e-6)
