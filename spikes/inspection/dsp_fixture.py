"""Print the reference cases that ``spikes/maxgraph/dsp.check.mjs`` compares the page's DSP against.

Run from the repository root::

    uv run --locked python -m spikes.inspection.dsp_fixture > dsp-fixture.json

Every expected value comes from NumPy, through ``views.spectrum`` and ``views.stats``
where the page mirrors them, so the two implementations cannot drift apart silently.
"""

import json

import numpy as np

from spikes.inspection import views

FFT_LENGTHS = (1, 2, 3, 8, 257, 1000, 4096)
HANN_LENGTHS = (0, 1, 2, 5, 4096)


def floats(array):
    """Return a plain list of Python floats."""
    return np.asarray(array, dtype=np.float64).tolist()


def text(array):
    """Return numbers as strings, the only JSON form that can carry NaN and Inf."""
    return [str(float(v)) for v in np.asarray(array, dtype=np.float64)]


def fft_cases(rng):
    """Return complex random inputs with ``numpy.fft.fft`` of each."""
    cases = []
    for n in FFT_LENGTHS:
        x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        out = np.fft.fft(x)
        cases.append(
            {"n": n, "re": floats(x.real), "im": floats(x.imag), "out_re": floats(out.real), "out_im": floats(out.imag)}
        )
    return cases


def hann_cases():
    """Return ``numpy.hanning`` for each tested length."""
    return [{"n": n, "w": floats(np.hanning(n))} for n in HANN_LENGTHS]


def spectrum_case(name, data, rate):
    """Return one case with the reference ``(freq, db)`` of ``views.spectrum``."""
    freq, db = views.spectrum(data, rate)
    is_complex = np.iscomplexobj(data)
    return {
        "name": name,
        "rate": rate,
        "re": floats(data.real),
        "im": floats(data.imag) if is_complex else None,
        "freq": floats(freq),
        "db": floats(db),
    }


def spectrum_cases(rng):
    """Return real and complex spectrum cases, including a non power-of-two length and a missing rate."""
    t = np.arange(4096)
    two_tones = np.sin(2 * np.pi * 5000 / 48000 * t) + 0.01 * np.sin(2 * np.pi * 12345.6 / 48000 * t)
    two_tones += 1e-3 * rng.standard_normal(4096)
    tone = np.exp(2j * np.pi * -300e3 / 2e6 * t) + 1e-3 * (rng.standard_normal(4096) + 1j * rng.standard_normal(4096))
    integers = rng.integers(-5, 6, 500).astype(np.float64) + 1j * rng.integers(-5, 6, 500).astype(np.float64)
    return [
        spectrum_case("real 4096 two tones", two_tones, 48000.0),
        spectrum_case("real 1000", rng.standard_normal(1000), 1e6),
        spectrum_case("real 256 no rate", rng.standard_normal(256), None),
        spectrum_case("complex 4096 tone", tone, 2e6),
        spectrum_case("complex 1000", rng.standard_normal(1000) + 1j * rng.standard_normal(1000), 1e6),
        spectrum_case("complex 500 integer valued", integers, 1e3),
        # Odd lengths put the negative-frequency boundary off the midpoint.
        spectrum_case("complex 101 no rate", rng.standard_normal(101) + 1j * rng.standard_normal(101), None),
    ]


def stats_case(name, data):
    """Return one case with the reference ``views.stats``; samples are text so NaN and Inf survive JSON."""
    is_complex = np.iscomplexobj(data)
    return {
        "name": name,
        "re": text(data.real),
        "im": text(data.imag) if is_complex else None,
        "expected": views.stats(data),
    }


def stats_cases(rng):
    """Return statistics cases covering finite, non-finite, all-NaN and all-zero data."""
    mixed = rng.standard_normal(40)
    mixed[[3, 11, 12, 30]] = [np.nan, np.inf, -np.inf, np.nan]
    mixed_complex = rng.standard_normal(40) + 1j * rng.standard_normal(40)
    mixed_complex[[2, 9]] = [complex(np.nan, 1.0), complex(1.0, -np.inf)]
    return [
        stats_case("real", rng.standard_normal(1000) * 3),
        stats_case("complex", rng.standard_normal(1000) + 1j * rng.standard_normal(1000)),
        stats_case("real with NaN and Inf", mixed),
        stats_case("complex with NaN and Inf", mixed_complex),
        stats_case("all NaN", np.full(16, np.nan)),
        stats_case("all zero", np.zeros(16)),
        stats_case("huge values", np.array([1e300, -2e300, 3e300])),
    ]


def main():
    """Print the fixture document."""
    rng = np.random.default_rng(7)
    document = {
        "fft": fft_cases(rng),
        "hann": hann_cases(),
        "spectrum": spectrum_cases(rng),
        "stats": stats_cases(rng),
    }
    print(json.dumps(document, allow_nan=False))


if __name__ == "__main__":
    main()
