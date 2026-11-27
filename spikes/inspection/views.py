"""Display-sized payloads computed from raw arrays; nothing here knows about blocks."""

import enum
import math

import numpy as np


def numbers(array, digits=4):
    """Round to a JSON-friendly list of floats."""
    return np.round(np.asarray(array, dtype=np.float64), digits).tolist()


def stats(array):
    """Peak, RMS and non-finite count of an array's magnitudes."""
    magnitude = np.abs(np.asarray(array)).astype(np.float64)
    finite = np.isfinite(magnitude)
    if not finite.any():
        return {"nonfinite": int(magnitude.size), "peak": None, "rms": None}
    peak = float(magnitude[finite].max())
    # Scale by the peak so squaring very large values cannot overflow to infinity.
    rms = peak * float(np.sqrt(np.mean((magnitude[finite] / peak) ** 2))) if peak else 0.0
    return {"nonfinite": int(magnitude.size - np.count_nonzero(finite)), "peak": peak, "rms": rms}


def spectrum(data, rate=None):
    """Return (freq, db) for one channel: the reference the page's own transform is checked against.

    Hann window (NumPy's symmetric ``hanning``), power normalised by the
    window's energy, floor at -200 dB. Complex input gives a shifted two-sided
    spectrum, real input the one-sided one. Without a rate, frequency is in
    cycles per sample.
    """
    data = np.asarray(data).reshape(-1)
    if data.dtype.kind not in "fc":
        data = data.astype(np.float64)
    window = np.hanning(data.size)
    step = 1.0 / (rate or 1.0)
    if np.iscomplexobj(data):
        bins = np.fft.fftshift(np.fft.fft(data * window))
        freq = np.fft.fftshift(np.fft.fftfreq(data.size, step))
    else:
        bins = np.fft.rfft(data * window)
        freq = np.fft.rfftfreq(data.size, step)
    power = np.abs(bins) ** 2 / np.sum(window**2)
    return freq, 10 * np.log10(np.maximum(power, 1e-20))


def describe_frame(frame):
    """Return the facts every probe payload carries about its source array."""
    return {"dtype": frame.dtype.name, "shape": list(frame.shape)}


def reference(data, view, rate=None):
    """Reduce one channel at full resolution in NumPy.

    The page does its own reductions; this is what its numbers are checked
    against, and what a client too thin to transform anything could ask for.
    """
    data = np.asarray(data).reshape(-1)
    out = {"view": view, "count": int(data.size), "stats": stats(data)}
    if view == "spectrum":
        freq, db = spectrum(data, rate)
        out.update(freq=freq.tolist(), db=finite_list(db))
    elif np.iscomplexobj(data):
        out.update(re=finite_list(data.real), im=finite_list(data.imag))
    else:
        out["y"] = finite_list(data)
    return out


def finite_list(array):
    """Return a list of floats with None where JSON has no spelling for the value."""
    values = np.asarray(array, dtype=np.float64)
    return [value if math.isfinite(value) else None for value in values.tolist()]


def summarize(value):
    """Summarize a Control or State value for the inspector."""
    if isinstance(value, np.ndarray):
        head = value.reshape(-1)[:8]
        text = ", ".join(f"{v:.4g}" for v in (head if not np.iscomplexobj(head) else np.abs(head)))
        return {"kind": "array", "dtype": value.dtype.name, "shape": list(value.shape), "head": text, **stats(value)}
    if isinstance(value, enum.Enum):
        return {"kind": "scalar", "value": f"{type(value).__name__}.{value.name}"}
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, complex):
        # JSON has no complex number; the text form keeps both parts readable.
        return {"kind": "scalar", "value": f"{value.real:.6g}{value.imag:+.6g}j"}
    if isinstance(value, float) and not math.isfinite(value):
        # JSON has no NaN or infinity either, and a strict encoder raises on them.
        return {"kind": "scalar", "value": str(value)}
    if isinstance(value, (bool, int, float, str)) or value is None:
        return {"kind": "scalar", "value": value}
    return {"kind": "opaque", "type": type(value).__name__}


def summarize_all(members):
    """Summarize a mapping of names to Control or State values."""
    return {name: summarize(value) for name, value in members.items()}


def value_payload(value):
    """Return a configuration array in full, with a magnitude response when it looks like taps."""
    value = np.asarray(value)
    if value.dtype.kind not in "biufc" or value.size > 65536:
        return {"error": "not a displayable numeric array"}
    flat = value.reshape(-1)
    out = describe_frame(value)
    if np.iscomplexobj(flat):
        out.update(re=numbers(flat.real, 6), im=numbers(flat.imag, 6))
    else:
        out["y"] = numbers(flat, 6)
    if value.ndim == 1 and flat.size >= 8 and value.dtype.kind in "fc":
        response = np.abs(np.fft.fft(flat, 1024))
        if not np.iscomplexobj(flat):
            response = response[:513]
        peak = float(response.max()) or 1.0
        out["response_db"] = numbers(20 * np.log10(np.maximum(response / peak, 1e-8)), 2)
    return out
