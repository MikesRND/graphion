"""Display-sized payloads computed from raw arrays; nothing here knows about blocks."""

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


def pool_max(values, limit):
    """Decimate for display by keeping the maximum of each bucket."""
    if values.size <= limit:
        return values
    size = int(np.ceil(values.size / limit))
    padded = np.full(size * int(np.ceil(values.size / size)), -np.inf)
    padded[: values.size] = values
    return padded.reshape(-1, size).max(axis=1)


def describe_frame(frame):
    """Return the facts every probe payload carries about its source array."""
    return {"dtype": frame.dtype.name, "shape": list(frame.shape)}


def reduce(frame, view, limit, rate):
    """Reduce one frame to the requested view, falling back to a time view."""
    frame = np.asarray(frame)
    data = frame[0] if frame.ndim == 2 else frame.reshape(-1)
    out = {"view": view, **describe_frame(frame), "stats": stats(frame)}
    is_complex = np.iscomplexobj(data)
    if view == "constellation" and is_complex:
        tail = data[-limit:]
        tail = tail[np.isfinite(tail)]
        out.update(re=numbers(tail.real), im=numbers(tail.imag))
    elif view == "spectrum" and data.size >= 8 and data.dtype.kind in "fc":
        window = np.hanning(data.size)
        spectrum = np.fft.fft(data * window) if is_complex else np.fft.rfft(data * window)
        if is_complex:
            spectrum = np.fft.fftshift(spectrum)
        power = np.abs(spectrum) ** 2 / np.sum(window**2)
        db = 10 * np.log10(np.maximum(pool_max(power, limit), 1e-20))
        half = (rate or 1.0) / 2
        out.update(db=numbers(db, 2), f0=-half if is_complex else 0.0, f1=half, has_rate=rate is not None)
    elif view == "array" and not is_complex:
        # A whole-frame view for outputs that are already a display, such as a PSD.
        out.update(y=numbers(pool_max(data.astype(np.float64), limit)))
    else:
        head = data[:limit]
        out["view"] = "time"
        if is_complex:
            out.update(re=numbers(head.real), im=numbers(head.imag))
        else:
            out.update(y=numbers(head))
    return out


def summarize(value):
    """Summarize a Control or State value for the inspector."""
    if isinstance(value, np.ndarray):
        head = value.reshape(-1)[:8]
        text = ", ".join(f"{v:.4g}" for v in (head if not np.iscomplexobj(head) else np.abs(head)))
        return {"kind": "array", "dtype": value.dtype.name, "shape": list(value.shape), "head": text, **stats(value)}
    if isinstance(value, (bool, int, float, str)) or value is None:
        return {"kind": "scalar", "value": value}
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return {"kind": "scalar", "value": value.item()}
    return {"kind": "opaque", "type": type(value).__name__}


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
