"""Immutable configuration for the Python/NumPy frame Implementation Profile."""

import math
from dataclasses import dataclass, replace
from fractions import Fraction
from numbers import Integral

import numpy as np

from graphion.errors import SetupError


@dataclass(frozen=True)
class Ctx:
    rate: int | float | Fraction | np.integer | np.floating
    n: int
    channels: int
    dtype: object = np.float64
    xp: object = np

    def __post_init__(self):
        if isinstance(self.rate, (bool, np.bool_)):
            raise SetupError("rate must be an int, float, Fraction or NumPy integer/floating scalar")
        if isinstance(self.rate, (int, np.integer)):
            rate = int(self.rate)
        elif isinstance(self.rate, Fraction):
            rate = Fraction(self.rate.numerator, self.rate.denominator)
        elif isinstance(self.rate, (float, np.floating)):
            rate = float(self.rate)
        else:
            raise SetupError("rate must be an int, float, Fraction or NumPy integer/floating scalar")
        try:
            approximate_rate = float(rate)
        except (TypeError, ValueError, OverflowError) as exc:
            raise SetupError("rate must be representable as a positive finite Python float") from exc
        if not math.isfinite(approximate_rate) or approximate_rate <= 0:
            raise SetupError("rate must be representable as a positive finite Python float")
        object.__setattr__(self, "rate", rate)
        for name in ("n", "channels"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise SetupError(f"{name} must be a positive integer")
        try:
            dtype = np.dtype(self.dtype)
        except (TypeError, ValueError, OverflowError) as exc:
            raise SetupError("unsupported sample dtype") from exc
        if dtype.kind not in "biufc":
            raise SetupError("sample dtype must be numeric")
        object.__setattr__(self, "dtype", dtype)

    def replace(self, **changes):
        return replace(self, **changes)
