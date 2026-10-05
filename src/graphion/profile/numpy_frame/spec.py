"""Resolved storage interface of a locally configured port."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PortSpec:
    shape: tuple[int, ...]
    dtype: np.dtype
