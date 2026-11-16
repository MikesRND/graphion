"""Compatibility imports for the promoted DSP library filter."""

from graphion.dsp.blocks import FirDecimate
from graphion.dsp.filter_design import lowpass

__all__ = ["FirDecimate", "lowpass"]
