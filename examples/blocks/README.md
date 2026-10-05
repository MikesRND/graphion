# Example block library

These reusable blocks use Graphion and NumPy. They have no plotting imports
and are shared by the [runnable examples](../README.md).

```python
from examples.blocks import Tone, Spectrum, Tuner, FirDecimate, DDC, MultiChannelDDC, lowpass
```

| File | Purpose |
|---|---|
| [tone.py](tone.py) | Real sine-wave source with amplitude/frequency Controls and phase State. |
| [tuner.py](tuner.py) | Phase-continuous mixing to complex samples. |
| [fir_decimate.py](fir_decimate.py) | Windowed-sinc coefficient design and streaming FIR decimation. |
| [spectrum.py](spectrum.py) | Two-sided PSD with a Hann window and configurable exponential averaging. |
| [ddc.py](ddc.py) | One mixer and three filter/decimator stages with an explicit output context. |
| [multichannel_ddc.py](multichannel_ddc.py) | Three DDC branches sharing an input buffer. |

The DDC coefficients are designed for a 1.024 MHz ADC rate and decimation by
16, with a −3 dB edge near ±25.6 kHz. Their normalized frequency response
scales with the input rate. `output_ctx` describes the decimated stream;
starting with a Fraction rate preserves exact arithmetic through every stage.

Blocks follow Graphion's ordinary setup/run/reset and storage rules. Owners
call children in signal-flow order. Spectra and other outputs are reusable
buffers; copy a result to retain it across later calls.
