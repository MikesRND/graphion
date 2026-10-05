# Three-channel digital down converter

This example converts a real 1.024 MHz ADC stream into three complex 64 kHz
baseband streams. Each DDC mixes its band to zero, then filters and decimates
by 2, 2 and 4. The wanted offsets are +5, −8 and +12 kHz. An interferer is
20 dB stronger than the wanted tones, and the input has a seeded noise floor.

Set up the development environment as described in the root
[README](../../README.md); it includes Matplotlib. From the repository root:

```bash
uv run python -m examples.ddc.ddc_tb                 # live window, DDC 0 sweeps
uv run python -m examples.ddc.ddc_tb --save ddc.png   # headless snapshot and PASS/FAIL
```

Shared processing blocks live in [examples/blocks](../blocks/README.md); the figure and animation
use the [example plotting library](../plotting/README.md).

The live window updates at 20 fps and sweeps DDC 0's tuner ±25 kHz in 30
seconds. Its live spectrum uses no averaging to keep the moving peak sharp.
The wanted tone moves in its baseband plot and fades when it leaves the
passband. `--save` uses Agg, holds the tuners fixed and averages 40 frames.
All spectra use a Hann window and two-sided PSD in dB/Hz, with 250 Hz FFT bin
spacing. The real-input panel displays positive frequencies from 0 to 512 kHz.
The complex baseband panels display both positive and negative frequencies.

```text
Testbench (no ports)
├── stim: Stimulus ─────┬──> dut.x
│                      └──> scope_in: Spectrum
├── dut: MultiChannelDDC
│   x ─┬──> ddc0: DDC ──> y0 ──> scope0: Spectrum ──> check.psd0
│      ├──> ddc1: DDC ──> y1 ──> scope1: Spectrum ──> check.psd1
│      └──> ddc2: DDC ──> y2 ──> scope2: Spectrum ──> check.psd2
└── check: PeakCheck ──> offset_hz

Each DDC: x → tune → hb1↓2 → hb2↓2 → fir↓4 → y
```

Read the files from the leaves upward:

1. [tuner.py](../blocks/tuner.py): the mutable frequency Control and phase State.
2. [fir_decimate.py](../blocks/fir_decimate.py): unity-gain lowpass taps and streaming FIR history.
3. [ddc.py](../blocks/ddc.py): one tune/filter path and explicit child contexts.
4. [multichannel_ddc.py](../blocks/multichannel_ddc.py): shared input and three forwarded outputs.
5. [spectrum.py](../blocks/spectrum.py): a windowed PSD monitor with exponential averaging.
6. [ddc_tb.py](ddc_tb.py): stimulus, peak measurement, execution order and host plotting.
7. [live.py](../plotting/live.py): generic plot panels, line updates and animation.

Each DDC's input context stays at the ADC rate. Its `output_ctx` describes the
decimated complex output, and supplies the downstream monitors' contexts.
Starting with a `Fraction` rate preserves exact arithmetic through every stage.
The 23-tap halfbands protect the wanted band; the 101-tap final FIR has a
−3 dB edge near ±25.6 kHz and a stopband beginning at ±38.4 kHz.

Fan-out shares producer storage. Every owner runs producers before consumers;
the testbench calls stimulus, DUT, spectra and check in that order. Plotting
reads the most recent PSD buffers after a run, before the next frame overwrites
them. Numerical expressions create temporary arrays for readability.

The shared blocks use only Graphion/NumPy and other example blocks. Matplotlib
is imported when creating a plot, so tests and headless DSP use need no plotting library.
The tests cover interfaces, exact rational rates, processing across frame
boundaries, passband response and interference rejection without input noise.
