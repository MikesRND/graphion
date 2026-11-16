# DSP library

[Programming reference](index.md)

The blocks in `graphion.dsp.blocks` implement the Python/NumPy frame profile.
They import authoring types from `graphion.profile.numpy_frame`; the profile
itself has no DSP-specific rate, energy or delay rules. NumPy is the only runtime
dependency. Workspaces are allocated during setup, history is declared as State,
and reset replays a stream (including seeded noise). Outputs borrow reused storage.

## Stream contexts

`graphion.dsp.streams.stream_ctx(block, port_name)` derives a `Ctx` after the
block's local setup. Participating authors declare a plain tuple, such as
`stream_ports = ("x", "y")`. The **first entry is the reference port**: its
shape must equal `(ctx.channels, ctx.n)`. For a source this is its output.
`ctx.dtype` retains its ordinary meaning of `Sample`; literal port dtypes are
authoritative and need not equal it.

Listed ports promise continuous, non-overlapping frames of equal duration
`ctx.n / ctx.rate`. This is an author contract established by algorithm tests;
the helper cannot verify it. Every listed port must have a positive 2-D
`(channels, samples)` shape. The result takes shape and dtype from its `PortSpec`,
preserves `xp`, and scales rate by `Fraction(port_samples, ctx.n)`.
Equal sample counts leave rate unchanged; changed counts produce exact Fraction
rates from integer or Fraction inputs, and float rates from float inputs.

```python
from graphion.dsp.blocks import Prbs, SymbolMapper
from graphion.dsp.communications import qpsk
from graphion.dsp.streams import stream_ctx
from graphion.profile.numpy_frame import Ctx

source = Prbs()
source.setup(Ctx(rate=2_000_000, n=4096, channels=1))
mapper = SymbolMapper(constellation=qpsk())
mapper.setup(stream_ctx(source, "y"))  # uint8, 4096 bits
symbol_ctx = stream_ctx(mapper, "y")  # complex128, 2048 symbols, 1 MHz
```

Calling before local setup raises `SetupError`. Unlisted ports and blocks without
`stream_ports` raise the same "outside the stream-context convention" error.
Refusal also covers streams with different framing, such as overlapping windows.
The helper conveys neither latency nor sampling phase.

Gain, FrameDelay, OnePoleLowpass, Tuner, DDC and all new single-input stream
processors list `("x", "y")`. Tone and Prbs list `("y",)`. MultiChannelDDC lists
`("x", "y0", "y1", "y2")`. Spectrum lists only `("x",)`; PSD is not a time
stream. ErrorCounter lists `("x", "ref")`; its statistics are not streams.

## Block contracts

All stream shapes below have `C` independent channels; `N` describes the reference.

| Block | Configuration and behavior |
|---|---|
| `Gain` | Control `gain=1.0`; multiply each input element. No delay. |
| `FrameDelay` | Return the previous whole frame, initially zero. Split output/update supports feedback. |
| `OnePoleLowpass` | Param `pole=0.9` in [0,1); floating or complex samples; recursive single-pole filter. |
| `Prbs` | Params `order=15`, `seed=None`, `invert=False`; uint8 `(C,N)` unpacked bits, continuous across frames. |
| `SymbolMapper` | Param `constellation`; uint8 `(C,N)` bits to complex128 `(C,N/k)` symbols. MSB-first binary table index. |
| `SymbolDemapper` | Param `constellation`; complex64/128 `(C,N)` symbols to uint8 `(C,N*k)` bits. Hard nearest-point decision, ties to lowest index. |
| `FirInterpolate` | Params `taps`, `factor`; `(C,N)` to `(C,N*factor)`. Polyphase implementation of zero insertion followed by causal convolution. No implicit gain. |
| `FirDecimate` | Params `taps`, `factor`, `phase=0`; `(C,N)` to `(C,N/factor)`. Output m is filtered input sample `m*factor + phase`. |
| `Awgn` | Control `snr_db`; Params `signal_power`, `seed=0`. Nominal sample-SNR additive Gaussian noise; `+inf` copies input. |
| `SampleDelay` | Param `delay`, a nonnegative integer in samples. Zero-initialized, including delays longer than one frame; dtype-generic. |
| `ErrorCounter` | Param `skip=0`; dtype-generic `x` and `ref`. Cumulative per-channel inequality counts: uint64 `total`, `errors`, and float64 `rate`, all `(C,)`. |

### Binary sources and mapping

Prbs uses `x^n + x^q + 1` with `(n,q)` equal to `(7,6)`, `(9,5)`, `(15,14)`,
`(23,18)` or `(31,28)`. It emits the register MSB, shifts left and inserts
`bit[n-1] XOR bit[q-1]`. Each has period `2**n - 1`. The default seed is all
ones; a scalar broadcasts to all channels, or a one-dimensional array provides
one seed per channel. Seeds must be nonzero and fit in `order` bits. State uses
only `order` bits per channel. The recurrence is generated in vectorized chunks
of `q` bits, without a Python loop per bit. PRBS15 begins
`111111111111111000000000000001…`. ITU-T O.150 inverts orders 15/23/31;
set `invert=True` for that convention.

Constellation tables must be finite, distinct vectors of power-of-two length
at least two. `k = log2(len(table))`. Tables are applied as supplied; factories
perform normalization. Mapper bit frame length must be divisible by k.
Complex128 output is a mapper limitation: profile port dtypes are literal or
`ctx.dtype`, rather than expressions derived from other dtypes.

The mapper checks every bit before writing output and rejects values other than
0 or 1 (including packed bytes). As specified by the [failure contract](execution.md),
the mapper and each enclosing block through which the exception escapes then
require reset. Other blocks do not validate input values; ErrorCounter compares
generic elements, not just bits.

Demapping iterates over constellation points with reusable frame-sized metric,
best-metric, mask and index workspaces. Strict comparison preserves the first
index at ties. Memory is independent of table size. This can be slower than a
broadcast `(C,N,M)` distance array for medium constellations, but avoids that
array's potentially large memory cost.

### Filters and delay

Both FIR blocks support float32/float64/complex64/complex128 samples. Coefficients
are converted once to stream precision. Complex coefficients require a complex
stream. Factors are positive integers; decimator input length must be divisible
by its factor and `0 <= phase < factor`. Interpolation pads incomplete polyphase
branches with zeros, so tap count need not divide factor. History crosses frame
boundaries, including frames shorter than the filter memory.

A symmetric real FIR of length L has group delay `(L-1)/2` input samples at its
filtering rate. For interpolation that is the output rate. General coefficients
need not have constant group delay. A matched TX/RX pair has combined delay
`D = (len(tx) + len(rx) - 2)/2` at the oversampled rate. With integer D, choosing
`phase = D % sps` leaves `(D-phase)//sps` symbols of reference delay. The helper
`stream_ctx` does not apply this alignment automatically.

`graphion.dsp.filter_design.lowpass(ntaps, cutoff, beta=8.0)` returns unity-DC-gain
windowed-sinc coefficients (odd positive length, cutoff in input cycles/sample).
`rrc(span, sps, rolloff)` returns `span*sps + 1` float64, unit-energy
root-raised-cosine taps. Span and sps are positive integers; rolloff is in [0,1].
The zero and `abs(4*rolloff*t) == 1` singularities use analytic limits.
The defaults `(12,8,0.35)` give 97 taps and a matched cascade peak at sample 96.
With nonzero rolloff, `sps=1` aliases the intended continuous-time pulse and
loses its ideal Nyquist property. Integer `sps>=2` avoids this aliasing, although
finite tap length can still leave residual inter-symbol interference. A
conjugate-reversed discrete filter remains the matched filter in either case.

### Energy, noise and measurement

`graphion.dsp.communications.bpsk()` returns `[+1,-1]`. `qpsk()` returns
`[1+1j, 1-1j, -1+1j, -1-1j]/sqrt(2)`: the first bit selects I sign, the second Q
sign, with zero positive. Both are unit energy, and QPSK has Gray labeling.

Awgn uses nominal signal power P without per-frame estimation. For
`gamma = 10**(snr_db/10)`, real noise variance is `P/gamma`; complex noise splits
that power equally between independent I and Q. Seeds accept NumPy's generator
seed forms, including SeedSequence. Setup saves generator state and reset
restores it. The channel has no modulation-specific knowledge.

The communications helpers use dB throughout:

- `ebn0_to_snr_db(ebn0_db, bits_per_symbol, samples_per_symbol, real=False)` adds
  `10*log10(k/sps)` to Eb/N0.
- `esn0_to_snr_db(esn0_db, samples_per_symbol, real=False)` subtracts
  `10*log10(sps)` from Es/N0.
- Both add `10*log10(2)` for `real=True` (a keyword-only argument).
- `bpsk_qpsk_ber(ebn0_db)` is `0.5*erfc(sqrt(10**(ebn0_db/10)))`, implemented with
  `math.erfc`, for scalar or array inputs.

For pulse-shaped symbols, nominal channel power is
`mean(abs(table)**2) * sum(abs(tx_taps)**2) / sps`.

ErrorCounter excludes `skip` leading elements **per channel** from both totals.
After a run, its rate is NaN if no element has yet been compared. Every run
defines all three outputs, including frames entirely excluded by `skip`.
Counts use uint64 State and reset to zero; output contents are unspecified
after setup/reset until the next run. Only compared elements enter the BER denominator.
