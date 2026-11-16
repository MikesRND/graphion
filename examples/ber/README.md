# BPSK and QPSK BER in AWGN

From the repository root, after [developer setup](../../README.md):

```bash
uv run python -m examples.ber.ber_tb --save ber.png --csv ber.csv
uv run python -m examples.ber.ber_tb
uv run python -m examples.ber.ber_tb --diagram
uv run python -m examples.ber.ber_tb --start 0 --stop 6 --step 2 --error-target 500 --bit-cap 2000000 --frame-size 2048 --seed 42 --rolloff 0.35 --span 12 --sps 8 --save quick.png
```

`--save` uses a dedicated Matplotlib Agg canvas without a GUI or a global backend change. The default sweep covers 0–8 dB in
1 dB steps for both modulations. It prints compared bits, errors, measured BER,
ideal AWGN theory, finite-filter prediction, prediction/theory ratio, deviation
in predicted standard errors, status and 95% intervals.
CSV contains the same measurements. Plotting imports are deferred; the simulation
itself requires only NumPy and Graphion.

`--diagram` opens a separate model window while the sweep runs. It shows the
actual testbench's child blocks, recorded port connections, configuration and
resolved output shapes/dtypes, obtained from `describe()`. The window updates
for each modulation/SNR point and shows frame progress; it stays available
alongside the BER plot when the sweep finishes. Closing it early does not stop
the simulation or reopen it at the next point. This requires a GUI Matplotlib
backend and is mutually exclusive with headless `--save`.

The renderer in [diagram.py](diagram.py) uses a layout specific to this testbench,
including its optional constellation viewer.
It is a model topology view, not an execution trace or an inferred schedule.
It never advances the model. For a programmatic headless diagram, construct
`BerDiagram(headless=True)`, call `update(tb, "QPSK", frame, frames)` on a
configured testbench, then `save("model.png")`.

```text
Prbs → SymbolMapper → FirInterpolate ×8 → Awgn → FirDecimate ÷8 → SymbolDemapper → ErrorCounter.x
  └──────────────────────────── SampleDelay ─────────────────────────────────→ ErrorCounter.ref
```

[ber_tb.py](ber_tb.py) builds a fresh testbench per modulation/SNR point with
explicit connections and run order. It uses the reusable
[DSP library](../../docs/reference/dsp-library.md), deriving child contexts with
`stream_ctx`. The root is 4096 bits, one channel, at `k * 1 MHz` as a Fraction,
where k is bits per symbol. PRBS15 is non-inverted with an all-ones seed.

BPSK maps 0 to +1 and 1 to −1. QPSK maps 00, 01, 10, 11 to
`[1+1j, 1-1j, -1+1j, -1-1j]/sqrt(2)`. Thus the MSB selects I sign, the LSB selects
Q sign, and both modulations have unit symbol energy.

TX uses a unit-energy, 12-symbol-span root-raised-cosine filter, rolloff 0.35 and
8 samples/symbol: 97 taps. RX uses conjugate-reversed TX taps. Nominal channel
power is `mean(abs(table)**2) * sum(abs(tx)**2) / sps`, or 1/8 for the defaults.
Eb/N0 converts to sample SNR by adding `10*log10(k/sps)`; complex noise shares its
total variance equally across I and Q.

## Live constellations

```bash
uv run python -m examples.ber.live
uv run python -m examples.ber.live --diagram
uv run python -m examples.ber.live --esn0-db 6 --ticks 60 --save constellations.png
```

[live.py](live.py) runs the same `BerTestbench` continuously and shows the
received symbols: one BPSK and one QPSK model, side by side, with the ideal
points marked. `BerTestbench(viewer_depth=...)` adds a
[ConstellationViewer](../../docs/reference/dsp-library.md#block-contracts) on
the matched-filter output, the samples the demapper decides on. It skips the
same startup symbols the error counter excludes. The default `viewer_depth=0`
builds no viewer, so the BER sweep above is unchanged and pays nothing for it.

Both models run at one `--esn0-db` (default 10), so their clouds have the same
spread. Each model receives `Eb/N0 = Es/N0 - 10*log10(k)`: QPSK runs 3.01 dB
below BPSK in Eb/N0. Noise settings stay fixed for the whole run. Panel titles
show each Eb/N0 and the running measured BER.

Each tick advances both models by one frame of `--frame-size` bits and redraws
the most recent `--depth` symbols (default 1024). The models advance exactly
`--ticks` times (default 300); nothing runs before the first tick, and the
panels stay empty until then because they read the viewer's retained history. The run then stops with the final constellations
on screen. Closing the window ends the run early, and the runner prints how
many ticks completed. `--save` performs the same ticks without a GUI and
writes the final frame.

`--diagram` also opens the model window described above, drawn by the same
[diagram.py](diagram.py), for one of the two testbenches: `--diagram bpsk` or
`--diagram qpsk` (QPSK if not named). It shows the viewer beside the `rx`
block it taps and follows the tick count, refreshing a few times a second.
Closing the diagram leaves the run going; closing the constellation window
ends the run and closes the diagram too. It needs a GUI and cannot be
combined with `--save`.

## Alignment and startup

RX is the conjugate reverse of TX, so the combined delay is
`D = len(tx) - 1` oversampled samples, always an integer. Decimation phase is
`p = D % sps`; reference delay is `((D-p)//sps)*k` bits. The default is 96 samples,
phase 0 and `12*k` bits. Prepending a zero to TX taps gives D=97, phase 1, still
`12*k` bits. This is covered by a noiseless regression test.

Startup exclusion is twice the reference delay (`24*k` default). This only
defines an exclusion window; no BER claim is made during startup. Excluded bits
never enter the denominator. SampleDelay and all filter histories reset to zero.

## Statistical agreement

Each point's frame-rounded budget is fixed **before** the run, targeting 1,000
expected errors under the finite-filter prediction with at most 10 million
compared bits. Rounding down at the cap
keeps it a strict cap. Noise seeds use
`SeedSequence([seed, modulation_index, point_index])`. There is no early stopping
based on observed errors.

[prediction.py](prediction.py) predicts BER only for the factory BPSK/QPSK
mappings used by this sweep, whose hard decisions are per-axis signs. It forms
the TX/matched-RX convolution from the actual taps, samples at the chosen phase,
and computes steady-state decision values over a complete PRBS15 symbol period.
The 32,767-symbol period consumes one bit period for BPSK and two for QPSK.
The decimator's output at index m is compared with prediction index
`(m - symbol_delay) % 32767` after startup. Main-tap gain and residual interference
are included, without assuming unit-energy taps.

If channel sample SNR is gamma, the matched-filter noise variance per axis is
`signal_power * sum(abs(rx_taps)**2) / (2*gamma)`. Given signed decision margin a,
the conditional bit-error probability is `0.5*erfc(a/sqrt(2*variance))`. Averaging
these probabilities gives the prediction. Noiseless regression tests compare
the actual decimator samples to the predicted decision values, including scaled
taps, nonzero phase, complex coefficients, short filters and one sample/symbol.
An independent scalar PRBS/convolution calculation cross-checks the predictor.

The full-period average approximates the exact transmitted subsequence; default
low-SNR runs are shorter than one period. Their difference is small relative to
the five-standard-error band. The band uses a binomial standard-error estimate;
it is a diagnostic rather than an exact confidence guarantee for correlated errors.
For `N*p_prediction <= 25`, excess errors fail when the Poisson upper tail is below the one-sided 5σ threshold.

The demo and regression tests share `agreement`:

- `sigma = sqrt(p_prediction * (1-p_prediction) / N)`.
- `PASS` when `abs(p_measured-p_prediction) <= 5*sigma`; otherwise `FAIL`.
- `INCONCLUSIVE` when `N*p_prediction <= 25` unless the Poisson upper-tail check fails.

An observed error with a zero prediction is always `FAIL`, checked before the
inconclusive gate. A zero prediction with zero observed errors remains
`INCONCLUSIVE`; floating-point underflow alone does not establish noiselessness.

Ideal AWGN theory remains a separate column and dashed curve. The
`prediction/theory` ratio reports the filter penalty, with no second verdict.
The verdict compares measurement with the finite-filter prediction and has no
extra truncation allowance. For example, `--span 4` or `--sps 1` can show a large
penalty while still passing the simulation check.

At one sample/symbol, nonzero rolloff aliases the intended continuous-time RRC
pulse. Two or more integer samples/symbol avoids that aliasing, but finite span
can still introduce residual interference. The conjugate-reversed filter is
always the discrete matched filter; the demo allows all these configurations.

The default sweep's smallest expectation is about 1,000 errors, so no point is
inconclusive. Any FAIL makes the CLI exit nonzero; INCONCLUSIVE does not.
Finite RRC truncation adds about 0.035% relative BER at 8 dB, well below sampling
uncertainty. An independent conditional-BER calculation over complete PRBS cycles
checks the default filter within 0.1% of ideal theory across the sweep.

The 95% Wilson intervals are informational. At zero errors the report uses an
exact one-sided 95% upper bound, `1 - 0.05**(1/N)`. With 18 points there is roughly
a 60% chance at least one 95% interval misses its prediction; that is not a failed run.
