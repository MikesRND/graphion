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

The renderer in [diagram.py](diagram.py) uses a layout specific to this testbench.
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

## Alignment and startup

The testbench computes combined delay `D = (len(tx) + len(rx) - 2)/2` in
oversampled samples and requires it to be an integer. Decimation phase is
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

The demo and regression tests share `agreement`:

- `sigma = sqrt(p_prediction * (1-p_prediction) / N)`.
- `PASS` when `abs(p_measured-p_prediction) <= 5*sigma`; otherwise `FAIL`.
- `INCONCLUSIVE` when `N*p_prediction <= 25`, where the band can admit zero errors.

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
