// Checks src/dsp.ts against NumPy. The module must already be bundled to dist/dsp.js and the
// reference cases written by `python -m spikes.inspection.dsp_fixture`:
//
//   npx esbuild src/dsp.ts --bundle --format=esm --outfile=dist/dsp.js
//   node dsp.check.mjs <fixture.json>
import { readFileSync } from 'node:fs';
import { fft, hann, minMaxDecimate, spectrum, stats } from './dist/dsp.js';

const fixture = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const results = [];
const report = (name, ok, detail = '') => {
  results.push(ok);
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? `  (${detail})` : ''}`);
};
const f64 = (list) => Float64Array.from(list);
const num = (s) => ({ nan: NaN, inf: Infinity, '-inf': -Infinity })[s] ?? Number(s);
const decode = (list) => Float64Array.from(list, num);
const maxAbs = (a) => a.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
const sci = (x) => x.toExponential(2);

for (const c of fixture.fft) {
  const re = f64(c.re);
  const im = f64(c.im);
  fft(re, im);
  let err = 0;
  let scale = 0;
  for (let k = 0; k < c.n; k++) {
    err = Math.max(err, Math.abs(re[k] - c.out_re[k]), Math.abs(im[k] - c.out_im[k]));
    scale = Math.max(scale, Math.hypot(c.out_re[k], c.out_im[k]));
  }
  const rel = err / (scale || 1);
  report(`fft n=${c.n} matches numpy.fft.fft`, rel < 1e-9, `relative error ${sci(rel)}`);
}
{
  const re = new Float64Array(0);
  fft(re, new Float64Array(0));
  report('fft of an empty array is a no-op', re.length === 0);
}

for (const c of fixture.hann) {
  const w = hann(c.n);
  let err = w.length === c.n ? 0 : Infinity;
  for (let k = 0; k < Math.min(c.n, w.length); k++) err = Math.max(err, Math.abs(w[k] - c.w[k]));
  report(`hann n=${c.n} matches numpy.hanning`, err < 1e-12, `max error ${sci(err)}`);
}

for (const c of fixture.spectrum) {
  const real = f64(c.re);
  const imag = c.im ? f64(c.im) : null;
  const copies = [real.slice(), imag?.slice()];
  const out = spectrum({ real, imag, rate: c.rate });
  const unchanged = real.every((v, k) => v === copies[0][k]) && (!imag || imag.every((v, k) => v === copies[1][k]));
  const shape = out.freq.length === c.freq.length && out.db.length === c.db.length;
  let freqErr = 0;
  let dbErr = 0;
  let floorOk = true;
  let compared = 0;
  if (shape) {
    for (let k = 0; k < c.freq.length; k++) {
      freqErr = Math.max(freqErr, Math.abs(out.freq[k] - c.freq[k]) / Math.max(1, Math.abs(c.freq[k])));
      if (c.db[k] > -150) {
        dbErr = Math.max(dbErr, Math.abs(out.db[k] - c.db[k]));
        compared++;
      } else if (!(out.db[k] < -140)) {
        floorOk = false;
      }
    }
  }
  const kind = imag ? 'complex' : 'real';
  report(`spectrum ${c.name}: ${c.freq.length} bins and hasRate`, shape && out.hasRate === (c.rate !== null), `${out.freq.length} bins, hasRate ${out.hasRate}`);
  report(`spectrum ${c.name}: ${kind} frequencies`, shape && freqErr < 1e-9, `relative error ${sci(freqErr)}`);
  report(`spectrum ${c.name}: dB within 1e-6`, shape && dbErr < 1e-6 && floorOk, `max error ${sci(dbErr)} over ${compared} bins, floor ${floorOk ? 'ok' : 'violated'}`);
  report(`spectrum ${c.name}: input untouched`, unchanged);
}

const sameNumber = (a, b) => (a === null || b === null ? a === b : Math.abs(a - b) <= 1e-12 * Math.max(1, Math.abs(b)));
for (const c of fixture.stats) {
  const got = stats({ real: decode(c.re), imag: c.im ? decode(c.im) : null, rate: null });
  const want = c.expected;
  const ok = got.nonfinite === want.nonfinite && sameNumber(got.peak, want.peak) && sameNumber(got.rms, want.rms);
  report(`stats ${c.name}`, ok, `got ${JSON.stringify(got)} want ${JSON.stringify(want)}`);
}

// Decimation is checked in JS only: its contract is structural, not numeric.
{
  const n = 10_000;
  const ramp = Float64Array.from({ length: n }, (_, k) => k / n);
  ramp[3333] = 50;
  ramp[7001] = -40;
  const { index, value } = minMaxDecimate(ramp, 200);
  let increasing = index.length === value.length;
  for (let k = 1; k < index.length; k++) if (!(index[k] > index[k - 1])) increasing = false;
  let match = true;
  for (let k = 0; k < index.length; k++) if (value[k] !== ramp[index[k]]) match = false;
  report('decimate keeps the spike and the dip', value.includes(50) && value.includes(-40) && index.includes(3333) && index.includes(7001));
  report('decimate indices strictly increase and point at their values', increasing && match);
  report('decimate output is at most two points per column', index.length <= 400 && index.length > 200, `${index.length} points`);
  report('decimate indices are a Float64Array', index instanceof Float64Array && value instanceof Float64Array);
}
{
  const short = Float64Array.from([1, 5, 2, 8, 3]);
  const { index, value } = minMaxDecimate(short, 3);
  report('decimate returns a short input unchanged', value.length === 5 && value.every((v, k) => v === short[k]) && index.every((v, k) => v === k));
}
{
  const values = new Float64Array(100).fill(NaN);
  for (let k = 50; k < 100; k++) values[k] = k;
  const { index, value } = minMaxDecimate(values, 2);
  report('decimate emits one NaN for a bucket with no finite value', value.length === 3 && Number.isNaN(value[0]) && index[0] === 0 && value[1] === 50 && value[2] === 99, `values ${Array.from(value)} at ${Array.from(index)}`);
}
{
  const { index, value } = minMaxDecimate(new Float64Array(30).fill(4), 5);
  report('decimate emits a flat bucket once', index.length === 5 && value.every((v) => v === 4), `${index.length} points`);
}
{
  const empty = minMaxDecimate(new Float64Array(0), 10);
  report('decimate of nothing is nothing', empty.index.length === 0 && empty.value.length === 0);
}

console.log(`${results.filter(Boolean).length} of ${results.length} checks passed`);
process.exit(results.every(Boolean) ? 0 : 1);
