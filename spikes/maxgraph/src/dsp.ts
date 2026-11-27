// Signal reductions done in the page, so the model's process only copies bytes.
//
// The Python reference for every function here is spikes/inspection/views.py
// (`spectrum`, `stats`); dsp.check.mjs compares the two.

/** One probed channel, decoded. `imag` is null for real data. */
export interface Signal {
  real: Float64Array;
  imag: Float64Array | null;
  /** Sample rate in Hz, or null when the port is outside the stream convention. */
  rate: number | null;
}

export interface Spectrum {
  /** Bin frequencies: Hz when the rate is known, else cycles per sample. */
  freq: Float64Array;
  /** Power in dB, floored at -200. */
  db: Float64Array;
  hasRate: boolean;
}

export interface Stats {
  nonfinite: number;
  peak: number | null;
  rms: number | null;
}

interface Radix2Plan {
  rev: Uint32Array;
  cos: Float64Array;
  /** Stored negated so the loop is the forward (numpy) sign convention. */
  sin: Float64Array;
}

interface BluesteinPlan {
  m: number;
  inner: Radix2Plan;
  chirpRe: Float64Array;
  chirpIm: Float64Array;
  /** FFT of the conjugate chirp, zero-padded and mirrored for circular convolution. */
  kernelRe: Float64Array;
  kernelIm: Float64Array;
  /** Scratch reused between calls; the page is single-threaded and fft never re-enters. */
  scratchRe: Float64Array;
  scratchIm: Float64Array;
}

// Tables for a long, odd length are large, and a page that scrubs through
// lengths would otherwise keep every one alive.
const CACHE_LIMIT = 8;
const radix2Plans = new Map<number, Radix2Plan>();
const bluesteinPlans = new Map<number, BluesteinPlan>();
const windows = new Map<number, Float64Array>();

function cached<T>(cache: Map<number, T>, key: number, make: () => T): T {
  const hit = cache.get(key);
  if (hit !== undefined) return hit;
  if (cache.size >= CACHE_LIMIT) cache.delete(cache.keys().next().value as number); // Map iterates oldest first
  const made = make();
  cache.set(key, made);
  return made;
}

function radix2Plan(n: number): Radix2Plan {
  return cached(radix2Plans, n, () => {
    const bits = Math.log2(n);
    const rev = new Uint32Array(n);
    for (let i = 1; i < n; i++) rev[i] = (rev[i >> 1]! >> 1) | ((i & 1) << (bits - 1));
    const cos = new Float64Array(n >> 1);
    const sin = new Float64Array(n >> 1);
    for (let k = 0; k < n >> 1; k++) {
      cos[k] = Math.cos((2 * Math.PI * k) / n);
      sin[k] = -Math.sin((2 * Math.PI * k) / n);
    }
    return { rev, cos, sin };
  });
}

function radix2(re: Float64Array, im: Float64Array, plan: Radix2Plan, n: number): void {
  const { rev, cos, sin } = plan;
  for (let i = 0; i < n; i++) {
    const j = rev[i]!;
    if (j > i) {
      const tr = re[i]!;
      re[i] = re[j]!;
      re[j] = tr;
      const ti = im[i]!;
      im[i] = im[j]!;
      im[j] = ti;
    }
  }
  for (let size = 2; size <= n; size <<= 1) {
    const half = size >> 1;
    const step = n / size;
    for (let start = 0; start < n; start += size) {
      for (let j = 0, k = 0; j < half; j++, k += step) {
        const a = start + j;
        const b = a + half;
        const wr = cos[k]!;
        const wi = sin[k]!;
        const xr = re[b]! * wr - im[b]! * wi;
        const xi = re[b]! * wi + im[b]! * wr;
        re[b] = re[a]! - xr;
        im[b] = im[a]! - xi;
        re[a] = re[a]! + xr;
        im[a] = im[a]! + xi;
      }
    }
  }
}

function bluesteinPlan(n: number): BluesteinPlan {
  return cached(bluesteinPlans, n, () => {
    let m = 1;
    while (m < 2 * n - 1) m <<= 1;
    const inner = radix2Plan(m);
    const chirpRe = new Float64Array(n);
    const chirpIm = new Float64Array(n);
    const kernelRe = new Float64Array(m);
    const kernelIm = new Float64Array(m);
    for (let k = 0; k < n; k++) {
      // exp(-i*pi*k^2/n) only depends on k^2 mod 2n; reducing first keeps the
      // angle small, where a raw k^2 would lose precision for large n.
      const angle = (Math.PI * ((k * k) % (2 * n))) / n;
      const c = Math.cos(angle);
      const s = Math.sin(angle);
      chirpRe[k] = c;
      chirpIm[k] = -s;
      kernelRe[k] = c;
      kernelIm[k] = s;
      if (k > 0) {
        kernelRe[m - k] = c;
        kernelIm[m - k] = s;
      }
    }
    radix2(kernelRe, kernelIm, inner, m);
    return { m, inner, chirpRe, chirpIm, kernelRe, kernelIm, scratchRe: new Float64Array(m), scratchIm: new Float64Array(m) };
  });
}

function bluestein(re: Float64Array, im: Float64Array, n: number): void {
  const { m, inner, chirpRe, chirpIm, kernelRe, kernelIm, scratchRe: ar, scratchIm: ai } = bluesteinPlan(n);
  ar.fill(0);
  ai.fill(0);
  for (let k = 0; k < n; k++) {
    ar[k] = re[k]! * chirpRe[k]! - im[k]! * chirpIm[k]!;
    ai[k] = re[k]! * chirpIm[k]! + im[k]! * chirpRe[k]!;
  }
  radix2(ar, ai, inner, m);
  // Inverse transform as conj(fft(conj(.))): the product is stored conjugated.
  for (let k = 0; k < m; k++) {
    const pr = ar[k]! * kernelRe[k]! - ai[k]! * kernelIm[k]!;
    const pi = ar[k]! * kernelIm[k]! + ai[k]! * kernelRe[k]!;
    ar[k] = pr;
    ai[k] = -pi;
  }
  radix2(ar, ai, inner, m);
  for (let k = 0; k < n; k++) {
    const yr = ar[k]! / m;
    const yi = -ai[k]! / m;
    re[k] = yr * chirpRe[k]! - yi * chirpIm[k]!;
    im[k] = yr * chirpIm[k]! + yi * chirpRe[k]!;
  }
}

/** In-place forward DFT of any length, unnormalised; matches numpy.fft.fft. */
export function fft(re: Float64Array, im: Float64Array): void {
  const n = re.length;
  if (n <= 1) return;
  if ((n & (n - 1)) === 0) radix2(re, im, radix2Plan(n), n);
  else bluestein(re, im, n);
}

/** numpy.hanning(n): the symmetric Hann window. The result is cached and shared: do not modify it. */
export function hann(n: number): Float64Array {
  return cached(windows, n, () => {
    const w = new Float64Array(n);
    if (n === 1) w[0] = 1;
    else for (let k = 0; k < n; k++) w[k] = 0.5 - 0.5 * Math.cos((2 * Math.PI * k) / (n - 1));
    return w;
  });
}

/** Matches views.spectrum in Python. */
export function spectrum(signal: Signal): Spectrum {
  const { real, imag, rate } = signal;
  const n = real.length;
  if (n === 0) return { freq: new Float64Array(0), db: new Float64Array(0), hasRate: rate !== null };
  const w = hann(n);
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  let energy = 0;
  for (let k = 0; k < n; k++) {
    re[k] = real[k]! * w[k]!;
    if (imag) im[k] = imag[k]! * w[k]!;
    energy += w[k]! * w[k]!;
  }
  fft(re, im);

  // numpy builds frequencies as integer * (1 / (n * step)) with step = 1 / rate.
  const scale = 1 / (n * (1 / (rate || 1)));
  const bins = imag ? n : (n >> 1) + 1;
  const freq = new Float64Array(bins);
  const db = new Float64Array(bins);
  // fftshift moves bin k to (k + n//2) mod n; the shifted bin j reads from j - n//2.
  const shift = imag ? n >> 1 : 0;
  const positive = (n - 1) >> 1; // last non-negative index of fftfreq
  for (let j = 0; j < bins; j++) {
    const src = imag ? (j - shift + n) % n : j;
    const power = (re[src]! * re[src]! + im[src]! * im[src]!) / energy;
    db[j] = 10 * Math.log10(Math.max(power, 1e-20));
    // Real input uses rfftfreq, whose last bin is +Nyquist, unlike fftfreq's -Nyquist.
    freq[j] = (imag ? (src <= positive ? src : src - n) : j) * scale;
  }
  return { freq, db, hasRate: rate !== null };
}

/** Matches views.stats in Python: peak and RMS of magnitudes over finite samples. */
export function stats(signal: Signal): Stats {
  const { real, imag } = signal;
  const n = real.length;
  const magnitude = (k: number): number => (imag ? Math.hypot(real[k]!, imag[k]!) : Math.abs(real[k]!));
  let finite = 0;
  let peak = 0;
  for (let k = 0; k < n; k++) {
    const x = magnitude(k);
    if (!Number.isFinite(x)) continue;
    finite++;
    if (x > peak) peak = x;
  }
  if (finite === 0) return { nonfinite: n, peak: null, rms: null };
  let rms = 0;
  if (peak) {
    // Scale by the peak so squaring very large values cannot overflow to infinity.
    let sum = 0;
    for (let k = 0; k < n; k++) {
      const x = magnitude(k);
      if (Number.isFinite(x)) sum += (x / peak) ** 2;
    }
    rms = peak * Math.sqrt(sum / finite);
  }
  return { nonfinite: n - finite, peak, rms };
}

/**
 * Reduce to at most two points per pixel column, the minimum and maximum of
 * each bucket in the order they occur, so a spike never disappears.
 * Returns the input unchanged when it already fits.
 */
export function minMaxDecimate(values: Float64Array, columns: number): { index: Float64Array; value: Float64Array } {
  const n = values.length;
  columns = Math.max(1, Math.floor(columns));
  if (n <= 2 * columns) {
    const index = new Float64Array(n);
    for (let k = 0; k < n; k++) index[k] = k;
    return { index, value: values };
  }
  const index = new Float64Array(2 * columns);
  const value = new Float64Array(2 * columns);
  let count = 0;
  for (let b = 0; b < columns; b++) {
    const start = Math.floor((b * n) / columns);
    const end = Math.floor(((b + 1) * n) / columns);
    let lo = -1;
    let hi = -1;
    for (let k = start; k < end; k++) {
      const x = values[k]!;
      if (!Number.isFinite(x)) continue;
      if (lo < 0 || x < values[lo]!) lo = k;
      if (hi < 0 || x > values[hi]!) hi = k;
    }
    if (lo < 0) {
      index[count] = start;
      value[count++] = NaN;
      continue;
    }
    const first = Math.min(lo, hi);
    const last = Math.max(lo, hi);
    index[count] = first;
    value[count++] = values[first]!;
    if (last !== first) {
      index[count] = last;
      value[count++] = values[last]!;
    }
  }
  return { index: index.slice(0, count), value: value.slice(0, count) };
}
