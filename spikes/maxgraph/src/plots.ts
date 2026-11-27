// Minimal canvas plots: the non-interactive thumbnails inside blocks, and parameter arrays.
// Each call redraws the whole canvas. The interactive scopes are in scope.ts.

import { minMaxDecimate, type Signal, spectrum } from './dsp';
import type { View } from './types';

const AXIS = '#c5ccd6';
const SERIES = ['#3b5bdb', '#e8590c'];

function prepare(canvas: HTMLCanvasElement) {
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 160;
  const height = canvas.clientHeight || 100;
  if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
  }
  const g = canvas.getContext('2d')!;
  g.setTransform(ratio, 0, 0, ratio, 0, 0);
  g.clearRect(0, 0, width, height);
  return { g, width, height };
}

function range(series: ArrayLike<number>[]): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (const values of series) {
    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [-1, 1];
  if (lo === hi) return [lo - 1, hi + 1];
  const pad = (hi - lo) * 0.06;
  return [lo - pad, hi + pad];
}

export function drawLines(canvas: HTMLCanvasElement, series: ArrayLike<number>[], caption = '') {
  const { g, width, height } = prepare(canvas);
  const [lo, hi] = range(series);
  const pad = (hi - lo) * 0.06 / 1.12;
  if (lo < 0 && hi > 0) {
    const zero = height - ((0 - lo) / (hi - lo)) * height;
    g.strokeStyle = AXIS;
    g.lineWidth = 1;
    g.beginPath();
    g.moveTo(0, zero);
    g.lineTo(width, zero);
    g.stroke();
  }
  series.forEach((values, index) => {
    if (values.length < 2) return;
    g.strokeStyle = SERIES[index % SERIES.length];
    g.lineWidth = 1.2;
    g.beginPath();
    for (let i = 0; i < values.length; i++) {
      const x = (i / (values.length - 1)) * width;
      const y = height - ((values[i] - lo) / (hi - lo)) * height;
      if (i) g.lineTo(x, y);
      else g.moveTo(x, y);
    }
    g.stroke();
  });
  g.fillStyle = '#66758a';
  g.font = '10px system-ui, sans-serif';
  g.fillText(`${(hi - pad).toPrecision(3)}`, 3, 10);
  g.fillText(`${(lo + pad).toPrecision(3)}`, 3, height - 3);
  if (caption) g.fillText(caption, width - g.measureText(caption).width - 3, height - 3);
}

export function drawConstellation(canvas: HTMLCanvasElement, re: ArrayLike<number>, im: ArrayLike<number>) {
  const { g, width, height } = prepare(canvas);
  let limit = 0;
  for (let i = 0; i < re.length; i++) limit = Math.max(limit, Math.abs(re[i]), Math.abs(im[i]));
  limit = (limit || 1) * 1.1;
  const scale = Math.min(width, height) / (2 * limit);
  const cx = width / 2;
  const cy = height / 2;
  g.strokeStyle = AXIS;
  g.lineWidth = 1;
  g.beginPath();
  g.moveTo(cx, 0);
  g.lineTo(cx, height);
  g.moveTo(0, cy);
  g.lineTo(width, cy);
  g.stroke();
  g.fillStyle = SERIES[0];
  for (let i = 0; i < re.length; i++) g.fillRect(cx + re[i] * scale - 1, cy - im[i] * scale - 1, 2, 2);
}

function message(canvas: HTMLCanvasElement, text: string) {
  const { g, width, height } = prepare(canvas);
  g.fillStyle = '#8a97a8';
  g.font = '11px system-ui, sans-serif';
  g.fillText(text, (width - g.measureText(text).width) / 2, height / 2);
}

// A thumbnail has a few hundred pixels; two points per column keep every spike of a long frame.
const thin = (values: Float64Array, canvas: HTMLCanvasElement) => minMaxDecimate(values, canvas.clientWidth || 160).value;

/** Draw a small, non-interactive picture of one signal, or say why there is none. */
export function drawThumb(canvas: HTMLCanvasElement, signal: Signal | null, view: View, note = '') {
  if (!signal) {
    message(canvas, note);
  } else if (view === 'constellation' && signal.imag) {
    // The newest points; a thumbnail of thousands would be a solid blot.
    const from = Math.max(0, signal.real.length - 512);
    drawConstellation(canvas, signal.real.subarray(from), signal.imag.subarray(from));
  } else if (view === 'spectrum') {
    drawLines(canvas, [thin(spectrum(signal).db, canvas)], 'dB');
  } else if (signal.imag) {
    drawLines(canvas, [thin(signal.real, canvas), thin(signal.imag, canvas)], 're / im');
  } else {
    drawLines(canvas, [thin(signal.real, canvas)]);
  }
}
