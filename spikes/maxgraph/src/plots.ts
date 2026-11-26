// Minimal canvas plots. Each call redraws the whole canvas from one payload.

import type { ProbePayload } from './types';

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

function range(series: number[][]): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (const values of series) {
    for (const v of values) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [-1, 1];
  if (lo === hi) return [lo - 1, hi + 1];
  const pad = (hi - lo) * 0.06;
  return [lo - pad, hi + pad];
}

export function drawLines(canvas: HTMLCanvasElement, series: number[][], caption = '') {
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
    values.forEach((v, i) => {
      const x = (i / (values.length - 1)) * width;
      const y = height - ((v - lo) / (hi - lo)) * height;
      if (i) g.lineTo(x, y);
      else g.moveTo(x, y);
    });
    g.stroke();
  });
  g.fillStyle = '#66758a';
  g.font = '10px system-ui, sans-serif';
  g.fillText(`${(hi - pad).toPrecision(3)}`, 3, 10);
  g.fillText(`${(lo + pad).toPrecision(3)}`, 3, height - 3);
  if (caption) g.fillText(caption, width - g.measureText(caption).width - 3, height - 3);
}

export function drawConstellation(canvas: HTMLCanvasElement, re: number[], im: number[]) {
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

function hz(value: number): string {
  const magnitude = Math.abs(value);
  if (magnitude >= 1e6) return `${+(value / 1e6).toPrecision(3)} MHz`;
  if (magnitude >= 1e3) return `${+(value / 1e3).toPrecision(3)} kHz`;
  return `${+value.toPrecision(3)} Hz`;
}

export function drawProbe(canvas: HTMLCanvasElement, payload: ProbePayload) {
  if (payload.unrun) {
    const { g, width, height } = prepare(canvas);
    g.fillStyle = '#8a97a8';
    g.font = '11px system-ui, sans-serif';
    g.fillText('not run yet', (width - g.measureText('not run yet').width) / 2, height / 2);
  } else if (payload.view === 'constellation' && payload.re && payload.im) {
    drawConstellation(canvas, payload.re, payload.im);
  } else if (payload.view === 'spectrum' && payload.db) {
    const span = payload.has_rate ? `${hz(payload.f0 ?? 0)} … ${hz(payload.f1 ?? 0)}` : 'rate unknown';
    drawLines(canvas, [payload.db], `dB · ${span}`);
  } else if (payload.re && payload.im) {
    drawLines(canvas, [payload.re, payload.im], 're / im');
  } else {
    drawLines(canvas, [payload.y ?? []]);
  }
}
