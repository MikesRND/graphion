// Standalone host for the scope component: synthetic, deterministic signals fed
// at 30 Hz into four side-panel scopes and one pop-out sized scope. Exposes
// window.demo for scope.check.mjs.

import type { Signal } from './dsp';
import { createScope, type ScopeHandle, type ScopeView } from './scope';

// mulberry32: a tiny seeded generator, so every frame k is reproducible.
function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gauss(r: () => number): number {
  return Math.sqrt(-2 * Math.log(r() + 1e-300)) * Math.cos(2 * Math.PI * r());
}

const N = 4096;
const RATE = 48000;

const signals = {
  /** Complex tone at 3 kHz plus a weaker one at -7.5 kHz; the amplitude wobbles. */
  tone(k: number): Signal {
    const r = rng(1000 + k);
    const amp = 1 + 0.4 * Math.sin(2 * Math.PI * 0.3 * (k / 30));
    const real = new Float64Array(N);
    const imag = new Float64Array(N);
    const t0 = k * N;
    for (let i = 0; i < N; i++) {
      const p1 = (2 * Math.PI * 3000 * (t0 + i)) / RATE;
      const p2 = (-2 * Math.PI * 7500 * (t0 + i)) / RATE;
      real[i] = amp * Math.cos(p1) + 0.1 * Math.cos(p2) + 0.05 * gauss(r);
      imag[i] = amp * Math.sin(p1) + 0.1 * Math.sin(p2) + 0.05 * gauss(r);
    }
    return { real, imag, rate: RATE };
  },
  /** An output that is already a display: a 512-bin PSD in dB with a drifting peak. */
  psd(k: number): Signal {
    const r = rng(2000 + k);
    const n = 512;
    const real = new Float64Array(n);
    const centre = 256 + 120 * Math.sin(k / 40);
    for (let i = 0; i < n; i++) {
      const d = (i - centre) / 6;
      real[i] = -80 + 50 / (1 + d * d) + 3 * gauss(r);
    }
    return { real, imag: null, rate: null };
  },
  /** Noisy QPSK with a slow carrier phase drift. */
  qpsk(k: number): Signal {
    const r = rng(3000 + k);
    const n = 1024;
    const real = new Float64Array(n);
    const imag = new Float64Array(n);
    const rot = 0.15 * Math.sin(k / 50);
    const c = Math.cos(rot);
    const s = Math.sin(rot);
    for (let i = 0; i < n; i++) {
      const re = (r() < 0.5 ? -1 : 1) * Math.SQRT1_2 + 0.08 * gauss(r);
      const im = (r() < 0.5 ? -1 : 1) * Math.SQRT1_2 + 0.08 * gauss(r);
      real[i] = re * c - im * s;
      imag[i] = re * s + im * c;
    }
    return { real, imag, rate: null };
  },
  /** Real chirp with an envelope, 4096 samples at 1 MHz. */
  chirp(k: number): Signal {
    const r = rng(4000 + k);
    const real = new Float64Array(N);
    const sweep = 20e3 + 15e3 * Math.sin(k / 60);
    for (let i = 0; i < N; i++) {
      const t = i / 1e6;
      const env = 0.6 + 0.4 * Math.sin((2 * Math.PI * i) / N + k / 10);
      real[i] = env * Math.sin(2 * Math.PI * (5e3 * t + 0.5 * sweep * 244 * t * t)) + 0.03 * gauss(r);
    }
    return { real, imag: null, rate: 1e6 };
  },
};

type Source = keyof typeof signals;
const layout: { title: string; view: ScopeView; source: Source }[] = [
  { title: 'time · complex tone', view: 'time', source: 'tone' },
  { title: 'spectrum · complex tone', view: 'spectrum', source: 'tone' },
  { title: 'array · psd', view: 'array', source: 'psd' },
  { title: 'constellation · qpsk', view: 'constellation', source: 'qpsk' },
];

const side = document.getElementById('side')!;
const scopes: ScopeHandle[] = [];
for (const item of layout) {
  const card = document.createElement('div');
  card.className = 'card';
  card.innerHTML = `<div class="head"><span></span><span><button title="Pop out (demo only)">↗</button> <button title="Close (demo only)">×</button></span></div>`;
  card.querySelector('.head span')!.textContent = item.title;
  const host = document.createElement('div');
  card.append(host);
  side.append(card);
  scopes.push(createScope(host, { view: item.view, compact: true }));
}
const tall = createScope(document.getElementById('tall')!, { view: 'time' });
scopes.push(tall);
const sources: Source[] = [...layout.map((l) => l.source), 'chirp'];

const popview = document.getElementById('popview') as HTMLSelectElement;
popview.addEventListener('change', () => tall.setView(popview.value as ScopeView));

const last: (Signal | null)[] = scopes.map(() => null);
// Per-scope update costs (ms), for the check's median.
const costs: number[][] = scopes.map(() => []);
const costEl = document.getElementById('cost')!;

function feed(i: number, s: { real: ArrayLike<number>; imag: ArrayLike<number> | null; rate: number | null }) {
  const signal: Signal = {
    real: s.real instanceof Float64Array ? s.real : Float64Array.from(s.real),
    imag: s.imag == null ? null : s.imag instanceof Float64Array ? s.imag : Float64Array.from(s.imag),
    rate: s.rate,
  };
  scopes[i].update(signal);
  last[i] = signal;
  const c = costs[i];
  c.push(scopes[i].state().lastUpdateMs);
  if (c.length > 600) c.shift();
}

let frame = 0;
function tick() {
  frame++;
  const made = new Map<Source, Signal>();
  scopes.forEach((_, i) => {
    const src = sources[i];
    if (!made.has(src)) made.set(src, signals[src](frame));
    feed(i, made.get(src)!);
  });
  if (frame % 15 === 0) {
    costEl.textContent = `update ${scopes.map((s) => s.state().lastUpdateMs.toFixed(2)).join(' / ')} ms`;
  }
}

let timer: number | null = null;
function setRunning(on: boolean) {
  if (on && timer == null) timer = window.setInterval(tick, 1000 / 30);
  if (!on && timer != null) {
    window.clearInterval(timer);
    timer = null;
  }
  const btn = document.getElementById('run')!;
  btn.textContent = on ? 'Pause' : 'Run';
  btn.classList.toggle('on', on);
}
document.getElementById('run')!.addEventListener('click', () => setRunning(timer == null));

tick();
setRunning(true);

declare global {
  interface Window {
    demo: {
      scopes: ScopeHandle[];
      setRunning(on: boolean): void;
      signals: typeof signals;
      feed: typeof feed;
      tick: typeof tick;
      last: (Signal | null)[];
      costs: number[][];
    };
  }
}
window.demo = { scopes, setRunning, signals, feed, tick, last, costs };
