// Interactive signal scope: one probed signal, shown in the 380 px side panel or
// alone in a pop-out window. The page receives whole raw signals (up to 30 a
// second) and reduces them here.
//
// uPlot draws the time, array and spectrum views. Its linear path builder
// already reduces dense data to min/max per pixel column (when there are at
// least 4 points per canvas pixel), so the full-resolution arrays go to uPlot
// unchanged and markers, peak search and readouts read the same arrays.
// uPlot has no wheel zoom, no pan and no markers; those are written here on
// top of its scales and draw hook. The constellation is a plain canvas: uPlot's
// scatter mode (mode 2) ships no path builder for it, no equal aspect ratio and
// none of the gestures, so it would add nothing but weight.

import uPlot from 'uplot';
import 'uplot/dist/uPlot.min.css';
import { spectrum, type Signal } from './dsp';

export type ScopeView = 'time' | 'array' | 'spectrum' | 'constellation';
export interface Marker { id: number; index: number; x: number; y: number; series: number }
export interface ScopeState {
  view: ScopeView;
  points: number;
  x: [number, number];
  y: [number, number];
  zoomed: boolean;
  autoscale: boolean;
  markers: Marker[];
  delta: { dx: number; dy: number } | null;
  readout: string;
  updates: number;
  lastUpdateMs: number;
}
export interface ScopeHandle {
  update(signal: Signal | null, message?: string): void;
  setView(view: ScopeView): void;
  state(): ScopeState;
  plotBox(): { left: number; top: number; width: number; height: number };
  toClient(x: number, y: number): { x: number; y: number };
  setX(min: number, max: number): void;
  resetZoom(): void;
  setAutoscale(on: boolean): void;
  addMarker(x: number): Marker;
  peakSearch(): Marker | null;
  clearMarkers(): void;
  destroy(): void;
}

const SERIES = ['#3b5bdb', '#e8590c'];
const INK = '#1c2430';
const MUTED = '#66758a';
const GRID = '#e9edf2';
const FONT = '10px system-ui, -apple-system, "Segoe UI", sans-serif';
const HINT = 'drag: zoom · wheel: zoom · shift/middle-drag: pan · dbl-click: reset · click: marker';
const CONS_HINT = 'wheel: zoom · drag: pan · dbl-click: reset';
// A press that moves less than this is a click (marker), not a drag (zoom). It is
// also uPlot's drag.dist, so the two never disagree about what happened.
const CLICK_SLOP = 4;

const CSS = `
.gscope { display: flex; flex-direction: column; height: 100%; min-height: 0; gap: 4px; padding: 6px; box-sizing: border-box;
  font: 12px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--text, ${INK}); }
.gscope.compact { padding: 0; gap: 3px; }
.gscope-bar { display: flex; align-items: center; gap: 4px; flex: none; min-width: 0; }
.gscope-bar button { font-family: inherit; font-size: 11px; line-height: 1.3; color: inherit; border: 1px solid var(--line, #d9dee5);
  background: var(--panel, #fff); border-radius: 5px; padding: 1px 7px; cursor: pointer; white-space: nowrap; flex: none; }
.gscope-bar button:hover { border-color: #9aa7b8; }
.gscope-bar button.on { background: #e7ecff; border-color: var(--accent, #3b5bdb); }
.gscope-bar button:disabled { opacity: .45; cursor: default; }
.gscope-info { margin-left: auto; color: var(--muted, ${MUTED}); font-size: 11px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; min-width: 0; }
.gscope-plot { position: relative; flex: 1 1 0; min-height: 60px; overflow: hidden; }
.gscope.compact .gscope-plot { flex: 1 1 160px; min-height: 150px; }
.gscope-plot > canvas { position: absolute; left: 0; top: 0; width: 100%; height: 100%; display: block; cursor: grab; }
.gscope-u { position: absolute; left: 0; top: 0; }
.gscope-u .u-over { cursor: crosshair; }
.gscope-msg { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; text-align: center;
  background: var(--panel, #fff); color: #8a97a8; font-size: 11px; padding: 8px; }
.gscope-msg[hidden] { display: none; }
.gscope-readout { flex: none; height: 28px; overflow: hidden; white-space: pre-wrap; overflow-wrap: anywhere;
  font: 11px/14px ui-monospace, "SF Mono", Consolas, monospace; color: var(--text, ${INK}); }
.gscope-readout.idle { color: var(--muted, ${MUTED}); }
`;

function injectStyle() {
  if (document.getElementById('gscope-style')) return;
  const style = document.createElement('style');
  style.id = 'gscope-style';
  style.textContent = CSS;
  document.head.append(style);
}

// ---- formatting ----------------------------------------------------------

function num(v: number, digits = 5): string {
  if (!Number.isFinite(v)) return String(v);
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1e6 || a < 1e-4) return v.toExponential(digits - 1).replace(/\.?0+e/, 'e');
  return String(+v.toPrecision(digits));
}

/** Frequency with an engineering unit; 5 significant digits, trailing zeros dropped. */
export function fmtHz(v: number, digits = 5): string {
  const a = Math.abs(v);
  if (a >= 1e6) return `${num(v / 1e6, digits)} MHz`;
  if (a >= 1e3) return `${num(v / 1e3, digits)} kHz`;
  return `${num(v, digits)} Hz`;
}

function fmtTime(s: number): string {
  const a = Math.abs(s);
  if (a >= 1 || a === 0) return `${num(s, 4)} s`;
  if (a >= 1e-3) return `${num(s * 1e3, 4)} ms`;
  if (a >= 1e-6) return `${num(s * 1e6, 4)} µs`;
  return `${num(s * 1e9, 4)} ns`;
}

// All ticks of one axis share a unit, so "500 Hz, 1 kHz" never mixes on one axis.
function hzTicks(splits: number[]): string[] {
  let m = 0;
  for (const v of splits) m = Math.max(m, Math.abs(v));
  const [div, unit] = m >= 1e6 ? [1e6, 'MHz'] : m >= 1e3 ? [1e3, 'kHz'] : [1, 'Hz'];
  return splits.map((v) => `${+(v / div).toPrecision(4)} ${unit}`);
}

function niceStep(raw: number): number {
  const p = 10 ** Math.floor(Math.log10(raw));
  const m = raw / p;
  return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 5 ? 5 : 10) * p;
}

// ---- searching sorted x arrays (sample index or ascending frequency) ------

function nearest(xs: Float64Array, v: number): number {
  let lo = 0;
  let hi = xs.length - 1;
  if (hi <= 0 || v <= xs[0]) return 0;
  if (v >= xs[hi]) return hi;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] < v) lo = mid;
    else hi = mid;
  }
  return v - xs[lo] <= xs[hi] - v ? lo : hi;
}

/** First index with xs[i] >= v. */
function lowerBound(xs: Float64Array, v: number): number {
  let lo = 0;
  let hi = xs.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] < v) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/** Last index with xs[i] <= v (-1 when none). */
function upperBound(xs: Float64Array, v: number): number {
  let lo = 0;
  let hi = xs.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= v) lo = mid + 1;
    else hi = mid;
  }
  return lo - 1;
}

function padRange(lo: number | null, hi: number | null): uPlot.Range.MinMax {
  if (lo == null || hi == null || !Number.isFinite(lo) || !Number.isFinite(hi)) return [-1, 1];
  if (lo === hi) {
    const d = Math.abs(lo) * 0.1 || 1;
    return [lo - d, hi + d];
  }
  const pad = (hi - lo) * 0.06;
  return [lo - pad, hi + pad];
}

interface ConsView { cx: number; cy: number; half: number } // half: axis units across half the shorter canvas side

export function createScope(host: HTMLElement, options: { view: ScopeView; compact?: boolean }): ScopeHandle {
  injectStyle();
  const compact = !!options.compact;
  const make = <K extends keyof HTMLElementTagNameMap>(tag: K, cls: string, parent: HTMLElement) => {
    const e = document.createElement(tag);
    e.className = cls;
    parent.append(e);
    return e;
  };
  const root = make('div', compact ? 'gscope compact' : 'gscope', host);
  const bar = make('div', 'gscope-bar', root);
  const button = (label: string, title: string, act: () => void) => {
    const b = make('button', '', bar);
    b.type = 'button';
    b.textContent = label;
    b.title = title;
    b.addEventListener('click', act);
    return b;
  };
  const autoBtn = button('Auto', 'Fit y to the visible data on every update', () => handle.setAutoscale(!autoscale));
  const peakBtn = button('Peak', 'Put the active marker on the maximum inside the visible range', () => handle.peakSearch());
  const clearBtn = button('Clear', 'Remove the markers', () => handle.clearMarkers());
  const resetBtn = button('Reset zoom', 'Show the whole signal (double-click does the same)', () => handle.resetZoom());
  const info = make('span', 'gscope-info', bar);
  const plot = make('div', 'gscope-plot', root);
  const msg = make('div', 'gscope-msg', plot);
  msg.hidden = true;
  const readout = make('div', 'gscope-readout', root);

  let view = options.view;
  let signal: Signal | null = null;
  let message = '';
  let updates = 0;
  let lastUpdateMs = 0;
  let autoscale = true;

  // What the uPlot views draw: x, then one or two y series. ys[0] is what markers
  // and peak search read. Arrays come straight from the signal or spectrum().
  let xs: Float64Array = new Float64Array(0);
  let ys: Float64Array[] = [];
  let shownLen = -1; // length and rate of the last drawn signal; a change makes the old x range meaningless
  let shownRate: number | null = null;
  let hasRate = false;
  let indexX = new Float64Array(0);
  let mag = new Float64Array(0);

  let u: uPlot | null = null;
  let uKey = '';
  let userX: [number, number] | null = null; // the user's x range; null shows the full signal
  let hoverIdx: number | null = null;

  interface Mark { id: number; index: number; order: number }
  let markers: Mark[] = [];
  let order = 0;
  // A double-click starts with a click that would place a marker; the dblclick
  // puts the markers back as they were before that click.
  let beforeClick: Mark[] | null = null;

  let canvas: HTMLCanvasElement | null = null;
  let consFit: ConsView | null = null;
  let consUser: ConsView | null = null;
  let consHover: [number, number] | null = null;

  // Listeners on the current renderer, removed when it is replaced or destroyed.
  let detach: (() => void)[] = [];
  let panEnd: (() => void) | null = null;

  const observer = new ResizeObserver(() => resize());
  observer.observe(plot);

  // ---- shared helpers ----------------------------------------------------

  const plotSize = () => ({ w: Math.max(plot.clientWidth, 40), h: Math.max(plot.clientHeight, 40) });

  function fullX(): [number, number] {
    const n = xs.length;
    if (n === 0) return [0, 1];
    if (n === 1) return [xs[0] - 0.5, xs[0] + 0.5];
    return [xs[0], xs[n - 1]];
  }

  function isFull(mn: number, mx: number): boolean {
    const [f0, f1] = fullX();
    const tol = (f1 - f0) * 1e-9;
    return Math.abs(mn - f0) <= tol && Math.abs(mx - f1) <= tol;
  }

  // Keep a requested x range inside the data, preserving its span where possible,
  // and never narrower than two sample spacings (uPlot ignores spans < 1e-16).
  function clampX(mn: number, mx: number): [number, number] {
    const [f0, f1] = fullX();
    const full = f1 - f0;
    const minSpan = xs.length > 1 ? (2 * full) / (xs.length - 1) : full;
    let span = mx - mn;
    if (!(span < full)) return [f0, f1];
    if (span < minSpan) {
      const c = (mn + mx) / 2;
      span = minSpan;
      mn = c - span / 2;
      mx = c + span / 2;
    }
    if (mn < f0) [mn, mx] = [f0, f0 + span];
    if (mx > f1) [mn, mx] = [f1 - span, f1];
    return [mn, mx];
  }

  function showMessage(text: string) {
    msg.textContent = text;
    msg.hidden = false;
  }

  function teardownRenderer() {
    panEnd?.();
    for (const off of detach) off();
    detach = [];
    u?.destroy();
    u = null;
    uKey = '';
    canvas?.remove();
    canvas = null;
    hoverIdx = null;
    consHover = null;
  }

  function listen<K extends keyof HTMLElementEventMap>(target: HTMLElement, type: K, fn: (e: HTMLElementEventMap[K]) => void, opts?: AddEventListenerOptions) {
    target.addEventListener(type, fn as EventListener, opts);
    detach.push(() => target.removeEventListener(type, fn as EventListener, opts));
  }

  // Drag helper for pans: follows the mouse on window so it keeps working outside the plot.
  function startDrag(onMove: (dx: number, dy: number) => void, x0: number, y0: number) {
    panEnd?.();
    const move = (e: MouseEvent) => onMove(e.clientX - x0, e.clientY - y0);
    const up = () => panEnd?.();
    window.addEventListener('mousemove', move);
    window.addEventListener('mouseup', up);
    panEnd = () => {
      window.removeEventListener('mousemove', move);
      window.removeEventListener('mouseup', up);
      panEnd = null;
    };
  }

  // ---- uPlot views -------------------------------------------------------

  function xLabel(v: number): string {
    if (view === 'spectrum') return hasRate ? fmtHz(v) : `${num(v)} cyc/smp`;
    const rate = signal?.rate;
    return view === 'time' && rate ? `#${v} ${fmtTime(v / rate)}` : `#${v}`;
  }

  function yLabel(v: number): string {
    return view === 'spectrum' ? `${num(v, 4)} dB` : num(v);
  }

  function buildPlot() {
    const { w, h } = plotSize();
    const base: uPlot.Axis = { font: FONT, stroke: MUTED, gap: 3, grid: { stroke: GRID, width: 1 }, ticks: { stroke: GRID, width: 1, size: 3 } };
    const xAxis: uPlot.Axis = { ...base, size: 20, space: 56 };
    if (view === 'spectrum' && hasRate) {
      xAxis.values = (_u, splits) => hzTicks(splits);
      xAxis.space = 64;
    } else if (view === 'spectrum') {
      xAxis.label = 'cycles/sample';
      xAxis.labelSize = 13;
      xAxis.labelFont = FONT;
      xAxis.labelGap = 0;
    }
    const yAxis: uPlot.Axis = { ...base, size: compact ? 42 : 50, space: 22 };
    const names = view === 'time' && ys.length === 2 ? ['re', 'im'] : [view === 'spectrum' ? 'dB' : 'y'];
    const series: uPlot.Series[] = [{ label: 'x' }, ...ys.map((_, i) => ({ label: names[i], stroke: SERIES[i], width: 1, points: { show: false } }))];
    const opts: uPlot.Options = {
      width: w,
      height: h,
      class: 'gscope-u',
      legend: { show: false },
      padding: [8, 10, 0, 0],
      scales: {
        x: { time: false },
        // auto() is read on every rescale, so the Auto toggle needs no rebuild. With it
        // off, uPlot leaves y alone even when x changes.
        y: { auto: () => autoscale, range: (_u, lo, hi) => padRange(lo, hi) },
      },
      axes: [xAxis, yAxis],
      series,
      cursor: {
        y: false,
        points: { size: 6 },
        drag: { x: true, y: false, dist: CLICK_SLOP, setScale: true },
        // Shift+left is our pan; keep it from starting uPlot's selection.
        bind: {
          mousedown: (_u, targ, handler) => (e) => {
            if (e.button === 0 && e.target === targ && !e.shiftKey) handler(e);
            return null;
          },
        },
      },
      hooks: {
        setScale: [(plotU, key) => {
          if (key !== 'x') return;
          const mn = plotU.scales.x.min ?? 0;
          const mx = plotU.scales.x.max ?? 0;
          userX = isFull(mn, mx) ? null : [mn, mx];
          syncButtons();
        }],
        setCursor: [(plotU) => {
          const idx = (plotU.cursor.left ?? -1) >= 0 ? (plotU.cursor.idx ?? null) : null;
          if (idx !== hoverIdx) {
            hoverIdx = idx;
            renderReadout();
          }
        }],
        draw: [drawMarkers],
      },
    };
    u = new uPlot(opts, [xs, ...ys] as uPlot.AlignedData, plot);
    plot.append(msg); // keep the message above uPlot's root
    const over = u.over;
    let downX = 0;
    let downY = 0;
    listen(over, 'mousedown', (e) => {
      downX = e.clientX;
      downY = e.clientY;
      if (e.button === 1 || (e.button === 0 && e.shiftKey)) {
        e.preventDefault(); // no middle-button autoscroll, no text selection
        if (!u) return;
        const mn = u.scales.x.min ?? 0;
        const mx = u.scales.x.max ?? 0;
        const width = over.clientWidth || 1;
        startDrag((dx) => {
          const dv = (dx / width) * (mx - mn);
          handle.setX(mn - dv, mx - dv);
        }, e.clientX, e.clientY);
      }
    });
    listen(over, 'click', (e) => {
      if (e.button !== 0 || e.shiftKey || e.detail > 1) return;
      if (Math.abs(e.clientX - downX) >= CLICK_SLOP || Math.abs(e.clientY - downY) >= CLICK_SLOP) return;
      if (!u || xs.length === 0) return;
      beforeClick = markers.map((m) => ({ ...m }));
      handle.addMarker(u.posToVal(e.clientX - over.getBoundingClientRect().left, 'x'));
    });
    // uPlot's own dblclick handler resets x to the full data range.
    listen(over, 'dblclick', () => {
      if (beforeClick) markers = beforeClick;
      beforeClick = null;
      redrawNow();
      renderReadout();
    });
    listen(over, 'wheel', (e) => {
      if (!u || xs.length < 2) return;
      e.preventDefault();
      const dy = e.deltaY * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 400 : 1);
      const f = Math.exp(dy * 0.0015);
      const c = u.posToVal(e.clientX - over.getBoundingClientRect().left, 'x');
      const mn = u.scales.x.min ?? 0;
      const mx = u.scales.x.max ?? 0;
      handle.setX(c - (c - mn) * f, c + (mx - c) * f);
    }, { passive: false });
  }

  // Markers are drawn on uPlot's canvas after each redraw, from the stored sample
  // index, so they follow zoom, pan, resize and new data without bookkeeping.
  function drawMarkers(plotU: uPlot) {
    if (!markers.length || !ys.length) return;
    const g = plotU.ctx;
    const { left, top, width, height } = plotU.bbox;
    const r = uPlot.pxRatio;
    g.save();
    g.beginPath();
    g.rect(left, top, width, height);
    g.clip();
    g.font = `${10 * r}px system-ui, sans-serif`;
    // uPlot leaves its last axis alignment on the context (often right or center).
    g.textAlign = 'left';
    g.textBaseline = 'top';
    for (const m of markers) {
      const xv = xs[m.index];
      const yv = ys[0][m.index];
      const px = Math.round(plotU.valToPos(xv, 'x', true)) + 0.5;
      if (px < left - 1 || px > left + width + 1) continue;
      g.strokeStyle = INK;
      g.globalAlpha = 0.45;
      g.lineWidth = r;
      g.setLineDash([3 * r, 3 * r]);
      g.beginPath();
      g.moveTo(px, top);
      g.lineTo(px, top + height);
      g.stroke();
      g.setLineDash([]);
      g.globalAlpha = 1;
      if (Number.isFinite(yv)) {
        const py = plotU.valToPos(yv, 'y', true);
        const s = 4.5 * r;
        g.fillStyle = INK;
        g.beginPath();
        g.moveTo(px, py - s);
        g.lineTo(px + s, py);
        g.lineTo(px, py + s);
        g.lineTo(px - s, py);
        g.closePath();
        g.fill();
        g.strokeStyle = '#fff';
        g.lineWidth = r;
        g.stroke();
      }
      const label = `M${m.id}`;
      const tw = g.measureText(label).width + 6 * r;
      const lx = Math.min(px + 3 * r, left + width - tw);
      const ly = top + (2 + (m.id - 1) * 15) * r; // M2 on a second row, so close markers stay readable
      g.fillStyle = INK;
      g.fillRect(lx, ly, tw, 13 * r);
      g.fillStyle = '#fff';
      g.fillText(label, lx + 3 * r, ly + 1.5 * r);
    }
    g.restore();
  }

  // Draw synchronously: uPlot otherwise defers its commit to a microtask.
  function redrawNow() {
    if (u) u.batch(() => {});
    else if (canvas) drawCons();
  }

  function renderSeries(s: Signal) {
    const n = s.real.length;
    let nextX: Float64Array;
    let nextY: Float64Array[];
    if (view === 'spectrum') {
      const sp = spectrum(s);
      nextX = sp.freq;
      nextY = [sp.db];
      hasRate = sp.hasRate;
    } else {
      if (indexX.length !== n) {
        indexX = new Float64Array(n);
        for (let i = 0; i < n; i++) indexX[i] = i;
      }
      nextX = indexX;
      hasRate = s.rate != null;
      if (view === 'time') nextY = s.imag ? [s.real, s.imag] : [s.real];
      else if (s.imag) {
        // An array view of complex data has no single value per index; show magnitude.
        if (mag.length !== n) mag = new Float64Array(n);
        const re = s.real;
        const im = s.imag;
        for (let i = 0; i < n; i++) mag[i] = Math.hypot(re[i], im[i]);
        nextY = [mag];
      } else nextY = [s.real];
    }
    if (nextX.length === 0) {
      showMessage('empty signal');
      return;
    }
    if (nextX.length !== shownLen || s.rate !== shownRate) {
      userX = null;
      markers = [];
      beforeClick = null;
    }
    shownLen = nextX.length;
    shownRate = s.rate;
    xs = nextX;
    ys = nextY;
    msg.hidden = true;
    const key = `${view}|${ys.length}|${hasRate}`;
    if (!u || key !== uKey) {
      const keepY = u && !autoscale ? { min: u.scales.y.min ?? -1, max: u.scales.y.max ?? 1 } : null;
      teardownRenderer();
      buildPlot();
      uKey = key;
      if (keepY) u!.setScale('y', keepY);
    } else {
      const plotU = u;
      const [mn, mx] = userX ?? fullX();
      // setData(..., true) would re-range x to the full data and drop the user's
      // zoom; with false it only swaps data, and setting x explicitly re-ranges
      // the auto y scale over the visible indices.
      plotU.batch(() => {
        plotU.setData([xs, ...ys] as uPlot.AlignedData, false);
        plotU.setScale('x', { min: mn, max: mx });
      });
    }
  }

  // ---- constellation (plain canvas) -------------------------------------

  function fitCons(re: Float64Array, im: Float64Array): ConsView {
    let x0 = Infinity;
    let x1 = -Infinity;
    let y0 = Infinity;
    let y1 = -Infinity;
    for (let i = 0; i < re.length; i++) {
      const x = re[i];
      const y = im[i];
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      if (x < x0) x0 = x;
      if (x > x1) x1 = x;
      if (y < y0) y0 = y;
      if (y > y1) y1 = y;
    }
    if (!(x1 >= x0)) return { cx: 0, cy: 0, half: 1 };
    const cx = (x0 + x1) / 2;
    const cy = (y0 + y1) / 2;
    const half = (Math.max(x1 - x0, y1 - y0) / 2) * 1.12 || Math.max(Math.abs(cx), Math.abs(cy)) * 0.1 || 1;
    return { cx, cy, half };
  }

  const consView = (): ConsView => consUser ?? consFit ?? { cx: 0, cy: 0, half: 1 };

  function consGeom() {
    const w = canvas?.clientWidth || plotSize().w;
    const h = canvas?.clientHeight || plotSize().h;
    const v = consView();
    const k = Math.min(w, h) / (2 * v.half); // pixels per axis unit, the same on both axes
    return { w, h, v, k, x0: v.cx - w / 2 / k, x1: v.cx + w / 2 / k, y0: v.cy - h / 2 / k, y1: v.cy + h / 2 / k };
  }

  function buildCanvas() {
    const c = document.createElement('canvas');
    plot.prepend(c);
    canvas = c;
    let dragged = false;
    listen(c, 'mousedown', (e) => {
      if (e.button !== 0 && e.button !== 1) return;
      e.preventDefault();
      const start = consView();
      const { k } = consGeom();
      dragged = false;
      c.style.cursor = 'grabbing';
      startDrag((dx, dy) => {
        if (Math.abs(dx) + Math.abs(dy) >= CLICK_SLOP) dragged = true;
        if (!dragged) return;
        consUser = { cx: start.cx - dx / k, cy: start.cy + dy / k, half: start.half };
        drawCons();
        syncButtons();
      }, e.clientX, e.clientY);
      const prevEnd = panEnd;
      panEnd = () => {
        prevEnd?.();
        c.style.cursor = '';
      };
    });
    listen(c, 'wheel', (e) => {
      e.preventDefault();
      const { w, h, v, k } = consGeom();
      const rect = c.getBoundingClientRect();
      const vx = v.cx + (e.clientX - rect.left - w / 2) / k;
      const vy = v.cy - (e.clientY - rect.top - h / 2) / k;
      const dy = e.deltaY * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 400 : 1);
      const f = Math.exp(dy * 0.0015);
      consUser = { cx: vx - (vx - v.cx) * f, cy: vy - (vy - v.cy) * f, half: v.half * f };
      drawCons();
      syncButtons();
    }, { passive: false });
    listen(c, 'dblclick', () => handle.resetZoom());
    listen(c, 'mousemove', (e) => {
      const { w, h, v, k } = consGeom();
      const rect = c.getBoundingClientRect();
      consHover = [v.cx + (e.clientX - rect.left - w / 2) / k, v.cy - (e.clientY - rect.top - h / 2) / k];
      renderReadout();
    });
    listen(c, 'mouseleave', () => {
      consHover = null;
      renderReadout();
    });
  }

  function drawCons() {
    if (!canvas) return;
    const r = window.devicePixelRatio || 1;
    const { w, h, v, k, x0, x1, y0, y1 } = consGeom();
    if (canvas.width !== Math.round(w * r) || canvas.height !== Math.round(h * r)) {
      canvas.width = Math.round(w * r);
      canvas.height = Math.round(h * r);
    }
    const g = canvas.getContext('2d')!;
    g.setTransform(r, 0, 0, r, 0, 0);
    g.clearRect(0, 0, w, h);
    const px = (x: number) => w / 2 + (x - v.cx) * k;
    const py = (y: number) => h / 2 - (y - v.cy) * k;
    // Grid on round values, labelled along the bottom and left edges.
    const step = niceStep(56 / k);
    g.lineWidth = 1;
    g.font = FONT;
    g.fillStyle = MUTED;
    g.textBaseline = 'bottom';
    for (let x = Math.ceil(x0 / step) * step; x <= x1; x += step) {
      const p = Math.round(px(x)) + 0.5;
      g.strokeStyle = Math.abs(x) < step * 1e-6 ? '#c5ccd6' : GRID;
      g.beginPath();
      g.moveTo(p, 0);
      g.lineTo(p, h);
      g.stroke();
      // The left edge belongs to the y labels.
      if (p > 24) g.fillText(num(Math.abs(x) < step * 1e-6 ? 0 : x, 3), p + 2, h - 1);
    }
    g.textBaseline = 'top';
    for (let y = Math.ceil(y0 / step) * step; y <= y1; y += step) {
      const p = Math.round(py(y)) + 0.5;
      g.strokeStyle = Math.abs(y) < step * 1e-6 ? '#c5ccd6' : GRID;
      g.beginPath();
      g.moveTo(0, p);
      g.lineTo(w, p);
      g.stroke();
      // The bottom edge belongs to the x labels.
      if (p < h - 24) g.fillText(num(Math.abs(y) < step * 1e-6 ? 0 : y, 3), 2, p + 1);
    }
    const s = signal;
    if (!s || !s.imag) return;
    const re = s.real;
    const im = s.imag;
    g.fillStyle = SERIES[0];
    g.globalAlpha = 0.75;
    for (let i = 0; i < re.length; i++) {
      const x = px(re[i]);
      const y = py(im[i]);
      // fillRect ignores NaN on its own; the bounds test skips off-screen points.
      if (x >= -2 && x <= w + 2 && y >= -2 && y <= h + 2) g.fillRect(x - 1, y - 1, 2, 2);
    }
    g.globalAlpha = 1;
  }

  function renderConstellation(s: Signal) {
    if (!s.imag) {
      teardownRenderer();
      showMessage('constellation needs complex samples');
      return;
    }
    if (s.real.length !== shownLen || s.rate !== shownRate) consUser = null;
    shownLen = s.real.length;
    shownRate = s.rate;
    if (!canvas) buildCanvas();
    if (autoscale || !consFit) consFit = fitCons(s.real, s.imag);
    msg.hidden = true;
    drawCons();
  }

  // ---- readout, toolbar, sizing ------------------------------------------

  function markerList(): Marker[] {
    if (!ys.length) return [];
    return [...markers].sort((a, b) => a.id - b.id).map((m) => ({ id: m.id, index: m.index, x: xs[m.index], y: ys[0][m.index], series: 0 }));
  }

  function deltaOf(list: Marker[]) {
    return list.length === 2 ? { dx: list[1].x - list[0].x, dy: list[1].y - list[0].y } : null;
  }

  function dxLabel(dx: number): string {
    if (view === 'spectrum') return hasRate ? fmtHz(dx) : `${num(dx)} cyc/smp`;
    const rate = signal?.rate;
    if (view === 'time' && rate && dx !== 0) return `${dx} smp ${fmtTime(dx / rate)} (1/Δ ${fmtHz(rate / Math.abs(dx))})`;
    return `${dx} smp`;
  }

  function readoutText(): string {
    if (!signal || !msg.hidden) return '';
    if (view === 'constellation') {
      return consHover ? `re ${num(consHover[0], 4)}  im ${num(consHover[1], 4)}` : '';
    }
    if (hoverIdx != null && hoverIdx < xs.length) {
      const i = hoverIdx;
      const vals = ys.length === 2 ? `re ${num(ys[0][i])}  im ${num(ys[1][i])}` : yLabel(ys[0][i]);
      return `${xLabel(xs[i])}  ${vals}`;
    }
    const list = markerList();
    if (!list.length) return '';
    const parts = list.map((m) => `M${m.id} ${xLabel(m.x)} ${yLabel(m.y)}`);
    const d = deltaOf(list);
    if (d) parts.push(`Δ ${dxLabel(d.dx)} ${view === 'spectrum' ? `${num(d.dy, 4)} dB` : num(d.dy)}`);
    return parts.join('   ');
  }

  function renderReadout() {
    const text = readoutText();
    const shown = text || (signal && msg.hidden ? (view === 'constellation' ? CONS_HINT : HINT) : '');
    if (readout.textContent !== shown) readout.textContent = shown;
    readout.classList.toggle('idle', !text);
  }

  function syncButtons() {
    autoBtn.classList.toggle('on', autoscale);
    autoBtn.textContent = autoscale ? 'Auto on' : 'Auto off';
    const uplotView = view !== 'constellation';
    peakBtn.disabled = !uplotView;
    clearBtn.disabled = !uplotView;
    resetBtn.disabled = !(uplotView ? userX : consUser);
    const s = signal;
    const rate = s?.rate ? ` · ${fmtHz(s.rate)}` : '';
    const text = s ? `${s.real.length} pts${s.imag ? ' cplx' : ''}${rate}` : '';
    if (info.textContent !== text) info.textContent = text;
  }

  function resize() {
    const { w, h } = plotSize();
    if (u && (u.width !== w || u.height !== h)) u.setSize({ width: w, height: h });
    else if (canvas) drawCons();
  }

  function render() {
    const s = signal;
    if (!s) {
      showMessage(message || 'no data');
    } else if (view === 'constellation') renderConstellation(s);
    else renderSeries(s);
    syncButtons();
    renderReadout();
  }

  // ---- public handle -----------------------------------------------------

  const handle: ScopeHandle = {
    update(s, text = '') {
      const t0 = performance.now();
      updates++;
      signal = s;
      message = text;
      render();
      lastUpdateMs = performance.now() - t0;
    },

    setView(next) {
      if (next === view) return;
      teardownRenderer();
      view = next;
      xs = new Float64Array(0);
      ys = [];
      shownLen = -1;
      userX = null;
      markers = [];
      consFit = consUser = null;
      render();
    },

    state() {
      const list = view === 'constellation' ? [] : markerList();
      let x: [number, number] = [0, 0];
      let y: [number, number] = [0, 0];
      let zoomed = false;
      if (u) {
        x = [u.scales.x.min ?? 0, u.scales.x.max ?? 0];
        y = [u.scales.y.min ?? 0, u.scales.y.max ?? 0];
        zoomed = !isFull(x[0], x[1]);
      } else if (canvas) {
        const gm = consGeom();
        x = [gm.x0, gm.x1];
        y = [gm.y0, gm.y1];
        zoomed = consUser != null;
      }
      return {
        view,
        points: signal ? signal.real.length : 0,
        x,
        y,
        zoomed,
        autoscale,
        markers: list,
        delta: deltaOf(list),
        readout: readout.classList.contains('idle') ? '' : readout.textContent ?? '',
        updates,
        lastUpdateMs,
      };
    },

    plotBox() {
      const target = u ? u.over : canvas ?? plot;
      const r = target.getBoundingClientRect();
      return { left: r.left, top: r.top, width: r.width, height: r.height };
    },

    toClient(x, y) {
      const box = handle.plotBox();
      if (u) return { x: box.left + u.valToPos(x, 'x'), y: box.top + u.valToPos(y, 'y') };
      const { w, h, v, k } = consGeom();
      return { x: box.left + w / 2 + (x - v.cx) * k, y: box.top + h / 2 - (y - v.cy) * k };
    },

    setX(min, max) {
      if (u && xs.length > 1) {
        const plotU = u;
        const [mn, mx] = clampX(Math.min(min, max), Math.max(min, max));
        plotU.batch(() => plotU.setScale('x', { min: mn, max: mx }));
      } else if (canvas) {
        const { w, h, v } = consGeom();
        const k = w / Math.max(Math.abs(max - min), 1e-12);
        consUser = { cx: (min + max) / 2, cy: v.cy, half: Math.min(w, h) / (2 * k) };
        drawCons();
        syncButtons();
      }
    },

    resetZoom() {
      if (u) {
        const plotU = u;
        const [mn, mx] = fullX();
        plotU.batch(() => plotU.setScale('x', { min: mn, max: mx }));
      } else if (canvas) {
        consUser = null;
        if (signal?.imag) consFit = fitCons(signal.real, signal.imag);
        drawCons();
      }
      syncButtons();
    },

    setAutoscale(on) {
      autoscale = on;
      if (on && u) {
        // Re-applying x re-ranges y now that its auto() returns true.
        const plotU = u;
        const [mn, mx] = userX ?? fullX();
        plotU.batch(() => plotU.setScale('x', { min: mn, max: mx }));
      } else if (on && canvas && signal?.imag) {
        consFit = fitCons(signal.real, signal.imag);
        drawCons();
      }
      syncButtons();
    },

    addMarker(x) {
      const index = nearest(xs, x);
      let m: Mark;
      if (markers.length < 2) {
        m = { id: markers.some((k) => k.id === 1) ? 2 : 1, index, order: ++order };
        markers.push(m);
      } else {
        m = markers[0].order < markers[1].order ? markers[0] : markers[1];
        m.index = index;
        m.order = ++order;
      }
      redrawNow();
      renderReadout();
      return markerList().find((k) => k.id === m.id)!;
    },

    peakSearch() {
      if (!u || !ys.length) return null;
      const values = ys[0];
      const i0 = lowerBound(xs, u.scales.x.min ?? -Infinity);
      const i1 = upperBound(xs, u.scales.x.max ?? Infinity);
      let best = -1;
      for (let i = i0; i <= Math.min(i1, values.length - 1); i++) {
        if (Number.isFinite(values[i]) && (best < 0 || values[i] > values[best])) best = i;
      }
      if (best < 0) return null;
      beforeClick = null;
      if (!markers.length) markers.push({ id: 1, index: best, order: ++order });
      else {
        const active = markers.reduce((a, b) => (b.order > a.order ? b : a));
        active.index = best;
        active.order = ++order;
      }
      redrawNow();
      renderReadout();
      const activeId = markers.reduce((a, b) => (b.order > a.order ? b : a)).id;
      return markerList().find((k) => k.id === activeId) ?? null;
    },

    clearMarkers() {
      markers = [];
      beforeClick = null;
      redrawNow();
      renderReadout();
    },

    destroy() {
      observer.disconnect();
      teardownRenderer();
      root.remove();
    },
  };

  syncButtons();
  return handle;
}
