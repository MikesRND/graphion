// Headless check of the scope component through scope-demo.html, with real mouse
// input where it matters. Builds the demo, serves this folder on a random local
// port and drives headless Chrome.
//
//   node spikes/maxgraph/scope.check.mjs
import { createServer } from 'node:http';
import { readFile, mkdir } from 'node:fs/promises';
import { dirname, extname, join, normalize, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import * as esbuild from 'esbuild';
import puppeteer from 'puppeteer-core';

const here = dirname(fileURLToPath(import.meta.url));
await esbuild.build({
  entryPoints: [join(here, 'src/scope-demo.ts')],
  bundle: true,
  format: 'esm',
  sourcemap: true,
  outdir: join(here, 'dist'),
  logLevel: 'warning',
});

const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.map': 'application/json' };
const server = createServer(async (req, res) => {
  const rel = normalize(decodeURIComponent(new URL(req.url, 'http://local').pathname)).replace(/^[/\\]+/, '');
  const file = join(here, rel);
  if (file !== here && !file.startsWith(here + sep)) return res.writeHead(403).end();
  try {
    const body = await readFile(file);
    res.writeHead(200, { 'content-type': types[extname(file)] ?? 'application/octet-stream' }).end(body);
  } catch {
    res.writeHead(404).end();
  }
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const url = `http://127.0.0.1:${server.address().port}/scope-demo.html`;

const browser = await puppeteer.launch({
  executablePath: process.env.CHROME ?? '/usr/bin/google-chrome',
  headless: true,
  args: ['--no-sandbox', '--window-size=1500,1150'],
});
const page = await browser.newPage();
await page.setViewport({ width: 1500, height: 1150 });
const problems = [];
page.on('console', (m) => {
  if (m.type() === 'error') problems.push(m.text());
});
page.on('pageerror', (e) => problems.push(`pageerror: ${e.message}`));
await page.goto(url, { waitUntil: 'networkidle0' });
await page.waitForFunction(() => window.demo && window.demo.scopes.length === 5, { timeout: 10000 });
// Every request from here on is a failure: the scope must not talk to anything.
const requests = [];
page.on('request', (r) => requests.push(r.url()));

const settle = (ms = 80) => new Promise((r) => setTimeout(r, ms));
const results = [];
const report = (name, ok, detail = '') => {
  results.push(ok);
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? `  (${detail})` : ''}`);
};
const S = (i) => page.evaluate((i) => window.demo.scopes[i].state(), i);
const box = (i) => page.evaluate((i) => window.demo.scopes[i].plotBox(), i);
const call = (i, fn, ...args) => page.evaluate((i, fn, args) => window.demo.scopes[i][fn](...args), i, fn, args);
const feed = (i, s) => page.evaluate((i, s) => window.demo.feed(i, s), i, s);
const show = (i) => page.evaluate((i) => document.querySelectorAll('.gscope')[i].scrollIntoView({ block: 'nearest' }), i);
const buttonAt = (i, label) =>
  page.evaluate(
    (i, label) => {
      const b = [...document.querySelectorAll('.gscope')[i].querySelectorAll('.gscope-bar button')].find((e) => e.textContent.startsWith(label));
      const r = b.getBoundingClientRect();
      return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
    },
    i,
    label,
  );
const clickButton = async (i, label) => {
  const p = await buttonAt(i, label);
  await page.mouse.click(p.x, p.y);
  await settle();
};
async function runUpdates(i, n) {
  const start = (await S(i)).updates;
  await page.evaluate(() => window.demo.setRunning(true));
  await page.waitForFunction((i, t) => window.demo.scopes[i].state().updates >= t, { polling: 10, timeout: 5000 }, i, start + n);
  await page.evaluate(() => window.demo.setRunning(false));
  return (await S(i)).updates - start;
}
const span = (r) => r[1] - r[0];

// Deterministic test signals, built here so expected values come from the same arrays.
function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function tone({ n = 4096, rate = 48000, freq = 3000, amp = 1, noise = 0.05, complex = true, seed = 1 }) {
  const r = rng(seed);
  const real = [];
  const imag = [];
  for (let i = 0; i < n; i++) {
    const p = (2 * Math.PI * freq * i) / rate;
    real.push(amp * Math.cos(p) + noise * (r() - 0.5));
    imag.push(amp * Math.sin(p) + noise * (r() - 0.5));
  }
  return { real, imag: complex ? imag : null, rate };
}
// Same rule as scope.ts fmtHz: 5 significant digits, trailing zeros dropped.
function num(v, d = 5) {
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1e6 || a < 1e-4) return v.toExponential(d - 1).replace(/\.?0+e/, 'e');
  return String(+v.toPrecision(d));
}
const fmtHz = (v) => (Math.abs(v) >= 1e6 ? `${num(v / 1e6)} MHz` : Math.abs(v) >= 1e3 ? `${num(v / 1e3)} kHz` : `${num(v)} Hz`);
const close = (a, b, tol = 1e-9) => Math.abs(a - b) <= tol;

// 1. Everything renders; points match what was fed.
await page.evaluate(() => window.demo.setRunning(false));
await settle(100);
{
  const facts = await page.evaluate(() =>
    window.demo.scopes.map((s, i) => {
      const st = s.state();
      const b = s.plotBox();
      return { points: st.points, fed: window.demo.last[i].real.length, w: b.width, h: b.height, view: st.view };
    }),
  );
  const ok = facts.every((f) => f.points === f.fed && f.w > 100 && f.h > 100) && problems.length === 0;
  report('all scopes render and points match the fed signal', ok, facts.map((f) => `${f.view} ${f.points}/${f.fed} ${Math.round(f.w)}x${Math.round(f.h)}`).join(', '));
}

// 2. Real drag over the middle third of the spectrum zooms; 10 updates keep the range.
{
  await show(1);
  const before = await S(1);
  const b = await box(1);
  const y = b.top + b.height / 2;
  await page.mouse.move(b.left + b.width / 3, y);
  await page.mouse.down();
  await page.mouse.move(b.left + (2 * b.width) / 3, y, { steps: 12 });
  await page.mouse.up();
  await settle();
  const zoomed = await S(1);
  report('drag across the middle third zooms the spectrum', zoomed.zoomed && span(zoomed.x) < span(before.x) * 0.45, `${span(before.x).toFixed(0)} → ${span(zoomed.x).toFixed(0)} Hz`);
  const n = await runUpdates(1, 10);
  const after = await S(1);
  report('10 data updates leave the zoomed x range exactly unchanged', n >= 10 && after.x[0] === zoomed.x[0] && after.x[1] === zoomed.x[1] && after.zoomed, `${n} updates, ${after.x.map((v) => v.toFixed(3)).join('..')}`);
}

// 3. Autoscale follows amplitude; switched off with a real click, y stays exactly put.
{
  await show(0);
  await feed(0, tone({ amp: 1 }));
  const y1 = (await S(0)).y;
  await feed(0, tone({ amp: 3 }));
  const y3 = (await S(0)).y;
  report('autoscale on: y follows the amplitude', span(y3) > span(y1) * 2.5, `${span(y1).toFixed(2)} → ${span(y3).toFixed(2)}`);
  await clickButton(0, 'Auto');
  const off = await S(0);
  const ys = [];
  for (const amp of [1, 5, 0.2]) {
    await feed(0, tone({ amp, seed: 7 }));
    ys.push((await S(0)).y);
  }
  report('autoscale off (real click): y stays exactly the same across updates', !off.autoscale && ys.every((y) => y[0] === off.y[0] && y[1] === off.y[1]), `${off.y.map((v) => v.toFixed(3)).join('..')}`);
  await clickButton(0, 'Auto');
  report('Auto button turns autoscale back on', (await S(0)).autoscale);
}

// 4. A real click places a marker on the nearest sample, with the fed value.
const sigA = tone({ freq: 1234, amp: 0.8, noise: 0.3, seed: 11 });
let markerIndex = -1;
{
  await clickButton(0, 'Clear');
  await feed(0, sigA);
  await call(0, 'setX', 1000, 1040); // ~7 px per sample, so the nearest sample is unambiguous
  await settle();
  const st = await S(0);
  const b = await box(0);
  const target = await call(0, 'toClient', 1017, 0);
  const cx = Math.round(target.x) + 2;
  const cy = Math.round(b.top + b.height / 2);
  await page.mouse.click(cx, cy);
  await settle();
  const after = await S(0);
  const clickedValue = st.x[0] + ((cx - b.left) / b.width) * span(st.x);
  const nearestIdx = Math.round(clickedValue); // x is the sample index
  const m = after.markers[0];
  markerIndex = m?.index ?? -1;
  report(
    'real click places a marker at the nearest sample with the fed value',
    after.markers.length === 1 && m.index === nearestIdx && m.x === nearestIdx && close(m.y, sigA.real[nearestIdx]),
    m ? `index ${m.index} (expected ${nearestIdx}), y ${m.y} vs ${sigA.real[nearestIdx]}` : 'no marker',
  );
}

// 5. Peak search on the spectrum of a known tone.
const RATE = 48000;
const N = 4096;
const BIN = RATE / N;
const toneFreq = 341 * BIN; // on a bin: 3996.09375 Hz
{
  await show(1);
  await call(1, 'resetZoom');
  await clickButton(1, 'Clear');
  await feed(1, tone({ freq: toneFreq, complex: false, noise: 0.01, seed: 5 }));
  await clickButton(1, 'Peak');
  const st = await S(1);
  const m = st.markers[0];
  report(
    'peak search lands within one bin of the tone and the readout shows its frequency',
    st.markers.length === 1 && Math.abs(m.x - toneFreq) <= BIN && st.readout.includes(fmtHz(toneFreq)),
    m ? `${m.x} Hz vs ${toneFreq} Hz; readout "${st.readout}"` : 'no marker',
  );
}

// 6. A second marker (real click) gives the delta of the two readings.
{
  const b = await box(1);
  await page.mouse.click(Math.round(b.left + b.width * 0.25), Math.round(b.top + b.height / 2));
  await settle();
  const st = await S(1);
  const [m1, m2] = st.markers;
  report(
    'two markers give delta = M2 − M1',
    st.markers.length === 2 && st.delta && st.delta.dx === m2.x - m1.x && st.delta.dy === m2.y - m1.y,
    st.delta ? `Δx ${st.delta.dx}, Δy ${st.delta.dy.toFixed(3)}` : 'no delta',
  );
  await page.mouse.move(5, 5); // leave the plot so the readout shows the markers
  await settle();
  const text = (await S(1)).readout;
  report('the readout shows both markers and the delta', text.includes('M1') && text.includes('M2') && text.includes('Δ'), text);
}

// 7. Marker y follows new data at the same index.
{
  const sigB = tone({ freq: 777, amp: 1.7, noise: 0.4, seed: 23 });
  await feed(0, sigB);
  const st = await S(0);
  const m = st.markers[0];
  report('marker y follows the data: same index, new sample value', m && m.index === markerIndex && close(m.y, sigB.real[markerIndex]), m ? `index ${m.index}, y ${m.y} vs ${sigB.real[markerIndex]}` : 'no marker');
}

// 8. Wheel zoom about the cursor, shift-drag and middle-drag pan, double-click reset.
{
  await show(0);
  await call(0, 'resetZoom');
  await settle();
  const b = await box(0);
  const px = Math.round(b.left + b.width * 0.3);
  const py = Math.round(b.top + b.height / 2);
  const full = await S(0);
  const valueAt = (st, x) => st.x[0] + ((x - b.left) / b.width) * span(st.x);
  const c0 = valueAt(full, px);
  await page.mouse.move(px, py);
  await page.mouse.wheel({ deltaY: -600 });
  await settle();
  const z = await S(0);
  const c1 = valueAt(z, px);
  report('wheel zooms x about the cursor', z.zoomed && span(z.x) < span(full.x) * 0.5 && Math.abs(c1 - c0) < span(full.x) * 0.005, `span ${span(full.x)} → ${span(z.x).toFixed(1)}, value under cursor ${c0.toFixed(1)} → ${c1.toFixed(1)}`);

  const dx = -50;
  const expectShift = (-dx / b.width) * span(z.x);
  await page.keyboard.down('Shift');
  await page.mouse.move(px, py);
  await page.mouse.down();
  await page.mouse.move(px + dx, py, { steps: 8 });
  await page.mouse.up();
  await page.keyboard.up('Shift');
  await settle();
  const p1 = await S(0);
  const shift1 = p1.x[0] - z.x[0];
  report('shift-drag pans x (drag left shows later samples)', Math.abs(shift1 - expectShift) < span(z.x) * 0.01 && Math.abs(span(p1.x) - span(z.x)) < 1e-6 && p1.markers.length === 1, `shift ${shift1.toFixed(2)} expected ${expectShift.toFixed(2)}, markers ${p1.markers.length}`);

  await page.mouse.move(px, py);
  await page.mouse.down({ button: 'middle' });
  await page.mouse.move(px - dx, py, { steps: 8 });
  await page.mouse.up({ button: 'middle' });
  await settle();
  const p2 = await S(0);
  report('middle-drag pans x back', Math.abs(p2.x[0] - z.x[0]) < span(z.x) * 0.01, `min ${z.x[0].toFixed(2)} → ${p1.x[0].toFixed(2)} → ${p2.x[0].toFixed(2)}`);

  await page.mouse.click(px, py, { count: 2 });
  await settle();
  const r = await S(0);
  report('double-click resets the zoom and leaves the markers alone', !r.zoomed && r.x[0] === full.x[0] && r.x[1] === full.x[1] && r.markers.length === 1 && r.markers[0].index === markerIndex, `x ${r.x.join('..')}, markers ${r.markers.map((m) => m.index)}`);
}

// 9. Constellation: wheel zoom survives updates, double-click resets.
{
  await show(3);
  await settle();
  const b = await box(3);
  const before = await S(3);
  await page.mouse.move(Math.round(b.left + b.width * 0.6), Math.round(b.top + b.height * 0.4));
  await page.mouse.wheel({ deltaY: -400 });
  await settle();
  const z = await S(3);
  const n = await runUpdates(3, 10);
  const after = await S(3);
  report('constellation: wheel zoom narrows the view', z.zoomed && span(z.x) < span(before.x) * 0.7 && span(z.y) < span(before.y) * 0.7, `x span ${span(before.x).toFixed(3)} → ${span(z.x).toFixed(3)}`);
  report('constellation: the zoom survives 10 updates', n >= 10 && after.x[0] === z.x[0] && after.x[1] === z.x[1] && after.y[0] === z.y[0] && after.y[1] === z.y[1], `${n} updates`);
  await page.mouse.click(Math.round(b.left + b.width / 2), Math.round(b.top + b.height / 2), { count: 2 });
  await settle();
  const r = await S(3);
  report('constellation: double-click resets', !r.zoomed && span(r.x) > span(z.x) * 1.3, `x span ${span(r.x).toFixed(3)}`);
  await feed(3, tone({ complex: false }));
  const msgText = await page.evaluate(() => {
    const m = document.querySelectorAll('.gscope')[3].querySelector('.gscope-msg');
    return m.hidden ? '' : m.textContent;
  });
  report('constellation: a real-only signal shows a message', msgText.length > 0, msgText);
  await page.evaluate(() => window.demo.tick());
}

// 10. No network traffic during the interactive checks.
report('no network requests after load', requests.length === 0, requests.slice(0, 3).join(', '));

// 11. The tall (non-compact) scope follows the window size.
{
  const b0 = await box(4);
  await page.setViewport({ width: 1200, height: 850 });
  await settle(300);
  const b1 = await box(4);
  await page.setViewport({ width: 1500, height: 1150 });
  await settle(300);
  const b2 = await box(4);
  report('resizing the window resizes the tall scope', b0.width - b1.width > 250 && b0.height - b1.height > 250 && Math.abs(b2.width - b0.width) < 2 && Math.abs(b2.height - b0.height) < 2, `${Math.round(b0.width)}x${Math.round(b0.height)} → ${Math.round(b1.width)}x${Math.round(b1.height)} → ${Math.round(b2.width)}x${Math.round(b2.height)}`);
}

// Extra: switching the tall scope through every view keeps drawing the last signal.
{
  const seen = [];
  for (const v of ['spectrum', 'array', 'constellation', 'time']) {
    await call(4, 'setView', v);
    await page.evaluate(() => window.demo.tick());
    await settle(50);
    const st = await S(4);
    const b = await box(4);
    seen.push(`${st.view} ${st.points} ${Math.round(b.width)}x${Math.round(b.height)}`);
  }
  report('setView switches through all views', seen.every((s, i) => s.startsWith(['spectrum', 'array', 'constellation', 'time'][i]) && / 4096 /.test(s)) && problems.length === 0, seen.join(', '));
}

// 12. Update cost while running at 30 Hz.
{
  await page.evaluate(() => window.demo.costs.forEach((c) => (c.length = 0)));
  await page.evaluate(() => window.demo.setRunning(true));
  await settle(3000);
  await page.evaluate(() => window.demo.setRunning(false));
  const costs = await page.evaluate(() => window.demo.costs);
  const median = (a) => {
    const s = [...a].sort((x, y) => x - y);
    return s.length ? s[s.length >> 1] : NaN;
  };
  const names = ['time 4096 cplx', 'spectrum 4096 cplx', 'array 512', 'constellation 1024', 'tall time 4096'];
  console.log(`INFO  median lastUpdateMs over ${costs[0].length} updates: ${costs.map((c, i) => `${names[i]} ${median(c).toFixed(2)} ms (max ${Math.max(...c).toFixed(2)})`).join(', ')}`);
}

// Screenshot with markers on the time and spectrum scopes.
{
  await page.evaluate(() => window.demo.tick());
  await call(1, 'peakSearch');
  await call(0, 'setX', 0, 400);
  await call(0, 'addMarker', 100);
  await call(0, 'addMarker', 125);
  await call(4, 'peakSearch');
  await page.mouse.move(5, 5);
  await settle(200);
  await mkdir(join(here, 'screenshots'), { recursive: true });
  await page.screenshot({ path: join(here, 'screenshots/scope-demo.png') });
}

report('no page errors', problems.length === 0, problems.join(' | '));
await browser.close();
server.close();
console.log(`${results.filter(Boolean).length} of ${results.length} checks passed`);
process.exit(results.every(Boolean) ? 0 : 1);
