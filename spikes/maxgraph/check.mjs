// Repeatable browser check for the spike. Drives the page with real mouse input in
// headless Chrome and reports what worked.
//
//   node check.mjs <url> <screenshot-dir>            against a live model
//   node check.mjs <url> <screenshot-dir> replay     against a served recording
import puppeteer from 'puppeteer-core';

const [url, outDir, mode = 'live'] = process.argv.slice(2);
const browser = await puppeteer.launch({
  executablePath: process.env.CHROME ?? '/usr/bin/google-chrome',
  headless: true,
  args: ['--no-sandbox', '--window-size=1700,1000'],
});
const page = await browser.newPage();
await page.setViewport({ width: 1700, height: 1000 });
const problems = [];
page.on('console', (m) => {
  if (m.type() === 'error' && !m.text().includes('404')) problems.push(m.text());
});
page.on('pageerror', (e) => problems.push(`pageerror: ${e.message}`));
await page.goto(url, { waitUntil: 'networkidle0' });
await page.waitForFunction(() => window.spike && window.spike.cellByPath.size > 0, { timeout: 15000 });

const settle = (ms = 350) => new Promise((r) => setTimeout(r, ms));
const shot = (name) => page.screenshot({ path: `${outDir}/${name}.png` });
const results = [];
const report = (name, ok, detail = '') => {
  results.push(ok);
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? `  (${detail})` : ''}`);
};
// Page position of a cell's centre, or of a point offset from its top-left corner.
const at = (path, dx = null, dy = null) =>
  page.evaluate(
    (path, dx, dy) => {
      const { graph, cellByPath } = window.spike;
      const state = graph.view.getState(cellByPath.get(path));
      const box = document.getElementById('canvas').getBoundingClientRect();
      return { x: box.x + state.x + (dx ?? state.width / 2), y: box.y + state.y + (dy ?? state.height / 2) };
    },
    path,
    dx,
    dy,
  );
const state = () => page.evaluate(() => window.spike.state());
const text = (selector) => page.$eval(selector, (e) => e.textContent).catch(() => '');
const hidden = (selector) => page.$eval(selector, (e) => e.hidden);
const paths = await page.evaluate(() => [...window.spike.cellByPath.keys()]);
const model = await text('#title');
const leaf = paths.includes('root/chan') ? 'root/chan' : 'root/stim';
const plotTitles = () => page.evaluate(() => [...document.querySelectorAll('.blk-plot')].map((c) => c.title));

// A real click selects a block and fills the inspector, in either mode.
let p = await at(leaf, 30, 12);
await page.mouse.click(p.x, p.y);
await settle();
let s = await state();
report('click selects a block and fills the inspector', s.selection?.path === leaf && (await text('#inspector h2')).includes(leaf.split('/').pop()));

if (mode === 'replay') {
  report('a recording is labelled and offers no live controls', (await text('#subtitle')).startsWith('recording') && (await hidden('#stepcall')) && (await hidden('#record')) && (await hidden('#tracebtn')) && !(await hidden('#seek')));
  report('a recording shows no editable Controls', (await page.$$('input.ctl')).length === 0);
  await page.click('#step');
  await settle();
  await page.click('#step');
  await settle();
  s = await state();
  const titles = await plotTitles();
  report('stepping a recording draws its plots from recorded data', s.position.index === 2 && titles.length > 0 && titles.every((t) => t.includes('recorded')), titles[0]);
  report('a recording carries the observed calls', (await page.$$('#trace div')).length > 3);
  const last = s.position.total;
  await page.$eval('#seek', (e, v) => {
    e.value = String(v);
    e.dispatchEvent(new Event('input', { bubbles: true }));
  }, last - 1);
  await settle();
  s = await state();
  report('the scrubber seeks to a recorded tick', s.position.index === last - 1, `${s.position.index} of ${last}`);
  await page.click('#run');
  await settle(900);
  s = await state();
  report('playing a recording stops at its end', s.position.state === 'ended' && s.position.index === last);
  await page.evaluate(() => document.querySelector('[data-act=probe][data-view=spectrum]').click());
  await settle();
  report('a new probe works on recorded data', /recorded/.test(await text('#scopes .stats')), await text('#scopes .stats'));
  await shot(`${model}-replay`);
} else {
  // Start from a known state; an earlier run may have left tracing on.
  if ((await state()).position.trace) {
    await page.click('#tracebtn');
    await settle();
  }
  // With tracing off the model runs untouched: no calls are shown.
  await page.click('#step');
  await settle();
  report('tracing off shows no observed calls', (await page.$$('#trace div')).length === 0 && !(await state()).position.trace);

  // Call tracing on: badges, timings and the call strip appear.
  await page.click('#tracebtn');
  await settle();
  const before = (await state()).position.tick;
  await page.click('#step');
  await settle();
  s = await state();
  const calls = (await page.$$('#trace div')).length;
  report('call tracing shows observed calls and timings', s.position.tick === before + 1 && calls > 3 && !!(await text('.blk-time')), `${calls} calls`);

  // Step by block call: the model pauses before a call and stale data is marked.
  await page.click('#stepcall');
  await settle();
  await page.click('#stepcall');
  await settle();
  s = await state();
  const pausedTitles = await plotTitles();
  report('step call pauses before a block call', s.position.state === 'paused' && !!s.position.at && (await text('#status')).startsWith('paused before'), await text('#status'));
  report('a paused tick marks data that has not been produced yet as stale', pausedTitles.some((t) => t.includes('stale')), pausedTitles.find((t) => t.includes('stale')));
  await shot(`${model}-paused`);
  await page.click('#step');
  await settle();
  s = await state();
  report('step finishes the paused tick', s.position.state === 'idle' && s.position.tick === before + 2);

  // A probe from the inspector adds a live scope.
  await page.evaluate(() => document.querySelector('[data-act=probe][data-view=spectrum]').click());
  await settle();
  report('probing a port adds a scope with live statistics', /peak .* tick \d+ observed/.test(await text('#scopes .stats')), await text('#scopes .stats'));

  // Dragging moves a block; the library keeps it inside its container.
  p = await at(leaf, 30, 12);
  await page.mouse.move(p.x, p.y);
  await page.mouse.down();
  await page.mouse.move(p.x + 40, p.y + 70, { steps: 8 });
  await page.mouse.up();
  await settle();
  const moved = await at(leaf, 30, 12);
  report('dragging moves a block', moved.y - p.y > 20, `dx=${Math.round(moved.x - p.x)} dy=${Math.round(moved.y - p.y)}`);
  await page.keyboard.press('Delete');
  await settle(150);
  report('delete key does not remove cells', await page.evaluate((n) => window.spike.cellByPath.size === n, paths.length));

  // Free run: the model runs on its own thread while the page watches.
  const rates = {};
  for (const tracing of [true, false]) {
    if ((await state()).position.trace !== tracing) {
      await page.click('#tracebtn');
      await settle();
    }
    const from = (await state()).position.tick;
    await page.click('#run');
    const began = Date.now();
    await settle(2500);
    const rateText = await text('#status');
    await page.click('#run');
    const seconds = (Date.now() - began) / 1000;
    await settle(400);
    s = await state();
    rates[tracing ? 'traced' : 'untraced'] = Math.round((s.position.tick - from) / seconds);
    if (tracing) await shot(`${model}-running`);
    report(`free run with tracing ${tracing ? 'on' : 'off'} keeps running while watched`, s.position.tick - from > 100 && s.position.state === 'idle', `${rateText}, UI ${s.uiCost.toFixed(1)} ms per poll`);
  }
  console.log(`INFO  model speed with the page attached: about ${rates.untraced} ticks/s untraced, ${rates.traced} ticks/s traced`);

  // Composite: toggle overlay collapses, double-click drills in, breadcrumb returns.
  const composite = paths.find((path) => path !== 'root' && path.split('/').length === 2 && paths.some((q) => q.startsWith(`${path}/`)));
  if (composite) {
    p = await at(composite, 14, 14);
    await page.mouse.click(p.x, p.y);
    await settle(700);
    const shown = await page.evaluate(() => window.spike.cellByPath.size);
    const portsKept = await page.evaluate((path) => [...window.spike.portCells.keys()].filter((k) => k.startsWith(`${path}::`)).every((k) => !!window.spike.graph.view.getState(window.spike.portCells.get(k))), composite);
    report('toggle overlay collapses a composite and keeps its ports', shown < paths.length && portsKept, `${paths.length} → ${shown} blocks shown`);
    p = await at(composite);
    await page.mouse.click(p.x, p.y, { count: 2 });
    await settle(900);
    report('double-click drills into a composite', (await state()).currentRoot === composite);
    await page.evaluate(() => document.querySelector('#crumbs button').click());
    await settle(700);
    report('breadcrumb returns to the root', (await state()).currentRoot === 'root');
  } else {
    console.log('SKIP  composite checks (model has one level)');
  }

  // A Control edited in the inspector reaches the model.
  const control = await page.evaluate(async () => {
    for (const path of window.spike.cellByPath.keys()) {
      window.spike.select({ kind: 'block', path });
      await new Promise((r) => setTimeout(r, 120));
      const input = document.querySelector('input.ctl');
      if (input) return { path, name: input.dataset.name, before: input.value };
    }
    return null;
  });
  if (control) {
    const next = Number(control.before) === 0.5 ? '0.25' : '0.5';
    await page.focus('input.ctl');
    await page.keyboard.down('Control');
    await page.keyboard.press('a');
    await page.keyboard.up('Control');
    await page.keyboard.type(next);
    await page.keyboard.press('Tab');
    await settle(500);
    report('editing a Control updates the running model', (await text('#live')).includes(`control ${control.name}${next}`), `${control.path}.${control.name}: ${control.before} → ${next}`);
  }

  // Record a few ticks with tracing on, so the recording carries the calls too.
  // The path is printed so a replay server can be pointed at it.
  if (!(await state()).position.trace) {
    await page.click('#tracebtn');
    await settle();
  }
  await page.click('#record');
  await settle();
  for (let i = 0; i < 6; i++) {
    await page.click('#step');
    await settle(150);
  }
  const recording = (await state()).position.recording;
  await page.click('#record');
  await settle();
  report('recording captures every stepped tick', recording?.ticks === 6 && (await text('#status')).startsWith('saved 6 ticks'), recording?.path);
  if (recording) console.log(`RECORDING ${recording.path}`);
  await page.click('#tracebtn');
  await settle();
}

const errorText = await text('#error');
report('no page errors', problems.length === 0 && !errorText, [...problems, errorText].filter(Boolean).join(' | '));
await browser.close();
console.log(`${results.filter(Boolean).length} of ${results.length} checks passed`);
process.exit(results.every(Boolean) ? 0 : 1);
