// Repeatable browser check for the spike. Drives the page with real mouse input in
// headless Chrome and reports what worked.
//
//   node check.mjs <url> <screenshot-dir>            against a live model
//   node check.mjs <url> <screenshot-dir> replay     against a served recording
//   node check.mjs <url> <screenshot-dir> unbound    against the model that cannot run
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
// The page polls for as long as it is open, so the network never goes idle.
await page.goto(url, { waitUntil: 'load' });
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
const leaf = ['root/chan', 'root/stim', 'root/tune'].find((path) => paths.includes(path));
const plotTitles = () => page.evaluate(() => [...document.querySelectorAll('.blk-plot')].map((c) => c.title));
const origin = new URL(url).origin;
// The newest scope in the side panel: what it probes, what it shows, and where its plot is on screen.
const scope = (target = page, name = 'spike') =>
  target.evaluate((name) => {
    const s = name === 'spike' ? window.spike.scopes.at(-1) : { handle: window.popout.handle, meta: window.popout.state().meta };
    return { probe: s.probe, meta: s.meta, state: s.handle.state(), box: s.handle.plotBox() };
  }, name);
// NumPy's reduction of the same snapshot the page was given.
const reference = async (probe, view) => (await fetch(`${origin}/api/reference?${new URLSearchParams({ path: probe.path, port: probe.port, view })}`)).json();
const near = (a, b, tolerance) => Math.abs(a - b) <= tolerance * Math.max(1, Math.abs(b));
const popout = async (opened) => {
  const target = await opened;
  const pop = await target.page();
  pop.on('pageerror', (e) => problems.push(`pop-out pageerror: ${e.message}`));
  await pop.waitForFunction(() => window.popout && window.popout.state().meta, { timeout: 10000 });
  return pop;
};
// Poll until a condition on a page holds, so a check does not depend on one fixed delay.
const until = async (condition, ms = 2000) => {
  const deadline = Date.now() + ms;
  while (Date.now() < deadline) {
    if (await condition()) return true;
    await settle(50);
  }
  return false;
};

// A real click selects a block and fills the inspector, in either mode.
let p = await at(leaf, 30, 12);
await page.mouse.click(p.x, p.y);
await settle();
let s = await state();
report('click selects a block and fills the inspector', s.selection?.path === leaf && (await text('#inspector h2')).includes(leaf.split('/').pop()));

if (mode === 'unbound') {
  // A model whose external input nobody bound: it must load, say so, and report the failed run.
  report('a model with an unbound input loads and draws', paths.length > 1 && !(await text('#error')));
  await page.evaluate(() => document.querySelector('[data-act=probe][data-port=x][data-view=time]').click());
  await settle();
  report('probing the unbound input says no value is bound', /no value is bound/.test(await text('#scopes .stats')), await text('#scopes .stats'));
  await page.click('#step');
  await settle();
  s = await state();
  report('running it is reported as a model failure', s.position.state === 'faulted' && /Model failed: .*unset inputs/.test(await text('#error')), await text('#error'));
  await page.click('#reset');
  await settle();
  report('reset clears the failure', (await state()).position.state === 'idle');
  await shot(`${model}-unbound`);
  await browser.close();
  console.log(`${results.filter(Boolean).length} of ${results.length} checks passed`);
  process.exit(results.every(Boolean) ? 0 : 1);
}

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
  // A popped-out scope is the same client on the same path, so it works on a recording too.
  let sc = await scope();
  const opened = browser.waitForTarget((t) => t.url().includes('scope.html'));
  await page.click('#scopes .scope a.pop');
  const pop = await popout(opened);
  let ps = await pop.evaluate(() => window.popout.state());
  const stats = await pop.$eval('#stats', (e) => e.textContent);
  report('a popped-out scope shows the recorded tick the main page is on', ps.position.tick === s.position.tick && ps.meta.stamp.tick === sc.meta.stamp.tick && /recorded/.test(stats), stats);
  await page.$eval('#seek', (e) => {
    e.value = '2';
    e.dispatchEvent(new Event('input', { bubbles: true }));
  });
  const followed = await until(async () => (await pop.evaluate(() => window.popout.state().position.index)) === 2);
  ps = await pop.evaluate(() => window.popout.state());
  s = await state();
  report('seeking in the main page moves the popped-out scope', followed && ps.meta.stamp.tick === s.position.tick, `tick ${ps.meta.stamp.tick}`);
  await pop.screenshot({ path: `${outDir}/${model}-replay-popout.png` });
  await pop.close();
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

  // The scope is interactive, and everything it does is local: watch the network while using it.
  // The side panel scrolls; bring the scope under the mouse before using it.
  await page.evaluate(() => window.spike.scopes.at(-1).host.scrollIntoView({ block: 'center' }));
  await settle(150);
  let sc = await scope();
  const full = sc.state.x;
  const sent = [];
  const note = (request) => sent.push({ path: new URL(request.url()).pathname, body: request.postData() ?? '' });
  page.on('request', note);
  await settle(500);
  const idlePoll = sent.at(-1);
  sent.length = 0;
  const y = sc.box.top + sc.box.height / 2;
  await page.mouse.move(sc.box.left + sc.box.width * 0.3, y);
  await page.mouse.down();
  await page.mouse.move(sc.box.left + sc.box.width * 0.7, y, { steps: 8 });
  await page.mouse.up();
  await settle(150);
  sc = await scope();
  const zoomed = sc.state.x;
  await page.mouse.click(sc.box.left + sc.box.width * 0.4, y);
  await settle(150);
  const placed = (await scope()).state.markers.at(-1);
  const peakAt = await page.evaluate(() => {
    const button = [...window.spike.scopes.at(-1).host.querySelectorAll('button')].find((b) => /peak/i.test(b.textContent));
    const box = button.getBoundingClientRect();
    return { x: box.x + box.width / 2, y: box.y + box.height / 2 };
  });
  await page.mouse.click(peakAt.x, peakAt.y);
  await settle(300);
  page.off('request', note);
  sc = await scope();
  const peak = sc.state.markers.at(-1);
  report('dragging across a scope zooms it', sc.state.zoomed && zoomed[1] - zoomed[0] < 0.6 * (full[1] - full[0]), `${zoomed.map((v) => v.toPrecision(4)).join(' … ')} of ${full.map((v) => v.toPrecision(4)).join(' … ')}`);
  const foreign = sent.filter((r) => r.path !== '/api/observe' || r.body !== idlePoll?.body);
  report('zoom, marker and peak search send nothing to the server', !!idlePoll && foreign.length === 0, `${sent.length} requests in that time, all the unchanged idle poll`);
  // The readouts are compared with NumPy's spectrum of the same frame.
  let ref = await reference(sc.probe, 'spectrum');
  const sameFrame = ref.tick === sc.meta.stamp.tick && ref.count === sc.state.points;
  report(
    'a marker reads the value NumPy computes for that bin',
    sameFrame && !!placed && near(placed.y, ref.db[placed.index], 1e-6) && near(placed.x, ref.freq[placed.index], 1e-9),
    placed ? `bin ${placed.index}: ${placed.y.toFixed(6)} dB at ${placed.x} against ${ref.db[placed.index]?.toFixed(6)} at ${ref.freq[placed.index]}` : 'no marker',
  );
  let best = -1;
  ref.freq.forEach((f, i) => {
    if (f >= zoomed[0] && f <= zoomed[1] && (best < 0 || ref.db[i] > ref.db[best])) best = i;
  });
  report('peak search lands on the largest bin in view', !!peak && peak.index === best && near(peak.y, ref.db[best], 1e-6), `bin ${peak?.index}, ${peak?.y.toFixed(3)} dB`);
  const updatesBefore = sc.state.updates;
  for (let i = 0; i < 3; i++) {
    await page.click('#step');
    await settle(200);
  }
  sc = await scope();
  ref = await reference(sc.probe, 'spectrum');
  const kept = sc.state.markers.find((m) => m.id === peak?.id);
  report('the zoom survives new data', sc.state.updates >= updatesBefore + 3 && sc.state.x[0] === zoomed[0] && sc.state.x[1] === zoomed[1], `${sc.state.updates - updatesBefore} updates`);
  report('a marker keeps its bin and reads the new data', !!kept && kept.index === peak.index && near(kept.y, ref.db[kept.index], 1e-6), `${kept?.y.toFixed(3)} dB at tick ${ref.tick}`);
  await shot(`${model}-scope`);

  // Pop the scope out: its own window and subscription, the same execution state.
  const opened = browser.waitForTarget((t) => t.url().includes('scope.html'));
  await page.click('#scopes .scope a.pop');
  const pop = await popout(opened);
  s = await state();
  let ps = await scope(pop, 'popout');
  let position = await pop.evaluate(() => window.popout.state().position);
  report('a popped-out scope shows the same tick and frame as the main page', position.tick === s.position.tick && ps.meta.stamp.tick === sc.meta.stamp.tick && ps.state.points === sc.state.points, `tick ${position.tick}, ${ps.state.points} points`);
  await page.click('#step');
  const stepped = await until(async () => (await pop.evaluate(() => window.popout.state().position.tick)) === s.position.tick + 1);
  report('a step in the main page reaches the popped-out scope', stepped);
  ps = await scope(pop, 'popout');
  await page.click('#run');
  await settle(1500);
  const live = await scope(pop, 'popout');
  position = await pop.evaluate(() => window.popout.state().position);
  const mainTick = (await state()).position.tick;
  report('both windows follow a running model', position.state === 'running' && live.state.updates - ps.state.updates >= 10 && Math.abs(position.tick - mainTick) < 400, `${live.state.updates - ps.state.updates} updates in the pop-out; ticks ${position.tick} and ${mainTick}`);
  await page.click('#run');
  const settled = await until(async () => {
    const [a, b] = [(await state()).position, await pop.evaluate(() => window.popout.state().position)];
    return a.state === 'idle' && b.state === 'idle' && a.tick === b.tick;
  });
  report('both windows stop on the same tick', settled);
  await pop.screenshot({ path: `${outDir}/${model}-popout.png` });
  await pop.close();

  // The picture inside a block is a thumbnail; clicking it opens the interactive scope.
  const thumb = await page.evaluate(() => {
    const view = document.getElementById('canvas').getBoundingClientRect();
    for (const [path, label] of window.spike.labels) {
      const box = label.canvas?.getBoundingClientRect();
      if (box && box.width > 20 && box.x > view.x && box.right < view.right && box.y > view.y && box.bottom < view.bottom) return { path, x: box.x + box.width / 2, y: box.y + box.height / 2 };
    }
    return null;
  });
  if (thumb) {
    const count = await page.evaluate(() => window.spike.scopes.length);
    await page.mouse.click(thumb.x, thumb.y);
    await settle();
    const added = await page.evaluate(() => window.spike.scopes.at(-1).probe.path);
    report('clicking a plot inside a block opens its scope', (await page.evaluate(() => window.spike.scopes.length)) === count + 1 && added === thumb.path, thumb.path);
  } else {
    console.log('SKIP  thumbnail click (no in-block plot in view)');
  }

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
    await settle(400);
    // Both ends of the speed measurement are read while the model runs, so neither
    // includes the time a start or a stop takes to arrive.
    const first = { tick: (await state()).position.tick, at: Date.now() };
    await settle(2100);
    const last = { tick: (await state()).position.tick, at: Date.now() };
    const rateText = await text('#status');
    await page.click('#run');
    await settle(400);
    s = await state();
    rates[tracing ? 'traced' : 'untraced'] = Math.round(((last.tick - first.tick) * 1000) / (last.at - first.at));
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
