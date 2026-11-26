// Repeatable browser check for the spike. Drives the page with real mouse input in
// headless Chrome and reports what worked.  usage: node check.mjs <url> <screenshot-dir>
import puppeteer from 'puppeteer-core';

const [url, outDir] = process.argv.slice(2);
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
      return { x: box.x + state.x + (dx ?? state.width / 2), y: box.y + state.y + (dy ?? state.height / 2), w: state.width, h: state.height };
    },
    path,
    dx,
    dy,
  );
const state = () => page.evaluate(() => window.spike.state());
const paths = await page.evaluate(() => [...window.spike.cellByPath.keys()]);
const model = await page.$eval('#title', (e) => e.textContent);
const leaf = paths.includes('root/chan') ? 'root/chan' : 'root/stim';

// 1. A real click selects a block and fills the inspector.
let p = await at(leaf, 30, 12);
await page.mouse.click(p.x, p.y);
await settle();
let s = await state();
const heading = await page.$eval('#inspector h2', (e) => e.textContent).catch(() => '');
report('click selects a block and fills the inspector', s.selection?.path === leaf && heading.includes(leaf.split('/').pop()), heading);

// 2. Step advances the model; badges and timings appear from the observed trace.
const tickNumber = async () => Number((await page.$eval('#ticks', (e) => e.textContent)).replace(/\D/g, ''));
const ticksAtStart = await tickNumber();
await page.click('#step');
await settle();
await page.click('#step');
await settle();
const afterStep = await page.evaluate(() => ({
  ticks: document.getElementById('ticks').textContent,
  calls: document.getElementById('trace').children.length,
  time: document.querySelector('.blk-time')?.textContent,
}));
report('step runs the root once per press and shows observed calls', (await tickNumber()) === ticksAtStart + 2 && afterStep.calls > 3 && !!afterStep.time, JSON.stringify(afterStep));

// 3. A probe from the inspector adds a live scope and marks the port.
await page.evaluate(() => document.querySelector('[data-act=probe][data-view=spectrum]').click());
await settle();
const scope = await page.evaluate(() => ({ cards: document.querySelectorAll('#scopes .scope').length, stats: document.querySelector('#scopes .stats')?.textContent }));
report('probing a port adds a scope with live statistics', scope.cards === 1 && /peak/.test(scope.stats ?? ''), scope.stats);
await shot(`${model}-1-selected`);

// 4. Dragging a block moves it and its connections follow without errors.
p = await at(leaf, 30, 12);
await page.mouse.move(p.x, p.y);
await page.mouse.down();
await page.mouse.move(p.x + 40, p.y + 70, { steps: 8 });
await page.mouse.up();
await settle();
const moved = await at(leaf, 30, 12);
// The library keeps a child inside its container, so the block may stop short of the pointer.
report('dragging moves a block', moved.y - p.y > 20, `dx=${Math.round(moved.x - p.x)} dy=${Math.round(moved.y - p.y)}`);

// 5. Topology cannot be edited: Delete leaves every cell in place.
await page.keyboard.press('Delete');
await settle(150);
const still = await page.evaluate((n) => window.spike.cellByPath.size === n && !!window.spike.graph.view.getState(window.spike.cellByPath.get('root')), paths.length);
report('delete key does not remove cells', still);

// 6. Run for a while: ticks advance and per-tick UI cost stays small.
const before = await tickNumber();
await page.click('#run');
await settle(3000);
await page.click('#run');
await settle(300);
const run = await page.evaluate(() => ({ ticks: document.getElementById('ticks').textContent, ...window.spike.state() }));
const advanced = Number(run.ticks.replace(/\D/g, '')) - before;
report('run loop advances with live plots', advanced > 15, `${advanced} ticks in 3 s, UI ${run.uiCost.toFixed(1)} ms/tick, round trip ${run.roundTrip.toFixed(1)} ms`);
await shot(`${model}-2-running`);

// 7. Composite: toggle overlay collapses, double-click drills in, breadcrumb returns.
const composite = paths.find((path) => path !== 'root' && path.split('/').length === 2 && paths.some((q) => q.startsWith(`${path}/`)));
if (composite) {
  const count = paths.length;
  p = await at(composite, 14, 14);
  await page.mouse.click(p.x, p.y);
  await settle(700);
  const collapsedCount = await page.evaluate(() => window.spike.cellByPath.size);
  const portsKept = await page.evaluate((path) => [...window.spike.portCells.keys()].filter((k) => k.startsWith(`${path}::`)).every((k) => !!window.spike.graph.view.getState(window.spike.portCells.get(k))), composite);
  report('toggle overlay collapses a composite and keeps its ports', collapsedCount < count && portsKept, `${count} → ${collapsedCount} blocks shown`);
  await shot(`${model}-3-collapsed`);
  p = await at(composite);
  await page.mouse.click(p.x, p.y, { count: 2 });
  await settle(900);
  s = await state();
  report('double-click drills into a composite', s.currentRoot === composite, s.currentRoot);
  await shot(`${model}-4-inside`);
  await page.evaluate(() => document.querySelector('#crumbs button').click());
  await settle(700);
  s = await state();
  report('breadcrumb returns to the root', s.currentRoot === 'root', s.currentRoot);
} else {
  console.log('SKIP  composite checks (model has one level)');
}

// 8. A Control edited in the inspector reaches the model.
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
  await page.focus('input.ctl');
  await page.keyboard.down('Control');
  await page.keyboard.press('a');
  await page.keyboard.up('Control');
  const next = Number(control.before) === 0.5 ? '0.25' : '0.5';
  await page.keyboard.type(next);
  await page.keyboard.press('Tab');
  await settle(500);
  const live = await page.$eval('#live', (e) => e.textContent);
  report('editing a Control updates the running model', live.includes(`control ${control.name}${next}`), `${control.path}.${control.name}: ${control.before} → ${next}`);
}

// 9. Native folding, for the record: does it keep a composite's own ports visible?
const native = await page.evaluate(() => {
  const { graph, cellByPath, portCells } = window.spike;
  const path = [...cellByPath.keys()].find((k) => [...portCells.keys()].some((q) => q.startsWith(`${k}::`)) && [...cellByPath.keys()].some((q) => q.startsWith(`${k}/`)));
  if (!path) return null;
  graph.foldCells(true, false, [cellByPath.get(path)]);
  const ports = [...portCells.keys()].filter((k) => k.startsWith(`${path}::`));
  return { path, ports: ports.length, visible: ports.filter((k) => !!graph.view.getState(portCells.get(k))).length };
});
if (native) console.log(`INFO  native fold of ${native.path}: ${native.visible} of ${native.ports} own ports still drawn`);

const errorText = await page.$eval('#error', (e) => e.textContent);
report('no page errors', problems.length === 0 && !errorText, [...problems, errorText].filter(Boolean).join(' | '));
await browser.close();
console.log(`${results.filter(Boolean).length} of ${results.length} checks passed`);
process.exit(results.every(Boolean) ? 0 : 1);
