// A scope in its own window or tab. The URL names what it shows:
//
//   scope.html?path=root/dut&port=y0&view=spectrum&channel=0
//
// It is a client like the diagram page: its own watch list and its own polls,
// on the execution state every window shares. It never changes that state.

import { Client, type Watch } from './client';
import { frameNote, frameText, isStale, signalFor } from './frames';
import { createScope } from './scope';
import type { FrameMeta, ModelResponse, Position, Probe, View } from './types';

const VIEWS: View[] = ['time', 'spectrum', 'constellation', 'array'];
const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const pause = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

const params = new URLSearchParams(location.search);
const probe: Probe = { id: 'scope', path: params.get('path') ?? '', port: params.get('port') ?? '', channel: Number(params.get('channel') ?? 0) };
let view = (VIEWS.includes(params.get('view') as View) ? params.get('view') : 'time') as View;
const watch: Watch = { probes: [probe], inspect: null };

let model: ModelResponse;
let position: Position | null = null;
let meta: FrameMeta | undefined;
let asking = false;
const client = new Client(() => void poll());
const handle = createScope($('host'), { view });

function showStatus() {
  if (!position) return;
  const where = position.state === 'paused' && position.at ? `paused before ${position.at.path}` : position.state;
  $('status').textContent = `tick ${position.tick.toLocaleString('en-US')} · ${where}`;
}

async function poll() {
  if (asking) return;
  asking = true;
  try {
    const observation = await client.observe(watch);
    if (!observation) return;
    position = observation.position;
    showStatus();
    meta = observation.frames[probe.id];
    // A port asked for while the model runs arrives with a later snapshot.
    if (!meta) return;
    const signal = signalFor(observation, probe, model);
    handle.update(signal, signal ? undefined : frameNote(meta));
    $('stats').textContent = frameText(meta, signal, position);
    $('host').style.opacity = isStale(meta, position) ? '0.45' : '1';
    $('error').textContent = '';
  } catch (error) {
    $('error').textContent = error instanceof Error ? error.message : String(error);
  } finally {
    asking = false;
  }
}

function showView() {
  for (const button of document.querySelectorAll<HTMLElement>('[data-view]')) button.classList.toggle('on', button.dataset.view === view);
  document.title = `${probe.port} · ${view}`;
}

$('views').addEventListener('click', (event) => {
  const next = (event.target as HTMLElement).dataset.view as View | undefined;
  if (!next || next === view) return;
  view = next;
  handle.setView(view);
  // The address bar stays a link to exactly what is shown.
  params.set('view', view);
  history.replaceState(null, '', `?${params}`);
  showView();
});

async function start() {
  model = await client.document();
  const known = model.facts[probe.path]?.ports[probe.port];
  $('name').textContent = `${probe.path}.${probe.port}`;
  $('model').textContent = `${model.capabilities.includes('execution.seek') ? 'recording of' : 'live'} ${model.model}`;
  $('views').innerHTML = VIEWS.map((v) => `<button data-view="${v}">${v}</button>`).join('');
  showView();
  if (!known) $('error').textContent = 'no such port in this model';
  // Exposed for the spike's own browser checks.
  Object.assign(window, { popout: { handle, client, state: () => ({ position, meta, view, serial: client.serial }) } });
  for (;;) {
    // A hidden tab stops asking, so its lease lapses and it costs the model nothing.
    if (!document.hidden) await poll();
    await pause(position?.state === 'running' ? 33 : 150);
  }
}

start().catch((error) => {
  $('error').textContent = error instanceof Error ? error.message : String(error);
});
