// Spike: a live, inspectable Graphion diagram on maxGraph.
//
// The diagram is rebuilt from the description whenever what is shown changes
// (collapse, in-block plots). Topology is never edited here; code owns it.

import '@maxgraph/core/css/common.css';
import {
  Cell,
  CellHighlight,
  CellOverlay,
  type CellStyle,
  type EventObject,
  type FitPlugin,
  Graph,
  ImageBox,
  InternalEvent,
  PanningHandler,
  Point,
  TooltipHandler,
} from '@maxgraph/core';
import { Client, type Watch } from './client';
import { dtypeName, frameNote, frameText, isStale, signalFor } from './frames';
import { type LaidNode, layoutModel, PORT, PORT_PITCH, PORT_TOP, portsOf } from './layout';
import { drawLines, drawThumb } from './plots';
import { createScope, type ScopeHandle } from './scope';
import type {
  BlockRecord,
  Connection,
  Declaration,
  DimRecord,
  FrameMeta,
  LiveBlock,
  LiveValue,
  ModelResponse,
  Observation,
  Position,
  Probe,
  TraceRecord,
  ValueRecord,
  View,
} from './types';

type CellInfo =
  | { kind: 'block'; path: string }
  | { kind: 'port'; path: string; port: string }
  | { kind: 'edge'; index: number; text?: string };

interface BlockLabel {
  root: HTMLElement;
  time: HTMLElement;
  bar: HTMLElement;
  canvas: HTMLCanvasElement | null;
}

interface Scope {
  probe: Probe;
  view: View;
  handle: ScopeHandle;
  host: HTMLElement;
  stats: HTMLElement;
  meta?: FrameMeta;
}

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const esc = (text: unknown) =>
  String(text).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]!);

let model: ModelResponse;
let blocks = new Map<string, BlockRecord>();
const collapsed = new Set<string>();
const inBlock = new Map<string, { port: string; view: View }>();
const scopes: Scope[] = [];
const cellByPath = new Map<string, Cell>();
const portCells = new Map<string, Cell>();
const labels = new Map<string, BlockLabel>();
const badges = new Map<string, { text: string; overlay: CellOverlay }>();
let currentRoot = 'root';
let selection: CellInfo | null = null;
let lastTrace: TraceRecord[] = [];
let lastLive: LiveBlock | null = null;
let position: Position = { tick: 0, state: 'idle', at: null, error: null, trace: false, total: null, ticks_per_s: null, recording: null };
let capabilities = new Set<string>();
// The adapter says what it can do; the page offers nothing it did not advertise.
const can = (name: string) => capabilities.has(name);
let running = false;
// A recording has no thread of its own; while playing, this page steps it.
let playing = false;
let showSignalLabels = true;
let scopeCounter = 0;

// ---------------------------------------------------------------- formatting

function valueText(record: ValueRecord | null): string {
  if (!record) return '';
  switch (record.kind) {
    case 'literal':
      return typeof record.value === 'object' ? JSON.stringify(record.value) : String(record.value);
    case 'fraction':
      return `${record.numerator}/${record.denominator}`;
    case 'summary':
      return `ndarray ${dtypeName(record.dtype)} [${record.shape.join(' × ')}]`;
    case 'required':
      return 'required';
    default:
      return `opaque ${record.type}`;
  }
}

function dimText(dim: DimRecord): string {
  switch (dim.kind) {
    case 'resolved':
      return String(dim.value);
    case 'fraction':
      return dim.denominator === 1 ? String(dim.numerator) : `${dim.numerator}/${dim.denominator}`;
    case 'expression':
      return `${dimText(dim.left)}${dim.operator === 'multiply' ? '·' : '/'}${dimText(dim.right)}`;
    default:
      return dim.name;
  }
}

function dtypeText(decl: Declaration): string {
  if (!decl.dtype) return '';
  return decl.dtype.kind === 'resolved' ? dtypeName(decl.dtype.value) : decl.dtype.kind === 'symbol' ? decl.dtype.name : 'invalid';
}

const shapeText = (dims: DimRecord[] | null) => (dims ? `[${dims.map(dimText).join(' × ')}]` : '');

function hz(value: number): string {
  if (value >= 1e6) return `${+(value / 1e6).toPrecision(4)} MHz`;
  if (value >= 1e3) return `${+(value / 1e3).toPrecision(4)} kHz`;
  return `${+value.toPrecision(4)} Hz`;
}

const bytes = (n: number) => (n >= 1 << 20 ? `${(n / (1 << 20)).toFixed(1)} MiB` : n >= 1024 ? `${(n / 1024).toFixed(1)} KiB` : `${n} B`);
const micros = (us: number) => (us >= 1000 ? `${(us / 1000).toFixed(2)} ms` : `${us.toFixed(0)} µs`);
const shortName = (path: string) => path.split('/').pop()!;

function declaration(path: string, name: string): Declaration | undefined {
  return blocks.get(path)?.declarations.find((d) => d.name === name);
}

function signalText(path: string, port: string): string {
  const decl = declaration(path, port);
  if (!decl) return '';
  const rate = model.facts[path]?.ports[port]?.rate;
  return `${dtypeText(decl)} ${shapeText(decl.shape ?? decl.declared_shape)}${rate ? ` @ ${hz(rate.hz)}` : ''}`;
}

// Compact two-line form for the diagram: a long label on every wire makes the whole layout wide.
function signalLabel(path: string, port: string): string {
  const decl = declaration(path, port);
  if (!decl) return '';
  const short = dtypeText(decl).replace('complex', 'c').replace('float', 'f').replace('uint', 'u').replace('int', 'i');
  const dims = (decl.shape ?? decl.declared_shape ?? []).map(dimText).join('×');
  const rate = model.facts[path]?.ports[port]?.rate;
  return `${short} [${dims}]${rate ? `\n${hz(rate.hz)}` : ''}`;
}

function signalColor(path: string, port: string): string {
  const decl = declaration(path, port);
  const code = decl?.dtype?.kind === 'resolved' ? decl.dtype.value : '';
  if (code.includes('c')) return '#7048e8';
  if (code.includes('f')) return '#1971c2';
  return '#5c6b7f';
}

// ------------------------------------------------------------------- server

async function api<T>(path: string): Promise<T> {
  const payload = await (await fetch(path)).json();
  if (payload.error) throw new Error(payload.error);
  return payload as T;
}

// What this page wants in each snapshot: its scopes, the plots inside blocks, and the inspected block.
function watchNow(): Watch {
  const probes: Probe[] = scopes.map((s) => s.probe);
  for (const [path, hint] of inBlock) {
    if (labels.get(path)?.canvas) probes.push({ id: `blk:${path}`, path, port: hint.port, channel: 0 });
  }
  return { probes, inspect: selection && selection.kind !== 'edge' ? selection.path : null };
}

// -------------------------------------------------------------------- graph

const container = $('canvas');
const graph = new Graph(container);
graph.setPanning(true);
graph.setTooltips(true);
graph.setConnectable(false);
graph.setCellsEditable(false);
graph.setCellsResizable(false);
graph.setCellsCloneable(false);
graph.setCellsDeletable(false);
graph.setCellsDisconnectable(false);
graph.setCellsBendable(false);
graph.setDropEnabled(false);
graph.setEdgeLabelsMovable(false);
graph.setVertexLabelsMovable(false);
graph.setAllowDanglingEdges(false);
graph.keepEdgesInBackground = false;
const panning = graph.getPlugin<PanningHandler>(PanningHandler.pluginId);
if (panning) panning.useLeftButtonForPanning = true;

const info = (cell: Cell | null | undefined) => (cell?.value ?? null) as CellInfo | null;

// Collapse is a rebuild with a different layout rather than native folding, which
// hides child cells and would take a composite's own ports with it.
graph.isCellFoldable = () => false;
graph.isCellMovable = (cell: Cell) => info(cell)?.kind === 'block';
graph.isHtmlLabel = (cell: Cell) => {
  const value = info(cell);
  return value?.kind === 'block' && labels.has(value.path);
};
graph.getLabel = (cell?: Cell | null) => {
  const value = info(cell);
  if (!value) return '';
  if (value.kind === 'port') return value.port;
  if (value.kind === 'edge') return value.text ?? '';
  const label = labels.get(value.path);
  // A DOM node label keeps a live canvas inside the block; the typings only admit strings.
  if (label) return label.root as unknown as string;
  return `${shortName(value.path)} · ${blocks.get(value.path)!.definition.qualname}`;
};
// Tooltips belong to a plugin in maxGraph, where mxGraph had them on the graph.
const tooltips = graph.getPlugin<TooltipHandler>(TooltipHandler.pluginId);
const tooltipFor = (cell: Cell) => {
  const value = info(cell);
  if (!value) return '';
  if (value.kind === 'edge') {
    const c = model.description.connections[value.index];
    return `${c.source.path}.${c.source.port} → ${c.destination.path}.${c.destination.port}<br>${esc(signalText(c.source.path, c.source.port))}`;
  }
  if (value.kind === 'port') return `${value.path}.${value.port}<br>${esc(signalText(value.path, value.port))}`;
  const block = blocks.get(value.path)!;
  const doc = model.facts[value.path]?.doc;
  return `<b>${esc(value.path)}</b><br>${esc(block.definition.module)}.${esc(block.definition.qualname)}${doc ? `<br>${esc(doc)}` : ''}`;
};
if (tooltips) tooltips.getTooltipForCell = tooltipFor;

function svgImage(svg: string, width: number, height: number): ImageBox {
  return new ImageBox(`data:image/svg+xml,${encodeURIComponent(svg)}`, width, height);
}

function badgeImage(text: string): ImageBox {
  const width = 12 + text.length * 6;
  return svgImage(
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="16"><rect width="${width}" height="16" rx="8" fill="#0f8a6c"/>` +
      `<text x="${width / 2}" y="11.5" font-family="system-ui,sans-serif" font-size="10" fill="#fff" text-anchor="middle">${esc(text)}</text></svg>`,
    width,
    16,
  );
}

function toggleImage(open: boolean): ImageBox {
  const path = open ? 'M4 6l4 4 4-4' : 'M6 4l4 4-4 4';
  return svgImage(
    `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"><rect x=".5" y=".5" width="15" height="15" rx="4" fill="#fff" stroke="#8a97a8"/>` +
      `<path d="${path}" fill="none" stroke="#3b4a5c" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
    16,
    16,
  );
}

function blockSize(block: BlockRecord) {
  const { inputs, outputs } = portsOf(block);
  const rows = Math.max(inputs.length, outputs.length, 1);
  const longest = (names: string[]) => names.reduce((n, name) => Math.max(n, name.length), 0);
  const plot = inBlock.has(block.path);
  const width = Math.max(plot ? 210 : 150, 7 * (longest(inputs) + longest(outputs)) + 70, 8 * shortName(block.path).length + 40);
  return { width, height: PORT_TOP + PORT_PITCH * rows + (plot ? 132 : 2) };
}

function makeLabel(block: BlockRecord, composite: boolean): BlockLabel {
  const root = document.createElement('div');
  root.className = 'blk';
  root.innerHTML =
    `<div class="blk-name">${esc(shortName(block.path))}</div>` +
    `<div class="blk-type">${esc(block.definition.qualname)}${composite ? ` · ${block.children.length} inside` : ''}<span class="blk-time"></span></div>` +
    `<div class="blk-bar"></div>`;
  let canvas: HTMLCanvasElement | null = null;
  const hint = inBlock.get(block.path);
  if (hint) {
    canvas = document.createElement('canvas');
    canvas.className = 'blk-plot';
    canvas.style.height = '118px';
    root.appendChild(canvas);
    // The picture in a block is a thumbnail; a click opens the interactive scope.
    // A press that moved is a drag of the block, not a click. maxGraph works in pointer events and cancels the mouse ones.
    let pressed: { x: number; y: number } | null = null;
    canvas.addEventListener('pointerdown', (event) => (pressed = { x: event.clientX, y: event.clientY }));
    canvas.addEventListener('click', (event) => {
      const still = pressed && Math.hypot(event.clientX - pressed.x, event.clientY - pressed.y) < 4;
      if (still && !scopes.some((s) => s.probe.path === block.path && s.probe.port === hint.port && s.view === hint.view)) {
        addScope(block.path, hint.port, hint.view);
      }
    });
  }
  return { root, time: root.querySelector('.blk-time')!, bar: root.querySelector('.blk-bar')!, canvas };
}

const LEAF: CellStyle = {
  shape: 'rectangle',
  rounded: true,
  arcSize: 10,
  fillColor: '#ffffff',
  strokeColor: '#5c6b7f',
  strokeWidth: 1.2,
  overflow: 'fill',
  foldable: false,
};
const COLLAPSED: CellStyle = { ...LEAF, fillColor: '#eef1f6', strokeWidth: 2.4, dashed: false };
const CONTAINER: CellStyle = {
  shape: 'swimlane',
  startSize: 28,
  rounded: true,
  arcSize: 6,
  fillColor: '#e7ebf1',
  swimlaneFillColor: '#f4f6f9',
  strokeColor: '#8a97a8',
  fontColor: '#1c2430',
  fontSize: 12,
  fontStyle: 1,
  align: 'left',
  spacingLeft: 28,
  foldable: false,
};

function portStyle(side: 'WEST' | 'EAST', probed: boolean): CellStyle {
  return {
    shape: 'rectangle',
    fillColor: probed ? '#0f8a6c' : '#ffffff',
    strokeColor: probed ? '#0f8a6c' : '#3b4a5c',
    fontSize: 10,
    fontColor: '#3b4a5c',
    labelPosition: side === 'WEST' ? 'right' : 'left',
    align: side === 'WEST' ? 'left' : 'right',
    verticalAlign: 'middle',
    spacingLeft: 3,
    spacingRight: 3,
    movable: false,
  };
}

function isProbed(path: string, port: string) {
  return scopes.some((s) => s.probe.path === path && s.probe.port === port) || inBlock.get(path)?.port === port;
}

function addBlock(node: LaidNode, parent: Cell) {
  const block = blocks.get(node.path)!;
  const composite = block.children.length > 0;
  if (!node.expanded) labels.set(node.path, makeLabel(block, composite));
  const cell = graph.insertVertex({
    parent,
    value: { kind: 'block', path: node.path } satisfies CellInfo,
    position: [node.x, node.y],
    size: [node.width, node.height],
    style: node.expanded ? CONTAINER : composite ? COLLAPSED : LEAF,
  });
  cellByPath.set(node.path, cell);
  for (const port of node.ports) {
    const portCell = graph.insertVertex({
      parent: cell,
      value: { kind: 'port', path: node.path, port: port.name } satisfies CellInfo,
      position: [port.x, port.y],
      size: [PORT, PORT],
      style: portStyle(port.side, isProbed(node.path, port.name)),
    });
    portCells.set(`${node.path}::${port.name}`, portCell);
  }
  node.children.forEach((child) => addBlock(child, cell));
  for (const edge of node.edges) {
    const connection: Connection = model.description.connections[edge.index];
    const color = signalColor(connection.source.path, connection.source.port);
    const edgeCell = graph.insertEdge({
      parent: cell,
      value: { kind: 'edge', index: edge.index } satisfies CellInfo,
      source: portCells.get(`${connection.source.path}::${connection.source.port}`)!,
      target: portCells.get(`${connection.destination.path}::${connection.destination.port}`)!,
      style: {
        strokeColor: color,
        strokeWidth: 1.4,
        rounded: true,
        endArrow: 'block',
        endSize: 5,
        movable: false,
      },
    });
    if (edge.label) {
      // The signal label is its own cell at the position ELK reserved for it.
      graph.insertVertex({
        parent: cell,
        value: { kind: 'edge', index: edge.index, text: edge.label.text } satisfies CellInfo,
        position: [edge.label.x, edge.label.y],
        size: [edge.label.width, edge.label.height],
        style: { fillColor: 'none', strokeColor: 'none', fontSize: 9, fontColor: color, movable: false },
      });
    }
    const geometry = edgeCell.getGeometry()!.clone();
    geometry.points = edge.points.map((p) => new Point(p.x, p.y));
    graph.getDataModel().setGeometry(edgeCell, geometry);
  }
  if (composite) {
    const overlay = new CellOverlay(toggleImage(node.expanded), node.expanded ? 'Collapse' : 'Expand', 'left', 'top', new Point(14, 14), 'pointer');
    overlay.addListener(InternalEvent.CLICK, () => toggleCollapsed(node.path));
    graph.addCellOverlay(cell, overlay);
  }
}

async function rebuild(fit = false) {
  const laid = await layoutModel(model.description, collapsed, blockSize, (index) => {
    const source = model.description.connections[index].source;
    return showSignalLabels ? signalLabel(source.path, source.port) : '';
  });
  const previous = selection;
  graph.home();
  graph.getDataModel().clear();
  cellByPath.clear();
  portCells.clear();
  labels.clear();
  badges.clear();
  graph.batchUpdate(() => addBlock(laid, graph.getDefaultParent()));
  if (currentRoot !== 'root' && cellByPath.has(currentRoot) && !collapsed.has(currentRoot)) {
    graph.enterGroup(cellByPath.get(currentRoot)!);
  } else {
    currentRoot = 'root';
  }
  renderCrumbs();
  select(previous);
  if (fit) fitView();
  applyTrace();
  await refresh();
}

function fitView() {
  graph.getPlugin<FitPlugin>('fit')?.fitCenter({ margin: 24 });
  if (graph.view.scale > 1.25) graph.zoomTo(1.25, true);
}

function ancestors(path: string): string[] {
  const parts = path.split('/');
  return parts.slice(1, -1).map((_, i) => parts.slice(0, i + 2).join('/'));
}

function cellFor(target: CellInfo | null): Cell | undefined {
  if (!target) return undefined;
  if (target.kind === 'block') return cellByPath.get(target.path);
  if (target.kind === 'port') return portCells.get(`${target.path}::${target.port}`);
  return undefined;
}

function select(target: CellInfo | null) {
  const cell = cellFor(target);
  if (cell) graph.setSelectionCell(cell);
  else if (target?.kind !== 'edge') graph.clearSelection();
}

async function toggleCollapsed(path: string) {
  if (collapsed.has(path)) collapsed.delete(path);
  else collapsed.add(path);
  if (currentRoot.startsWith(path) && collapsed.has(path)) currentRoot = 'root';
  await rebuild();
}

async function enter(path: string) {
  const block = blocks.get(path);
  if (!block || block.children.length === 0) return;
  const changed = [path, ...ancestors(path)].filter((p) => collapsed.delete(p)).length > 0;
  currentRoot = path;
  if (changed) {
    await rebuild(true);
    return;
  }
  graph.home();
  if (path !== 'root') graph.enterGroup(cellByPath.get(path)!);
  renderCrumbs();
  fitView();
}

async function reveal(path: string) {
  const hidden = ancestors(path).filter((p) => collapsed.delete(p)).length > 0;
  if (!path.startsWith(currentRoot)) currentRoot = 'root';
  selection = { kind: 'block', path };
  if (hidden || currentRoot === 'root') await rebuild();
  const cell = cellByPath.get(path);
  if (cell) {
    graph.setSelectionCell(cell);
    graph.scrollCellToVisible(cell, true);
  }
}

function renderCrumbs() {
  const parts = currentRoot.split('/');
  $('crumbs').innerHTML = parts
    .map((part, i) => {
      const path = parts.slice(0, i + 1).join('/');
      return i === parts.length - 1 ? `<b>${esc(part)}</b>` : `<button data-enter="${esc(path)}">${esc(part)}</button><span>/</span>`;
    })
    .join('');
}

// ----------------------------------------------------------- observed facts

function applyTrace() {
  const byPath = new Map<string, { order: number; count: number; us: number; self: number }>();
  const siblings = new Map<string, number>();
  const stack: string[] = [];
  let total = 0;
  for (const record of lastTrace) {
    stack.length = record.depth;
    const parentKey = stack.join('>');
    const order = (siblings.get(parentKey) ?? 0) + 1;
    siblings.set(parentKey, order);
    stack.push(`${record.path}#${record.order}`);
    // A call still open in a paused tick has no duration yet; it counts as zero so far.
    const us = record.us ?? 0;
    total = Math.max(total, record.t0_us + us);
    const seen = byPath.get(record.path);
    if (seen) {
      seen.count += 1;
      seen.us += us;
      seen.self += record.self_us ?? 0;
    } else {
      byPath.set(record.path, { order, count: 1, us, self: record.self_us ?? 0 });
    }
  }
  for (const [path, label] of labels) {
    const seen = byPath.get(path);
    // Timings change every tick, so they go straight to the label's DOM and never through the graph model.
    label.time.textContent = seen ? micros(seen.us) : '';
    label.bar.style.width = seen && total ? `${Math.min(100, (seen.us / total) * 100).toFixed(1)}%` : '0';
    const cell = cellByPath.get(path);
    const text = seen ? `#${seen.order}${seen.count > 1 ? ` ×${seen.count}` : ''}` : '';
    const badge = badges.get(path);
    if (!cell || badge?.text === text) continue;
    if (badge) graph.removeCellOverlay(cell, badge.overlay);
    badges.delete(path);
    if (text) {
      const overlay = new CellOverlay(badgeImage(text), 'Observed call order within its parent, last tick', 'right', 'top');
      badges.set(path, { text, overlay: graph.addCellOverlay(cell, overlay) });
    }
  }
  renderTrace(total);
}

function renderTrace(total: number) {
  const strip = $('trace');
  $('trace-total').textContent = lastTrace.length ? `${lastTrace.length} calls · ${micros(total)}` : 'none yet';
  if (!lastTrace.length || !total) {
    strip.innerHTML = '';
    return;
  }
  const selected = selection && selection.kind !== 'edge' ? selection.path : '';
  strip.innerHTML = lastTrace
    .filter((r) => r.depth < 4)
    .map((r) => {
      const left = (r.t0_us / total) * 100;
      const open = r.us === undefined;
      const width = Math.max(0.3, (((r.us ?? total - r.t0_us) || 0) / total) * 100);
      const title = open ? `${r.path} · ${r.entry} · still running` : `${r.path} · ${r.entry} · ${micros(r.us!)} (self ${micros(r.self_us ?? 0)})`;
      return `<div class="${r.path === selected ? 'sel' : ''}${open ? ' open' : ''}" data-path="${esc(r.path)}" title="${esc(title)}" style="left:${left.toFixed(2)}%;width:${width.toFixed(2)}%;top:${r.depth * 15}px">${esc(shortName(r.path))}</div>`;
    })
    .join('');
}

const pausedMark = new CellHighlight(graph, '#e8590c', 3);

function renderPosition() {
  const total = position.total != null ? ` · ${position.index} of ${position.total}` : '';
  $('ticks').textContent = `tick ${position.tick.toLocaleString('en-US')}${total}`;
  const rate = position.ticks_per_s != null && position.state === 'running' ? `${Math.round(position.ticks_per_s).toLocaleString('en-US')} ticks/s` : '';
  const where = position.state === 'paused' && position.at ? `paused before ${position.at.path} · ${position.at.entry}` : position.state;
  $('status').textContent = [where, rate].filter(Boolean).join(' · ');
  if (position.state === 'faulted') $('error').textContent = `Model failed: ${position.error}. Reset to continue.`;
  $('tracebtn').classList.toggle('on', position.trace);
  $('record').classList.toggle('on', !!position.recording);
  $('record').textContent = position.recording ? `Stop recording (${position.recording.ticks})` : 'Record';
  const seek = $<HTMLInputElement>('seek');
  if (position.total != null && document.activeElement !== seek) {
    seek.max = String(position.total);
    seek.value = String(position.index ?? 0);
  }
  // Mark the block about to run; inside a collapsed composite, mark the composite.
  let cell: Cell | undefined;
  if (position.state === 'paused' && position.at) {
    const parts = position.at.path.split('/');
    for (let n = parts.length; n > 0 && !cell; n--) {
      const candidate = cellByPath.get(parts.slice(0, n).join('/'));
      if (candidate && graph.view.getState(candidate)) cell = candidate;
    }
  }
  const state = cell ? graph.view.getState(cell) : null;
  if (state) pausedMark.highlight(state);
  else pausedMark.hide();
}

function applyObservation(result: Observation) {
  position = result.position;
  if (can('execution.free_run')) showRunning(position.state === 'running');
  renderPosition();
  lastTrace = result.trace ?? [];
  applyTrace();
  for (const scope of scopes) {
    const meta = result.frames[scope.probe.id];
    // A port asked for while the model runs arrives with a later snapshot.
    if (!meta) continue;
    const signal = signalFor(result, scope.probe, model);
    scope.meta = meta;
    scope.handle.update(signal, signal ? undefined : frameNote(meta));
    scope.stats.textContent = frameText(meta, signal, position);
    scope.host.style.opacity = isStale(meta, position) ? '0.45' : '1';
  }
  for (const [path, hint] of inBlock) {
    const id = `blk:${path}`;
    const meta = result.frames[id];
    const canvas = labels.get(path)?.canvas;
    if (!meta || !canvas) continue;
    const signal = signalFor(result, { id, path, port: hint.port, channel: 0 }, model);
    drawThumb(canvas, signal, hint.view, frameNote(meta));
    canvas.style.opacity = isStale(meta, position) ? '0.35' : '1';
    canvas.title = `${frameText(meta, signal, position)} · click to open a scope`;
  }
  lastLive = result.live;
  renderLive();
}

let uiCost = 0;
let roundTrip = 0;
let polling = true;
let asking = false;

// Apply one answer, with the cost of each half measured.
function accept(result: Observation | null, started: number) {
  if (!result) return;
  const received = performance.now();
  applyObservation(result);
  uiCost = 0.8 * uiCost + 0.2 * (performance.now() - received);
  roundTrip = 0.8 * roundTrip + 0.2 * (received - started);
  $('cost').textContent = `${(result.bytes / 1024).toFixed(1)} kB · round trip ${roundTrip.toFixed(1)} ms · UI ${uiCost.toFixed(1)} ms`;
}

// Ask for whatever is newer than what the page has. An unchanged model answers with nothing.
async function poll() {
  if (asking || !polling) return;
  asking = true;
  try {
    const started = performance.now();
    accept(await client.observe(watchNow()), started);
  } catch (error) {
    showError(error);
  } finally {
    asking = false;
  }
}

// Redraw everything, as after the page rebuilt the elements it draws into.
async function refresh() {
  client.forget();
  await poll();
}

// Change the shared execution state and show what came of it.
async function act(action: string, body: object = {}) {
  const started = performance.now();
  const result = await client.act(action, watchNow(), body);
  $('error').textContent = '';
  accept(result, started);
}

const pause = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

function showRunning(value: boolean) {
  running = value;
  $('run').textContent = running ? 'Pause' : 'Run';
  $('run').classList.toggle('on', running);
}

// The page watches at the display rate while the model runs and slowly otherwise,
// which is how it learns of a step taken in another window.
async function watch() {
  for (;;) {
    // A hidden tab stops asking, so its probes lapse and cost the model nothing.
    if (!document.hidden) await poll();
    await pause(position.state === 'running' ? 33 : 150);
  }
}

async function play() {
  while (playing) {
    try {
      await act('advance', { mode: 'tick' });
    } catch (error) {
      showError(error);
      break;
    }
    if (position.state !== 'idle') break;
    await pause(66);
  }
  playing = false;
  showRunning(false);
}

// A live model free-runs on its own thread and every window only watches it.
async function setRunning(value: boolean) {
  try {
    if (can('execution.free_run')) {
      if ((position.state === 'running') !== value) await act('run', { on: value });
    } else if (playing !== value) {
      playing = value;
      showRunning(value);
      if (value) void play();
    }
  } catch (error) {
    showError(error);
  }
}

async function step(mode: 'tick' | 'call') {
  await setRunning(false);
  if (mode === 'call' && !position.trace) await act('trace', { on: true });
  await act('advance', { mode });
}

const tick = () => step('tick');

function showError(error: unknown) {
  $('error').textContent = error instanceof Error ? error.message : String(error);
}

// ---------------------------------------------------------------- inspector

function rows(pairs: [string, string][]) {
  return `<table>${pairs.map(([k, v]) => `<tr><td class="k">${esc(k)}</td><td>${v}</td></tr>`).join('')}</table>`;
}

function probeButtons(path: string, port: string) {
  const views: View[] = ['time', 'spectrum', 'constellation'];
  const buttons = views.map((v) => `<button data-act="probe" data-path="${esc(path)}" data-port="${esc(port)}" data-view="${v}">${v}</button>`);
  const shown = inBlock.get(path)?.port === port;
  buttons.push(`<button data-act="inblock" data-path="${esc(path)}" data-port="${esc(port)}" class="${shown ? 'on' : ''}">in block</button>`);
  return `<span class="row-actions">${buttons.join('')}</span>`;
}

function liveText(value: LiveValue | undefined): string {
  if (!value) return '<span class="muted">–</span>';
  if (value.kind === 'scalar') return `<code>${esc(value.value)}</code>`;
  if (value.kind === 'opaque') return `<span class="muted">opaque ${esc(value.type)}</span>`;
  return `<code>${esc(value.head)}…</code><br><span class="muted">peak ${value.peak?.toPrecision(3) ?? '–'} · rms ${value.rms?.toPrecision(3) ?? '–'}${value.nonfinite ? ` · ${value.nonfinite} non-finite` : ''}</span>`;
}

function renderInspector() {
  const target = $('inspector');
  if (!selection) {
    target.innerHTML = '<span class="muted">Select a block, port or connection.</span>';
    return;
  }
  if (selection.kind === 'edge') {
    const c = model.description.connections[selection.index];
    target.innerHTML =
      `<h2>Connection</h2><h3><span class="tag declared">declared</span></h3>` +
      rows([
        ['source', `<code>${esc(c.source.path)}.${esc(c.source.port)}</code>`],
        ['destination', `<code>${esc(c.destination.path)}.${esc(c.destination.port)}</code>`],
        ['owner', `<code>${esc(c.owner)}</code>`],
        ['signal', esc(signalText(c.source.path, c.source.port))],
        ['probe', probeButtons(c.source.path, c.source.port)],
      ]);
    return;
  }
  const path = selection.path;
  const block = blocks.get(path)!;
  const extra = model.facts[path];
  const composite = block.children.length > 0;
  const ports = (kind: 'In' | 'Out') =>
    block.declarations
      .filter((d) => d.kind === kind)
      .map((d): [string, string] => {
        const facts = extra?.ports[d.name];
        const declared = shapeText(d.declared_shape);
        const resolved = shapeText(d.shape);
        const lines = [
          `${esc(dtypeText(d))} ${esc(resolved)}${declared !== resolved ? ` <span class="muted">from ${esc(declared)}</span>` : ''}`,
          `<span class="muted">${[facts?.rate ? hz(facts.rate.hz) : 'rate unknown', block.storage?.[d.name] ?? 'no storage', facts?.nbytes != null ? bytes(facts.nbytes) : ''].filter(Boolean).join(' · ')}</span>`,
          probeButtons(path, d.name),
        ];
        const focus = selection?.kind === 'port' && selection.port === d.name;
        return [`${focus ? '▸ ' : ''}${d.name}`, lines.join('<br>')];
      });
  const params = Object.entries(block.parameters).map(([name, value]): [string, string] => [
    name,
    `<code>${esc(valueText(value))}</code>` +
      (value.kind === 'summary' ? ` <span class="row-actions"><button data-act="value" data-path="${esc(path)}" data-name="${esc(name)}">plot</button></span>` : ''),
  ]);
  const controls = Object.entries(block.controls).map(([name, value]): [string, string] => [
    name,
    value.kind === 'literal' && typeof value.value === 'number'
      && can('values.write')
      ? `<input class="ctl" type="number" step="any" data-act="control" data-path="${esc(path)}" data-name="${esc(name)}" value="${esc(value.value)}">`
      : `<code>${esc(valueText(value))}</code>`,
  ]);
  const states = block.declarations.filter((d) => d.kind === 'State');
  const unknown = model.description.unresolved.filter((u) => u.path === path);
  const ctx = block.context;
  const section = (title: string, tag: string, body: string) => (body ? `<h3>${title} <span class="tag ${tag}">${tag}</span></h3>${body}` : '');
  target.innerHTML =
    `<h2>${esc(shortName(path))} <span class="muted">· ${esc(block.definition.qualname)}</span></h2>` +
    `<div class="muted mono">${esc(path)}</div>` +
    (extra?.doc ? `<div>${esc(extra.doc)}</div>` : '') +
    (composite ? `<div class="row-actions" style="margin-top:6px"><button data-act="enter" data-path="${esc(path)}">Open</button><button data-act="toggle" data-path="${esc(path)}">${collapsed.has(path) ? 'Expand' : 'Collapse'}</button></div>` : '') +
    section(
      'Definition',
      'declared',
      rows([
        ['class', `<code>${esc(block.definition.module)}.${esc(block.definition.qualname)}</code>`],
        ['source', extra?.source ? `<a href="${esc(extra.source.uri)}" title="Open in VS Code"><code>${esc(extra.source.text)}</code></a>` : '<span class="muted">unknown</span>'],
        ['context', ctx ? `${hz(ctx.rate)}${ctx.rate_exact ? '' : ' (float)'} · ${ctx.channels} × ${ctx.n} · ${dtypeName(ctx.dtype)}` : '<span class="muted">not set up</span>'],
        ['phases', Object.entries(block.phases).map(([hook, needs]) => `${esc(hook)} needs ${needs.length ? esc(needs.join(', ')) : 'no inputs'}`).join('<br>') || 'none'],
        ['execution', esc(block.execution_ownership)],
      ]),
    ) +
    section('Parameters', 'declared', params.length ? rows(params) : '') +
    section('Controls', 'declared', controls.length ? rows(controls) : '') +
    section('Inputs', 'declared', ports('In').length ? rows(ports('In')) : '') +
    section('Outputs', 'declared', ports('Out').length ? rows(ports('Out')) : '') +
    section('State', 'declared', states.length ? rows(states.map((d) => [d.name, d.shape ? `${esc(dtypeText(d))} ${esc(shapeText(d.shape))}` : 'scalar'])) : '') +
    `<div id="live"></div>` +
    section('Not known', 'unknown', unknown.length ? rows(unknown.map((u) => [u.declaration ?? u.fact, esc(u.reason)])) : '');
  renderLive();
}

function renderLive() {
  const target = document.getElementById('live');
  if (!target || !selection || selection.kind === 'edge') return;
  const path = selection.path;
  const calls = lastTrace.filter((r) => r.path === path);
  const pairs: [string, string][] = [];
  if (calls.length) {
    const us = calls.reduce((sum, r) => sum + (r.us ?? 0), 0);
    const self = calls.reduce((sum, r) => sum + (r.self_us ?? 0), 0);
    pairs.push(['last tick', `${calls.map((r) => r.entry).join(', ')} · ${micros(us)}${self !== us ? ` (self ${micros(self)})` : ''}`]);
  } else {
    pairs.push(['last tick', '<span class="muted">not called</span>']);
  }
  for (const [name, value] of Object.entries(lastLive?.state ?? {})) pairs.push([`state ${name}`, liveText(value)]);
  for (const [name, value] of Object.entries(lastLive?.controls ?? {})) {
    pairs.push([`control ${name}`, liveText(value)]);
    const input = document.querySelector<HTMLInputElement>(`input.ctl[data-name="${CSS.escape(name)}"]`);
    if (input && document.activeElement !== input && value.kind === 'scalar') input.value = String(value.value);
  }
  target.innerHTML = `<h3>Run <span class="tag observed">observed</span></h3>${rows(pairs)}`;
}

function addScope(path: string, port: string, view: View) {
  const id = `s${scopeCounter++}`;
  const card = document.createElement('div');
  card.className = 'scope';
  // The same scope as an address: it opens in a window or tab with its own subscription.
  const url = `scope.html?${new URLSearchParams({ path, port, view })}`;
  card.innerHTML =
    `<div class="head"><span><code>${esc(path)}.${esc(port)}</code> · ${view}</span>` +
    `<span class="row-actions"><a class="pop" href="${esc(url)}" target="_blank" title="Open in its own window. Ctrl-click for a tab.">pop out</a>` +
    `<button aria-label="Remove scope">×</button></span></div>` +
    `<div class="scope-host"></div><div class="stats"></div>`;
  $('scopes').prepend(card);
  $('scopes-empty').style.display = 'none';
  const host = card.querySelector<HTMLElement>('.scope-host')!;
  const scope: Scope = { probe: { id, path, port, channel: 0 }, view, handle: createScope(host, { view, compact: true }), host, stats: card.querySelector('.stats')! };
  scopes.push(scope);
  card.querySelector('a')!.addEventListener('click', (event) => {
    // A plain click opens a window; with a modifier the browser's own handling opens a tab.
    if (event.ctrlKey || event.metaKey || event.shiftKey || event.button !== 0) return;
    event.preventDefault();
    window.open(url, `scope-${id}`, 'popup,width=780,height=520');
  });
  card.querySelector('button')!.addEventListener('click', () => {
    scopes.splice(scopes.indexOf(scope), 1);
    scope.handle.destroy();
    card.remove();
    markPort(path, port);
  });
  markPort(path, port);
  void refresh();
}

function markPort(path: string, port: string) {
  const cell = portCells.get(`${path}::${port}`);
  if (!cell) return;
  const probed = isProbed(path, port);
  graph.batchUpdate(() => {
    graph.setCellStyles('fillColor', probed ? '#0f8a6c' : '#ffffff', [cell]);
    graph.setCellStyles('strokeColor', probed ? '#0f8a6c' : '#3b4a5c', [cell]);
  });
}

async function showValue(path: string, name: string) {
  const value = await api<{ y?: number[]; re?: number[]; im?: number[]; response_db?: number[]; dtype: string; shape: number[] }>(
    `/api/value?path=${encodeURIComponent(path)}&name=${encodeURIComponent(name)}`,
  );
  const card = document.createElement('div');
  card.className = 'scope';
  card.innerHTML =
    `<div class="head"><span><code>${esc(path)}.${esc(name)}</code> · ${esc(value.dtype)} [${value.shape.join(' × ')}]</span><button aria-label="Remove">×</button></div>` +
    `<canvas class="value" style="height:90px"></canvas>${value.response_db ? '<div class="stats">Magnitude response, dB</div><canvas class="value" style="height:90px"></canvas>' : ''}`;
  $('scopes').prepend(card);
  $('scopes-empty').style.display = 'none';
  card.querySelector('button')!.addEventListener('click', () => card.remove());
  const canvases = card.querySelectorAll('canvas');
  drawLines(canvases[0], value.y ? [value.y] : [value.re ?? [], value.im ?? []], 'values');
  if (value.response_db) drawLines(canvases[1], [value.response_db], '0 … fs/2');
}

// -------------------------------------------------------------------- wiring

graph.getSelectionModel().addListener(InternalEvent.CHANGE, () => {
  selection = info(graph.getSelectionCell());
  renderInspector();
  renderTrace(lastTrace.reduce((end, r) => Math.max(end, r.t0_us + (r.us ?? 0)), 0));
  void refresh();
});

graph.addListener(InternalEvent.DOUBLE_CLICK, (_sender: unknown, event: EventObject) => {
  const value = info(event.getProperty('cell') as Cell | null);
  if (value?.kind === 'block' && blocks.get(value.path)!.children.length) {
    event.consume();
    void enter(value.path);
  }
});

// A block the user moves no longer matches ELK's bend points, so its outside
// connections fall back to the library's own orthogonal router.
graph.addListener(InternalEvent.CELLS_MOVED, (_sender: unknown, event: EventObject) => {
  const moved = (event.getProperty('cells') as Cell[]) ?? [];
  const edges = new Set<Cell>();
  for (const cell of moved) {
    for (const port of cell.getChildCells(true, false)) {
      for (const edge of port.getEdges()) if (edge.getParent() !== cell) edges.add(edge);
    }
  }
  if (!edges.size) return;
  graph.batchUpdate(() => {
    for (const edge of edges) {
      const geometry = edge.getGeometry()!.clone();
      geometry.points = [];
      graph.getDataModel().setGeometry(edge, geometry);
    }
    graph.setCellStyles('edgeStyle', 'orthogonalEdgeStyle', [...edges]);
  });
});

InternalEvent.addMouseWheelListener((event: Event, up: boolean) => {
  if (up) graph.zoomIn();
  else graph.zoomOut();
  InternalEvent.consume(event);
}, container);

$('side').addEventListener('click', (event) => {
  const button = (event.target as HTMLElement).closest<HTMLElement>('[data-act]');
  if (!button || button.tagName !== 'BUTTON') return;
  const { act, path = '', port = '', view = 'time', name = '' } = button.dataset;
  if (act === 'probe') addScope(path, port, view as View);
  else if (act === 'enter') void enter(path);
  else if (act === 'toggle') void toggleCollapsed(path);
  else if (act === 'value') showValue(path, name).catch(showError);
  else if (act === 'inblock') {
    if (inBlock.get(path)?.port === port) inBlock.delete(path);
    else {
      const complex = dtypeText(declaration(path, port)!).startsWith('complex');
      inBlock.set(path, { port, view: complex ? 'constellation' : 'time' });
    }
    void rebuild();
  }
});

$('side').addEventListener('change', (event) => {
  const input = event.target as HTMLInputElement;
  if (input.dataset.act !== 'control') return;
  act('control', { path: input.dataset.path, name: input.dataset.name, value: Number(input.value) }).catch(showError);
});

$('crumbs').addEventListener('click', (event) => {
  const path = (event.target as HTMLElement).dataset.enter;
  if (path) void enter(path);
});
$('trace').addEventListener('click', (event) => {
  const path = (event.target as HTMLElement).dataset.path;
  if (path) void reveal(path);
});
$('run').addEventListener('click', () => void setRunning(!running));
$('step').addEventListener('click', () => step('tick').catch(showError));
$('stepcall').addEventListener('click', () => step('call').catch(showError));
$('reset').addEventListener('click', () => {
  setRunning(false)
    .then(() => act('reset'))
    .catch(showError);
});
$('tracebtn').addEventListener('click', () => act('trace', { on: !position.trace }).catch(showError));
$('record').addEventListener('click', () => {
  const saved = position.recording;
  act('record', { on: !saved })
    .then(() => {
      if (saved) $('status').textContent = `saved ${saved.ticks} ticks to ${saved.path}`;
    })
    .catch(showError);
});
$<HTMLInputElement>('seek').addEventListener('input', (event) => {
  playing = false;
  act('seek', { index: Number((event.target as HTMLInputElement).value) }).catch(showError);
});
$('fit').addEventListener('click', fitView);
$('expand').addEventListener('click', () => {
  collapsed.clear();
  void rebuild(true);
});
$('collapse').addEventListener('click', () => {
  for (const block of blocks.values()) if (block.children.length && block.path !== 'root') collapsed.add(block.path);
  currentRoot = 'root';
  void rebuild(true);
});
$('labels').addEventListener('click', () => {
  showSignalLabels = !showSignalLabels;
  $('labels').classList.toggle('on', showSignalLabels);
  void rebuild();
});
$<HTMLInputElement>('search').addEventListener('keydown', (event) => {
  if (event.key !== 'Enter') return;
  const query = (event.target as HTMLInputElement).value.trim().toLowerCase();
  const hit = [...blocks.keys()].find((path) => path.toLowerCase().includes(query));
  if (query && hit) void reveal(hit);
});

const client = new Client(() => void poll());

async function start() {
  model = await client.document();
  capabilities = new Set(model.capabilities);
  position = model.position;
  blocks = new Map(model.description.blocks.map((b) => [b.path, b]));
  const replay = can('execution.seek');
  $('title').textContent = `${blocks.get('root')!.definition.qualname}`;
  $('subtitle').textContent = `${replay ? 'recording' : 'live'} · ${model.description.profile} · ${blocks.size} blocks`;
  $('stepcall').hidden = !can('execution.call_step');
  $('tracebtn').hidden = !can('trace.toggle');
  $('record').hidden = !can('record');
  $('seek').hidden = !replay;
  if (replay) $('reset').textContent = 'Rewind';
  for (const block of blocks.values()) {
    // Start with the top two levels open; deeper composites open on demand.
    if (block.children.length && block.path.split('/').length > 2) collapsed.add(block.path);
    // Which port a display block draws comes from the document, not from this page.
    const hint = model.facts[block.path]?.display;
    if (hint && declaration(block.path, hint.port)) inBlock.set(block.path, hint);
  }
  await rebuild(true);
  // Exposed for the spike's own browser checks.
  const setPolling = (on: boolean) => (polling = on);
  Object.assign(window, { spike: { graph, cellByPath, portCells, labels, scopes, client, reveal, enter, toggleCollapsed, tick, step, select, setPolling, state: () => ({ currentRoot, selection, uiCost, roundTrip, position }) } });
  void watch();
}

start().catch(showError);
