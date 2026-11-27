// What a host says about one probed frame, shared by the diagram page and a popped-out scope.

import { type Signal, stats } from './dsp';
import type { FrameMeta, ModelResponse, Observation, Position, Probe } from './types';

/** NumPy dtype code to a readable name: '<c16' is complex128. */
export function dtypeName(code: string): string {
  const match = /^[<>|=]?([a-zA-Z])(\d+)$/.exec(code);
  if (!match) return code;
  const names: Record<string, string> = { f: 'float', c: 'complex', i: 'int', u: 'uint', b: 'bool' };
  return `${names[match[1]] ?? match[1]}${Number(match[2]) * 8}`;
}

/** Data is stale when its producer has not run in the tick being looked at. */
export function isStale(meta: FrameMeta | undefined, position: Position): boolean {
  if (!meta?.stamp || meta.stamp.basis === 'recorded') return false;
  return meta.stamp.tick < (position.state === 'paused' ? position.tick + 1 : position.tick);
}

/** The decoded samples of a probe with the port's rate, which only the document knows. */
export function signalFor(observation: Observation, probe: Probe, model: ModelResponse): Signal | null {
  const samples = observation.signals.get(probe.id);
  return samples ? { ...samples, rate: model.facts[probe.path]?.ports[probe.port]?.rate?.hz ?? null } : null;
}

/** Why a probe has nothing to draw. */
export function frameNote(meta: FrameMeta): string {
  if (meta.missing) return `no data: ${meta.missing}`;
  return meta.unrun ? 'not run yet' : '';
}

/** One line of facts about a frame: type, level, and which tick it is from. */
export function frameText(meta: FrameMeta, signal: Signal | null, position: Position): string {
  if (!signal || meta.missing || meta.unrun) return frameNote(meta);
  const s = stats(signal);
  let text = `${dtypeName(meta.dtype!)} [${meta.shape!.join(' × ')}] · peak ${s.peak?.toPrecision(3) ?? '–'} · rms ${s.rms?.toPrecision(3) ?? '–'}`;
  if (s.nonfinite) text += ` · ${s.nonfinite} non-finite`;
  if (meta.truncated) text += ` · first ${meta.count} of ${meta.total} samples`;
  if (meta.stamp) text += ` · tick ${meta.stamp.tick} ${meta.stamp.basis}${isStale(meta, position) ? ' · stale' : ''}`;
  return text;
}
