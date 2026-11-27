// Decoding the server's observation body: a JSON header, then raw frames.
// The layout is described in spikes/inspection/wire.py.

import type { Signal } from './dsp';
import type { FrameMeta, Observation, WireHeader } from './types';

type Samples = Omit<Signal, 'rate'>;

const VIEWS: Record<string, (buffer: ArrayBuffer, offset: number, count: number) => ArrayLike<number | bigint>> = {
  f8: (b, o, n) => new Float64Array(b, o, n),
  f4: (b, o, n) => new Float32Array(b, o, n),
  i1: (b, o, n) => new Int8Array(b, o, n),
  i2: (b, o, n) => new Int16Array(b, o, n),
  i4: (b, o, n) => new Int32Array(b, o, n),
  i8: (b, o, n) => new BigInt64Array(b, o, n),
  u1: (b, o, n) => new Uint8Array(b, o, n),
  u2: (b, o, n) => new Uint16Array(b, o, n),
  u4: (b, o, n) => new Uint32Array(b, o, n),
  u8: (b, o, n) => new BigUint64Array(b, o, n),
  b1: (b, o, n) => new Uint8Array(b, o, n),
};

// One channel as doubles. A float64 frame is a view of the response's own
// bytes; every other type is converted once, here.
function samples(buffer: ArrayBuffer, offset: number, meta: FrameMeta): Samples {
  const match = /^[<|=]([fciub])(\d+)$/.exec(meta.dtype ?? '');
  if (!match) throw new Error(`cannot read dtype ${meta.dtype}`);
  const [, kind, size] = match;
  const count = meta.count ?? 0;
  if (kind === 'c') {
    const parts = VIEWS[`f${Number(size) / 2}`]?.(buffer, offset, count * 2) as ArrayLike<number> | undefined;
    if (!parts) throw new Error(`cannot read dtype ${meta.dtype}`);
    const real = new Float64Array(count);
    const imag = new Float64Array(count);
    for (let i = 0; i < count; i++) {
      real[i] = parts[2 * i];
      imag[i] = parts[2 * i + 1];
    }
    return { real, imag };
  }
  const view = VIEWS[`${kind}${size}`]?.(buffer, offset, count);
  if (!view) throw new Error(`cannot read dtype ${meta.dtype}`);
  if (view instanceof Float64Array) return { real: view, imag: null };
  // 64-bit integers lose precision above 2^53 as doubles; a plot does not need more.
  return { real: Float64Array.from(view as ArrayLike<number | bigint>, Number), imag: null };
}

export function decode(buffer: ArrayBuffer): Observation {
  const length = new DataView(buffer).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 4, length))) as WireHeader;
  const start = 4 + length;
  const byKey = new Map<string, Samples>();
  const signals = new Map<string, Samples>();
  for (const [id, meta] of Object.entries(header.frames)) {
    if (!meta.key) continue;
    // Two probes on one signal share its array on the wire and here.
    if (!byKey.has(meta.key)) byKey.set(meta.key, samples(buffer, start + header.arrays[meta.key][0], meta));
    signals.set(id, byKey.get(meta.key)!);
  }
  return { ...header, signals, bytes: buffer.byteLength };
}
