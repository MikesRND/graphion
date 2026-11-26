// Shapes of the server payloads. The description follows Graphion's schema v2.

export type ValueRecord =
  | { kind: 'literal'; value: unknown; dtype?: string }
  | { kind: 'fraction'; numerator: number; denominator: number }
  | { kind: 'summary'; dtype: string; shape: number[] }
  | { kind: 'required' }
  | { kind: 'opaque'; type: string };

export type DimRecord =
  | { kind: 'resolved'; value: number }
  | { kind: 'symbol' | 'axis' | 'param'; name: string }
  | { kind: 'fraction'; numerator: number; denominator: number }
  | { kind: 'expression'; operator: 'multiply' | 'divide'; left: DimRecord; right: DimRecord };

export type DtypeRecord = { kind: 'resolved'; value: string } | { kind: 'symbol'; name: string } | { kind: 'invalid'; type: string };

export interface Declaration {
  name: string;
  kind: 'Param' | 'Control' | 'State' | 'In' | 'Out';
  value_type: { kind: string; module?: string; qualname?: string; type?: string };
  dtype: DtypeRecord | null;
  shape: DimRecord[] | null;
  declared_shape: DimRecord[] | null;
  default: ValueRecord | null;
}

export interface BlockRecord {
  path: string;
  definition: { module: string; qualname: string };
  execution_ownership: string;
  phases: Record<string, string[]>;
  parameters: Record<string, ValueRecord>;
  controls: Record<string, ValueRecord>;
  declarations: Declaration[];
  children: string[];
  context: null | { rate: number; rate_exact: null | { numerator: number; denominator: number }; n: number; channels: number; dtype: string };
  storage: null | Record<string, string>;
}

export interface Connection {
  owner: string;
  source: { path: string; port: string };
  destination: { path: string; port: string };
}

export interface Description {
  schema_version: number;
  profile: string;
  topology: string;
  blocks: BlockRecord[];
  connections: Connection[];
  unresolved: { path: string; declaration: string | null; fact: string; reason: string }[];
}

export interface Facts {
  source: { text: string; uri: string } | null;
  doc: string | null;
  display: { port: string; view: ProbePayload['view'] } | null;
  ports: Record<string, { nbytes: number; rate?: { text: string; hz: number } | null; producer?: string | null }>;
}

export interface Position {
  tick: number;
  state: 'idle' | 'running' | 'paused' | 'faulted' | 'ended';
  at: { path: string; entry: string } | null;
  error: string | null;
  trace: boolean;
  total: number | null;
  index?: number;
  ticks_per_s: number | null;
  recording: { path: string; ticks: number } | null;
}

export interface ModelResponse {
  schema: string;
  model: string;
  description: Description;
  facts: Record<string, Facts>;
  origins: Record<string, string>;
  capabilities: string[];
  position: Position;
}

export interface TraceRecord {
  path: string;
  entry: string;
  depth: number;
  order: number;
  t0_us: number;
  // Absent while the call is still open, which happens when a tick is paused part-way.
  us?: number;
  self_us?: number;
}

export interface ProbePayload {
  view: 'time' | 'array' | 'spectrum' | 'constellation';
  dtype?: string;
  shape?: number[];
  stats?: { nonfinite: number; peak: number | null; rms: number | null };
  stamp?: { tick: number; basis: 'observed' | 'assumed' | 'recorded' };
  missing?: string;
  re?: number[];
  im?: number[];
  y?: number[];
  db?: number[];
  f0?: number;
  f1?: number;
  has_rate?: boolean;
  unrun?: boolean;
}

export type LiveValue =
  | { kind: 'scalar'; value: unknown }
  | { kind: 'array'; dtype: string; shape: number[]; head: string; nonfinite: number; peak: number | null; rms: number | null }
  | { kind: 'opaque'; type: string };

export interface LiveBlock {
  controls: Record<string, LiveValue>;
  state: Record<string, LiveValue>;
}

export interface Observation {
  position: Position;
  probes: Record<string, ProbePayload>;
  live: LiveBlock | null;
  trace: TraceRecord[];
  trace_partial: boolean;
  error?: string;
}

export interface ProbeSpec {
  id: string;
  path: string;
  port: string;
  view: ProbePayload['view'];
  max: number;
}
