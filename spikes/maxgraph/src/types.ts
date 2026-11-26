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

export interface Extras {
  source: { text: string; uri: string } | null;
  doc: string | null;
  ports: Record<string, { nbytes: number; rate?: { text: string; hz: number } | null }>;
}

export interface ModelResponse {
  model: string;
  description: Description;
  extras: Record<string, Extras>;
  ticks: number;
}

export interface TraceRecord {
  path: string;
  entry: string;
  depth: number;
  order: number;
  t0_us: number;
  us: number;
  self_us: number;
}

export interface ProbePayload {
  view: 'time' | 'array' | 'spectrum' | 'constellation';
  dtype: string;
  shape: number[];
  stats: { nonfinite: number; peak: number | null; rms: number | null };
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
  ticks: number;
  probes: Record<string, ProbePayload>;
  live: LiveBlock | null;
  trace?: TraceRecord[];
  us_per_tick?: number | null;
  error?: string;
}

export interface ProbeSpec {
  id: string;
  path: string;
  port: string;
  view: ProbePayload['view'];
  max: number;
}
