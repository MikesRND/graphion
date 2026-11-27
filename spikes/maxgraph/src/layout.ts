// Layout is computed from the model by ELK, independently of the canvas library.

import ELK from 'elkjs/lib/elk.bundled.js';
import type { ElkExtendedEdge, ElkNode, ElkPort } from 'elkjs/lib/elk-api';
import type { BlockRecord, Description } from './types';

export const PORT = 8;
export const PORT_TOP = 44;
export const PORT_PITCH = 18;

export interface LaidPort {
  name: string;
  side: 'WEST' | 'EAST';
  x: number;
  y: number;
}

export interface LaidEdge {
  index: number;
  points: { x: number; y: number }[];
  label: { text: string; x: number; y: number; width: number; height: number } | null;
}

export interface LaidNode {
  path: string;
  x: number;
  y: number;
  width: number;
  height: number;
  expanded: boolean;
  ports: LaidPort[];
  children: LaidNode[];
  edges: LaidEdge[];
}

export function portsOf(block: BlockRecord) {
  return {
    inputs: block.declarations.filter((d) => d.kind === 'In').map((d) => d.name),
    outputs: block.declarations.filter((d) => d.kind === 'Out').map((d) => d.name),
  };
}

const portId = (path: string, port: string) => `${path}::${port}`;

// Approximate 9px label metrics; ELK only needs enough room reserved.
function labelSize(text: string) {
  const lines = text.split('\n');
  return { width: Math.max(...lines.map((line) => line.length)) * 5.1 + 4, height: 11.5 * lines.length };
}

export async function layoutModel(
  description: Description,
  collapsed: Set<string>,
  sizeOf: (block: BlockRecord) => { width: number; height: number },
  edgeLabel: (index: number) => string,
): Promise<LaidNode> {
  const blocks = new Map(description.blocks.map((b) => [b.path, b]));
  const expandedPaths = new Set<string>();

  const build = (path: string): ElkNode => {
    const block = blocks.get(path)!;
    const { inputs, outputs } = portsOf(block);
    const expanded = block.children.length > 0 && !collapsed.has(path);
    if (!expanded) {
      // A leaf fixes its port positions so labels and plots inside it stay clear of them.
      const { width, height } = sizeOf(block);
      const port = (name: string, index: number, side: 'WEST' | 'EAST'): ElkPort => ({
        id: portId(path, name),
        width: PORT,
        height: PORT,
        x: side === 'WEST' ? -PORT / 2 : width - PORT / 2,
        y: PORT_TOP + PORT_PITCH * index,
        layoutOptions: { 'elk.port.side': side },
      });
      return {
        id: path,
        width,
        height,
        ports: [...inputs.map((n, i) => port(n, i, 'WEST')), ...outputs.map((n, i) => port(n, i, 'EAST'))],
        layoutOptions: { 'elk.portConstraints': 'FIXED_POS' },
      };
    }
    expandedPaths.add(path);
    // ELK orders ports clockwise, so west ports count upward to read top to bottom.
    const port = (name: string, index: number, side: 'WEST' | 'EAST', count: number): ElkPort => ({
      id: portId(path, name),
      width: PORT,
      height: PORT,
      layoutOptions: {
        'elk.port.side': side,
        'elk.port.index': String(side === 'WEST' ? count - 1 - index : index),
        'elk.port.borderOffset': String(-PORT / 2),
      },
    });
    const edges: ElkExtendedEdge[] = [];
    description.connections.forEach((c, index) => {
      if (c.owner === path) {
        // Giving ELK the label lets it reserve room, so labels never sit on blocks.
        const text = edgeLabel(index);
        edges.push({
          id: `e${index}`,
          sources: [portId(c.source.path, c.source.port)],
          targets: [portId(c.destination.path, c.destination.port)],
          labels: text ? [{ id: `l${index}`, text, ...labelSize(text) }] : [],
        });
      }
    });
    return {
      id: path,
      children: block.children.map(build),
      edges,
      ports: [
        ...inputs.map((n, i) => port(n, i, 'WEST', inputs.length)),
        ...outputs.map((n, i) => port(n, i, 'EAST', outputs.length)),
      ],
      layoutOptions: {
        'elk.algorithm': 'layered',
        'elk.direction': 'RIGHT',
        'elk.edgeRouting': 'ORTHOGONAL',
        'elk.portConstraints': 'FIXED_ORDER',
        'elk.padding': '[top=44,left=28,bottom=22,right=28]',
        'elk.layered.spacing.nodeNodeBetweenLayers': '56',
        'elk.spacing.nodeNode': '30',
        'elk.spacing.edgeNode': '18',
        'elk.layered.nodePlacement.strategy': 'NETWORK_SIMPLEX',
      },
    };
  };

  const laid = await new ELK().layout({
    id: '__top__',
    children: [build('root')],
    layoutOptions: { 'elk.algorithm': 'layered', 'elk.direction': 'RIGHT', 'elk.padding': '[top=20,left=20,bottom=20,right=20]' },
  });

  const read = (node: ElkNode): LaidNode => {
    const block = blocks.get(node.id)!;
    const inputs = new Set(portsOf(block).inputs);
    return {
      path: node.id,
      x: node.x ?? 0,
      y: node.y ?? 0,
      width: node.width ?? 0,
      height: node.height ?? 0,
      expanded: expandedPaths.has(node.id),
      ports: (node.ports ?? []).map((p) => {
        const name = p.id.slice(node.id.length + 2);
        return { name, side: inputs.has(name) ? 'WEST' : 'EAST', x: p.x ?? 0, y: p.y ?? 0 };
      }),
      children: (node.children ?? []).map(read),
      edges: (node.edges ?? []).map((e) => {
        const section = e.sections?.[0];
        const label = e.labels?.[0];
        return {
          index: Number(e.id.slice(1)),
          points: section?.bendPoints ?? [],
          label: label?.text
            ? { text: label.text, x: label.x ?? 0, y: label.y ?? 0, width: label.width ?? 0, height: label.height ?? 0 }
            : null,
        };
      }),
    };
  };
  return read(laid.children![0]);
}
