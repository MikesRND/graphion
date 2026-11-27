// The page's side of the protocol, shared by the diagram page and a popped-out scope.
//
// A client holds everything that is its own: what it watches and the serial of
// the last snapshot it was given. The server keeps nothing per client.

import type { ModelResponse, Observation, Probe } from './types';
import { decode } from './wire';

export interface Watch {
  probes: Probe[];
  inspect: string | null;
}

// Other windows on the same model are told when this one changes it, so they ask at once instead of at their next idle poll.
const channel = 'BroadcastChannel' in window ? new BroadcastChannel('graphion-inspection-spike') : null;

export class Client {
  /** Serial of the newest snapshot applied; sent as `since` so an unchanged model answers with an empty 204. */
  serial = 0;
  /** Requests sent, by URL path, for the page's own checks. */
  readonly sent: Record<string, number> = {};
  private watched = '';

  constructor(private readonly onChange?: () => void) {
    channel?.addEventListener('message', () => this.onChange?.());
  }

  private async post(path: string, body: object): Promise<Response> {
    this.sent[path] = (this.sent[path] ?? 0) + 1;
    const response = await fetch(path, { method: 'POST', body: JSON.stringify(body) });
    if (response.status === 400) throw new Error((await response.json()).error);
    if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
    return response;
  }

  private accept(buffer: ArrayBuffer): Observation | null {
    const observation = decode(buffer);
    // A poll and an action can cross; the older answer must not overwrite the newer.
    if (observation.serial < this.serial) return null;
    this.serial = observation.serial;
    return observation;
  }

  async document(): Promise<ModelResponse> {
    const response = await fetch('/api/document');
    return (await response.json()) as ModelResponse;
  }

  /** Make the next observe ask for everything, as after the page rebuilt what it draws into. */
  forget() {
    this.watched = '';
  }

  /** Ask for whatever is newer than what this client has; null when nothing is. */
  async observe(watch: Watch): Promise<Observation | null> {
    const key = JSON.stringify(watch);
    // A changed watch list makes the last snapshot incomplete for this client, so ask for all of it again.
    const since = key === this.watched ? this.serial : null;
    // Noted when sent, not when answered, so a forget() while this is in flight still holds for the next one.
    this.watched = key;
    const response = await this.post('/api/observe', { ...watch, since });
    return response.status === 204 ? null : this.accept(await response.arrayBuffer());
  }

  /** Change the shared execution state, and get the state after it. */
  async act(action: string, watch: Watch, body: object = {}): Promise<Observation | null> {
    const response = await this.post(`/api/${action}`, { ...body, ...watch });
    this.watched = JSON.stringify(watch);
    channel?.postMessage(action);
    return this.accept(await response.arrayBuffer());
  }
}
