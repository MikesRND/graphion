# Inspection layer spike

A throwaway experiment, not part of Graphion. It tests a layer between a model
and its user interfaces: one protocol, a session behind it, and one adapter per
kind of target. The page in [`../maxgraph`](../maxgraph/README.md) is its only
client so far.

## Run it

From the repository root, with the page built (see the maxGraph README):

```bash
uv run --locked python -m spikes.inspection.server --model ddc
```

Open <http://127.0.0.1:8765/>. `--model ber` serves the BER testbench,
`--trace` starts with call tracing on, and `--port` changes the port.

To replay a recording with no model loaded:

```bash
uv run --locked python -m spikes.inspection.server --replay spikes/inspection/recordings/<name>
```

Recordings are made with the page's **Record** button and land in
`spikes/inspection/recordings/` (not tracked).

## Parts

| File | Role |
| --- | --- |
| `document.py` | Composes the model document: `describe()` plus facts from named providers, with each kind of fact's origin stated. Resolves every port to its direct producer. |
| `adapter.py` | The live adapter for the Python/NumPy frame profile. The only code that touches block objects. Runs the model on its own thread and observes calls with `sys.monitoring`. |
| `recording.py` | Writes the session's stream to disk, and a replay adapter that serves it back. |
| `session.py` | One protocol over any adapter. Never touches a block. |
| `views.py` | Reductions from raw arrays to display-sized payloads. |
| `server.py` | HTTP transport: URLs mapped to session calls. |
| `check.py`, `bench.py`, `verify.sh` | Checks, speed measurements, and the full run of both. |

The adapter contract is small: `capabilities`, `document()`, `position()`,
`safe()`, `read()`, `live()`, `stamp()`, `spans()`, `advance()`, `reset()`, and
the optional `run()`, `set_trace()`, `write_control()` and `seek()`. A client
offers only what the adapter advertises.

## What was checked

```bash
bash spikes/inspection/verify.sh
```

runs everything: 28 layer checks in Python, the page against each live model in
headless Chrome (14 on BER, 17 on DDC), and the page against a recording with no
model (9). All passed on the last run.

- **Call tracing is switchable.** Off removes the instrumentation; on gives
  call spans and per-block "produced at tick" stamps.
- **Call-level pause and step.** The model thread blocks inside the tracing
  callback before a block's call. Stepping visits every call in traced order.
  While paused, data from blocks that have not yet run this tick is marked
  stale.
- **Free run.** The model runs on its own thread and never waits for a client.
  Readers are let in between ticks; after each read the model is owed one tick,
  so a client polling in a tight loop cannot starve it.
- **Reset from a paused tick** finishes the tick first; a hook is never
  abandoned part-way.
- **Replay is exact.** Five recorded ticks replayed through the same session
  code give the same payloads as the live run did, for every probe and view.
- **Replay needs no model.** The page works from a recording: plots, call
  strip, inspector, new probes and a scrubber. Controls and live-only buttons
  are absent because the replay adapter does not advertise them.

## Model speed

`uv run --locked python -m spikes.inspection.bench`, one run on one machine;
run-to-run noise was about 3%.

| Condition | BER (10 blocks) | DDC (23 blocks) |
| --- | --- | --- |
| Bare model, plain loop | 1374 ticks/s | 1802 ticks/s |
| Attached, tracing off, nobody watching | 101% | 96% |
| Attached, tracing off, polled at 15 Hz | 100% | 95% |
| Attached, call tracing on | 97% | 87% |
| Attached, call tracing on, polled at 15 Hz | 98% | 83% |
| Client asks for one tick at a time | 76% | 50% |
| Recording every output | 54% | 44% |

With the real page attached in headless Chrome the model ran at about 1350
ticks/s (BER) and 1690 ticks/s (DDC) with tracing off, and 1305 and 1540 with
tracing on.

Tracing costs roughly 2 to 4 µs per block call. Recording is expensive because
it copies every output every tick and summarises every block's State.

## Known limits

- **Recording is all or nothing**: every direct producer's output, every tick.
  There is no selection, decimation or size cap.
- **Call stepping follows the traced order**; there are no breakpoints on a
  chosen block or condition.
- **A fault** (a hook raising) is reported as session state, but no check
  injects one.
- **One model per process**, built by name. A host script that owns its own
  loop, and models that are rebuilt (the BER sweep), are not handled.
- **HTTP polling** carries JSON for everything. There is no binary data path
  and no push.
- **Staleness with tracing off is assumed**, not observed, and says so.
