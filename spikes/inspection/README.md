# Inspection layer spike

A throwaway experiment, not part of Graphion. It tests a layer between a model
and its user interfaces: one protocol, a session behind it, and one adapter per
kind of target. The page in [`../maxgraph`](../maxgraph/README.md) is its only
client so far. Nothing under `src/graphion` was changed for it.

## Run it

From the repository root, with the page built (see the maxGraph README):

```bash
uv run --locked python -m spikes.inspection.server --model ddc
```

Open <http://127.0.0.1:8765/>. `--model ber` serves the BER testbench,
`--model unbound` a model that cannot run, `--trace` starts with call tracing
on, and `--port` changes the port.

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
| `snapshots.py` | A snapshot: one moment of a target, copied once, immutable, shared by every client. |
| `adapter.py` | The live adapter for the Python/NumPy frame profile. The only code that touches block objects. Runs the model on its own thread, publishes snapshots, and observes calls with `sys.monitoring`. |
| `recording.py` | Writes the session's stream to disk, and a replay adapter that serves it back through the same snapshot contract. |
| `session.py` | One protocol over any adapter, for any number of clients. Never touches a block. |
| `wire.py` | The binary form of an observation: a JSON header, then every frame's raw bytes. |
| `views.py` | Summaries of live values, and NumPy reference reductions the page's own are checked against. |
| `server.py` | HTTP transport: URLs mapped to session calls, connections kept alive. |
| `check.py`, `bench.py`, `bench_client.py`, `dsp_fixture.py`, `verify.sh` | Checks, speed measurements, and the full run of both. |

The adapter contract is small: `capabilities`, `document()`, `position()`,
`snapshot(ports, blocks, since)`, `safe()`, `read()`, `advance()`, `reset()`,
and the optional `run()`, `set_trace()`, `write_control()` and `seek()`. A
client offers only what the adapter advertises.

## How a client is served

A client sends the probes it shows, the block it inspects, and the serial of
the last snapshot it was given. The server keeps nothing per client. Execution
state belongs to the adapter and is shared, so a step taken in one window is
seen by every other.

- **Publishing is armed by demand.** A request arms the next publish. The
  model thread copies at the next tick boundary only if armed and at most 30
  times a second. A running model nobody asks copies nothing; the cost to it is
  one attribute read per tick.
- **Watched ports have a lease.** The watched set is the union across clients,
  and a port drops out three seconds after the last request that named it. A
  hidden browser tab stops asking, so it stops costing.
- **A stopped model is read on request.** At idle, at a call-step pause or
  after a fault no tick boundary is coming, so the asking thread copies under
  `safe()`. Staleness marking at a pause works as before.
- **Stamps are captured with the data.** Each frame carries the tick that was
  true when its bytes were copied, and the snapshot is swapped in by
  reference, so a slow client can still be sending one while the next is made.
- **`since` makes an unchanged model cheap.** A request naming the current
  serial gets an empty 204. While the model runs, such a request is held until
  the publish it armed, for at most one period, so a client asking in a loop
  sleeps instead of spinning.
- **Frames travel whole and binary.** One channel per probe, as raw
  little-endian bytes with dtype, full shape and stamp. A channel over 65,536
  samples is cut and marked `truncated`. Two probes on one signal share one
  copy in the snapshot and on the wire.
- **The page reduces.** Decimation, the windowed FFT and statistics run in the
  browser. `views.reference` keeps the NumPy version for the checks.

`since` is a snapshot serial, not a tick: at a call-step pause the data changes
while the tick does not.

## What was checked

```bash
bash spikes/inspection/verify.sh
```

runs everything. All passed on the last run.

| Check | Result |
| --- | --- |
| Layer checks in Python: two example models, the unbound model, awkward values | 63 of 63 |
| The page's transforms against NumPy, in Node | 56 of 56 |
| The scope component alone, in headless Chrome | 23 of 23 |
| The page against live BER | 25 of 25 |
| The page against live DDC | 28 of 28 |
| The page against a DDC recording, no model | 11 of 11 |
| The page against a model with an unbound input | 5 of 5 |

- **Call tracing is switchable.** Off removes the instrumentation; on gives
  call spans and per-block "produced at tick" stamps.
- **Call-level pause and step.** The model thread blocks inside the tracing
  callback before a block's call. Stepping visits every call in traced order.
  While paused, data from blocks that have not yet run this tick is marked
  stale.
- **Nobody watching costs no copies**, two clients on one port cost one copy
  per snapshot, the rate cap holds, and a port nobody asks for any more drops
  out after its lease.
- **A client asking in a tight loop does not starve the model.** Before
  requests were held, such a loop took the interpreter from the model thread
  and left it at 1% of its speed.
- **Reset from a paused tick** finishes the tick first; a hook is never
  abandoned part-way.
- **Replay is exact.** Five recorded ticks replayed through the same session
  code give every frame back byte for byte.
- **Replay needs no model.** The page works from a recording: plots, call
  strip, inspector, new probes, a scrubber and popped-out scopes.
- **A model with an unbound external input** attaches, describes itself,
  reports "no value is bound" for that port and for the ports it is forwarded
  to, and reports the failed run as a fault that reset clears. This is the
  first check that injects a fault.
- **NaN, infinity, complex and enum values** in State or Controls summarize to
  strict JSON.

## Model speed

`uv run --locked python -m spikes.inspection.bench`. Each figure is the
model's speed under a condition divided by its bare speed, as the median of
one-second rounds, with the lowest and highest round in brackets. Bare and
attached runs alternate on one model instance that is reset before each.
Clients are separate processes polling the real HTTP server, which runs in the
model's process.

"Before" is the layer as it was at commit `168f089`, measured with the same
harness (five rounds; "after" is seven): reduction under the lock, JSON, a new
connection per poll. One run of each on one machine. Bare speed was 1289
ticks/s for BER (10 blocks) and 1622 for DDC (23 blocks).

| Condition | BER before | BER after | DDC before | DDC after |
| --- | --- | --- | --- | --- |
| Attached, nobody watching | 103% (98–105) | 99.5% (99–101) | 99% (97–101) | 99.6% (98–102) |
| 4 probes at 30 Hz, one client | 97% (95–98) | 98.0% (97–99) | 96% (92–99) | 97.5% (97–99) |
| 4 probes at 30 Hz, three clients | 84% (84–87) | 96.5% (96–98) | 88% (86–90) | 96.6% (94–97) |
| Call tracing on, nobody watching | 99.5% (96–100) | 95.7% (94–99) | 91% (85–103) | 90.1% (89–93) |
| Call tracing on, 4 probes at 30 Hz, one client | 93% (90–100) | 93.8% (93–95) | 83% (77–86) | 87.4% (77–89) |
| One client asks for one tick at a time | 29% (28–30) | 56.7% (51–57) | 37% (37–38) | 57.1% (55–64) |
| Recording every output | 53% (47–65) | 61.3% (45–63) | 52% (38–55) | 53.9% (51–57) |

Each client received 29.2 updates a second in every polled row, before and
after. An update is 549 kB for BER and 45 kB for DDC now; it was 19 kB and
14 kB of reduced JSON before.

What the table shows:

- **Three clients no longer cost three times one.** Before, each poll reduced
  every probe while holding the model; now a snapshot is copied once and
  handed to all of them.
- **The target was 100% on BER and 95% on DDC** (the old in-process 15 Hz row)
  for four probes at 30 Hz with one and with three clients. DDC meets it, at
  97.5% and 96.6%. BER does not: 98.0% with one client and 96.5% with three.
  That 100% was measured in-process with no transport, once; with real HTTP
  clients the old layer gave 97% and 84%.
- **The old "96% with nobody watching" on DDC was not the adapter.** The DDC
  testbench itself slows down as it runs, by about 30% over its first 24,000
  ticks with nothing attached, because its stimulus takes cosines of an
  ever-growing time. The old bench compared two different stretches of that
  curve. Measured on one instance reset to the same state, the adapter loop
  costs about 1% or less.
- **Where a client's cost goes.** Copying four frames takes 20 to 170 µs per
  snapshot and one request about 0.2 ms from send to answer, yet one client
  first cost about 3%, which is about 1 ms of model time per update.
  Publishing and observing in-process at 30 Hz, with no HTTP, cost about 1%
  or less. The difference is consistent with hand-overs of the interpreter
  lock: every separate write to the socket releases the lock, and the handler
  then has to win it back from the model thread. Sending each response in one
  system call brought one client from about 97% to about 98%.
- **BER sends 549 kB per update** to each client, because two of its four
  probed signals are 16,384 complex samples. Whole frames are what make zoom
  and markers local.
- **Call tracing itself did not change.** The BER "tracing on, nobody
  watching" figures differ by less than the spread of the earlier run.
- **Stepping from a client roughly doubled in speed**, since a step no longer
  reduces anything on the server.
- **Recording is unchanged in design**: it still copies every output every
  tick and summarizes every block's State, and its rounds vary the most.

## What did not work, or was not done

- **Preallocated buffers** were dropped on purpose. A buffer shared with
  request threads would be overwritten while a slow client is still sending
  it. Each snapshot allocates fresh arrays instead. For BER's 549 kB a
  snapshot takes 100 to 170 µs although each frame alone copies in 2 to 6 µs;
  the extra appears only when several large arrays are allocated together,
  which points at page faults on fresh memory. That was not confirmed, and a
  pool that reuses a buffer once nothing refers to it was not built.
- **One write per frame**, to avoid joining the frames into one buffer, made
  things slower, not faster; see the hand-over cost above.
- **`cProfile` cannot run in a process with a live adapter**: the adapter holds
  `sys.monitoring`'s profiler slot.
- **Push** was not needed and not built. A held request already delivers a
  snapshot as soon as it is published.
- **Not built:** a browser-side ring buffer for history longer than one
  frame, cursors shared between windows, a held request for the idle state
  (idle pages poll every 150 ms and get empty 204s), and a transport in a
  process of its own, which is what would remove the hand-over cost.

## What the spike wanted from the core

Nothing was changed under `src/graphion`. These are the places the layer had
to work around something the core does not offer.

- **Whether an input is bound.** Reading an unbound port raises
  `UnavailablePortError`; the description gives its storage role
  (`external_input`) but not whether a value is bound. The layer catches the
  error.
- **Whether an output holds data, and from which tick.** Output storage is
  unspecified before the first run and after reset, and no block records when
  it last ran. The layer infers it by observing calls, or assumes it from the
  tick count and says so.
- **A plain-data summary of live values.** `describe()` covers declared and
  configured values. For live Control and State values the layer wrote its
  own summary, including NaN, complex and enum.
- **Rate for every port.** It comes from the DSP `stream_ports` convention, so
  a port outside it has none and its spectrum is drawn in cycles per sample.
- **Which axis is the channel.** The layer assumes axis 0 of a 2-D port. The
  declared shape names its axes (`C`, `N`), so this could be read instead of
  assumed.
- **A place to stand at a tick boundary.** The adapter owns the loop here. A
  host script that owns its own loop has no hook to publish from.
- **What state a model is in after a hook raises.** The layer reports the
  fault and offers only reset.
- **Display hints and source locations** still come from the layer's own
  registry and from Python's source lookup.

One finding is about an example, not the core: `examples/ddc/ddc_tb.py`
computes its stimulus from absolute time, which is why it slows down.

## Known limits

- **Recording is all or nothing**: every direct producer's output, every tick.
  There is no selection, decimation or size cap.
- **Call stepping follows the traced order**; there are no breakpoints on a
  chosen block or condition.
- **One model per process**, built by name. A host script that owns its own
  loop, and models that are rebuilt (the BER sweep), are not handled.
- **Staleness with tracing off is assumed**, not observed, and says so.
- **Restarting the server needs a page reload**: snapshot serials start again
  and the page would wait for one above the last it saw.
- **A recording started or stopped in one window** shows in another only at
  its next change of state.
