# maxGraph canvas spike

A throwaway experiment, not part of Graphion. It tests whether the mxGraph
lineage (through [maxGraph](https://github.com/maxGraph/maxGraph) 0.25.2) can
be the diagram canvas for a live, inspect-first Graphion UI, with layout done
separately by [ELK](https://github.com/kieler/elkjs).

## Run it

From the repository root, with Node 20+ and uv:

```bash
npm --prefix spikes/maxgraph install
```

```bash
npm --prefix spikes/maxgraph run build
```

```bash
uv run --locked python -m spikes.inspection.server --model ddc
```

Open <http://127.0.0.1:8765/>. `--model ber` serves the BER testbench instead;
`--port` changes the port. The DDC testbench is the better demonstration
because it has three levels of hierarchy. The server and everything behind it
live in [`../inspection`](../inspection/README.md).

## What to try

- Click a block, port or wire: the inspector shows its declared facts and, for
  blocks, what the last run observed. The `source` row is a link that opens
  the defining class in VS Code.
- **Run** lets the model free-run on its own thread while the page watches;
  **Step** runs one root iteration.
- **Call tracing** switches observation of block calls on or off. Off costs
  the model nothing; on gives the call-order badges, timings and call strip.
- **Step call** pauses before each block's call. The block about to run is
  outlined and plots whose data is from the previous tick are dimmed.
- **Record** saves every tick for replay; a served recording gets a scrubber.
- In the inspector, the `time` / `spectrum` / `constellation` buttons add a
  scope for that port. `in block` draws the port inside the block.
- `plot` beside an array parameter fetches it on demand (try FIR taps).
- Edit a Control value in the inspector; it is assigned between iterations.
- The chevron on a composite collapses or expands it. Double-click a composite
  to open it; the breadcrumb leads back.
- Drag blocks, scroll to zoom, drag the background to pan.

## What is in here

| File | Role |
| --- | --- |
| `src/layout.ts` | Description to ELK graph and back. No canvas code. |
| `src/main.ts` | maxGraph cells, labels, overlays, inspector, scopes, run loop. |
| `src/plots.ts` | Canvas line and constellation plots. |
| `check.mjs` | Drives the page in headless Chrome with real mouse input. |

`node check.mjs http://127.0.0.1:8765/ screenshots` (from this folder, server
running) repeats the checks and writes `screenshots/`. Add `replay` when the
server is serving a recording. `bash spikes/inspection/verify.sh` from the
repository root runs every check with its own servers.

## Results

Checked in headless Chrome: 14 of 14 checks on live BER, 17 of 17 on live DDC,
and 9 of 9 on a DDC recording with no model loaded.

- **Ports, wires, containers**: ports are child cells; ELK supplies positions
  and orthogonal bend points; composites are swimlane containers.
- **Drill-down**: the native `enterGroup` works and shows a composite's own
  ports at the edges of the view.
- **Badges**: cell overlays carry observed call order and a collapse toggle.
- **Live plot in a block**: a DOM node label hosts a canvas that survives
  redraws. Per-tick values are written straight to that DOM, so a tick costs
  about 1.5 to 1.8 ms of UI time with four in-block plots.
- **Locked topology**: delete, connect, clone and rename are all off; blocks
  can still be moved.

### What did not work as assumed

- **Native folding hides a composite's own ports** (0 of 4 stayed drawn).
  Collapse is therefore a rebuild with a new layout, not a fold.
- **HTML labels assume `content-box`**. A global `box-sizing: border-box`
  reset shifts every label by half its height.
- **Tooltips moved to a plugin** (`TooltipHandler`), so mxGraph examples do not
  port unchanged; DOM-node labels work but are not in the typings.
- **Wire labels are separate cells** placed by ELK. A wire rerouted after a
  drag leaves its label behind.
- **Dragged positions are lost** on any rebuild; layout pins are not done.

### What the description was missing

`describe()` was enough for structure. The inspection layer's document adds
what the UI needed:

- Source file and line, and the class's own docstring.
- Per-port rate. It comes from the DSP `stream_ports` convention, so a block
  outside that convention (the DDC `Stimulus`) shows "rate unknown".
- Bytes per buffer.
- Whether an output holds data yet. Output storage is unspecified before the
  first run and after reset, so the spike shows "not run yet" from its own
  tick count; the framework does not track this.
- A display hint. A registry in the layer says that `Spectrum.psd` and
  `ConstellationViewer.points` are worth drawing inside the block.

## Not covered

Large models (the biggest here has 23 blocks), a VS Code webview, dark theme,
keyboard access, touch input, Firefox and Safari.
