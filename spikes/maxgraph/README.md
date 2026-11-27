# maxGraph canvas spike

A throwaway experiment, not part of Graphion. It tests whether the mxGraph
lineage (through [maxGraph](https://github.com/maxGraph/maxGraph) 0.25.2) can
be the diagram canvas for a live, inspect-first Graphion UI, with layout done
separately by [ELK](https://github.com/kieler/elkjs), and whether
[uPlot](https://github.com/leeoniya/uPlot) 1.6.32 can be its interactive
scope.

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
  scope for that port. In a scope: drag to zoom, scroll to zoom about the
  cursor, Shift-drag or middle-drag to pan, double-click to reset, click to
  place a marker (two at most; the readout shows their difference). **Peak**
  moves the marker to the largest value in view, and **Auto** switches
  vertical autoscale. None of this talks to the server.
- **pop out** on a scope opens it in its own window; Ctrl-click opens a tab.
  The address names the port and view, so it can be bookmarked or opened by
  hand: `scope.html?path=root/dut&port=y0&view=spectrum`.
- `in block` draws the port inside the block as a thumbnail. Click the
  thumbnail to open its scope.
- Open the page in two windows: a step or a run in one shows in the other.
- `plot` beside an array parameter fetches it on demand (try FIR taps).
- Edit a Control value in the inspector; it is assigned between iterations.
- The chevron on a composite collapses or expands it. Double-click a composite
  to open it; the breadcrumb leads back.
- Drag blocks, scroll to zoom, drag the background to pan.

## What is in here

| File | Role |
| --- | --- |
| `src/layout.ts` | Description to ELK graph and back. No canvas code. |
| `src/main.ts` | maxGraph cells, labels, overlays, inspector, scope cards, polling. |
| `src/client.ts`, `src/wire.ts` | The page's side of the protocol: what it watches, the serial it has, and decoding of binary frames. |
| `src/dsp.ts` | FFT of any length, Hann window, spectrum, statistics and min/max decimation, checked against NumPy. |
| `src/scope.ts` | The interactive scope: uPlot for time, array and spectrum, a canvas for the constellation, and the marker layer. |
| `src/popout.ts`, `scope.html` | A scope alone in a window, as its own client. |
| `src/frames.ts`, `src/plots.ts` | Text and staleness for a frame; thumbnails and parameter plots. |
| `check.mjs` | Drives the page in headless Chrome with real mouse input. |
| `scope.check.mjs`, `scope-demo.html` | The scope alone, on synthetic signals. |
| `dsp.check.mjs` | The page's transforms against NumPy, in Node. |

`node check.mjs http://127.0.0.1:8765/ screenshots` (from this folder, server
running) repeats the checks and writes `screenshots/`. Add `replay` when the
server is serving a recording, or `unbound` for `--model unbound`.
`bash spikes/inspection/verify.sh` from the repository root runs every check
with its own servers.

## Results

Checked in headless Chrome: 25 of 25 checks on live BER, 28 of 28 on live DDC,
11 of 11 on a DDC recording with no model loaded, 5 of 5 on a model that
cannot run, and 23 of 23 on the scope alone. The page's transforms pass 56 of
56 comparisons with NumPy.

- **Ports, wires, containers**: ports are child cells; ELK supplies positions
  and orthogonal bend points; composites are swimlane containers.
- **Drill-down**: the native `enterGroup` works and shows a composite's own
  ports at the edges of the view.
- **Badges**: cell overlays carry observed call order and a collapse toggle.
- **Live plot in a block**: a DOM node label hosts a canvas that survives
  redraws. Per-update values are written straight to that DOM. With four
  in-block plots and one scope, an update costs the page about 1 to 2 ms.
- **Locked topology**: delete, connect, clone and rename are all off; blocks
  can still be moved.

### Scopes

- **Whole frames, reduced in the page.** The server sends each probed channel
  as raw bytes. The FFT, window, statistics and thumbnails are computed here.
  With the transform included, a 4096-point spectrum scope updates in about
  0.4 ms, so no worker was needed.
- **Markers read the data, not the picture.** In the browser check a marker's
  readout equals NumPy's value for that bin of the same frame to within
  1e-6 dB, and peak search lands on NumPy's largest bin in view.
- **Zoom survives updates**, and a marker keeps its sample index and reads the
  new data each update.
- **Interaction is local.** While the check zooms, places a marker and runs a
  peak search, the only requests the page sends are its unchanged idle poll.
- **A popped-out scope is another client.** It shows the same tick as the main
  page, follows a step or a seek made there, and works on a recording.
- **Update cost** of one scope, transform and draw together, median in
  headless Chrome: 0.4 ms for a 4096-point spectrum, 0.5 ms for 4096 complex
  points in time, 0.3 ms for a 1024-point constellation.

### What uPlot could not do

[`SCOPE-NOTES.md`](SCOPE-NOTES.md) has the detail, read from uPlot's source.

- **No wheel zoom and no pan.** Both are written here, on top of its x scale.
- **No markers or annotations.** Placement, drawing, peak search, delta and
  readout are all written here, in its draw hook.
- **No scatter plot worth using.** Its mode 2 has no default path builder and
  no equal aspect ratio, so the constellation is a plain canvas with its own
  zoom and pan. Markers on the constellation are not done.
- **It does not resize itself**; a ResizeObserver does it.
- **`setData` resets the zoom by default**, and with the reset off it does not
  redraw. Each update sets the data and the x range together in one batch.
- It did not need explicit decimation: its line builder already draws minimum
  and maximum per pixel column once there are four points per pixel. The
  decimation written here is used only by the thumbnails.

### What did not work as assumed

- **Native folding hides a composite's own ports** (0 of 4 stayed drawn).
  Collapse is therefore a rebuild with a new layout, not a fold.
- **HTML labels assume `content-box`**. A global `box-sizing: border-box`
  reset shifts every label by half its height.
- **Tooltips moved to a plugin** (`TooltipHandler`), so mxGraph examples do not
  port unchanged; DOM-node labels work but are not in the typings.
- **Wire labels are separate cells** placed by ELK. A wire rerouted after a
  drag leaves its label behind.
- **maxGraph cancels mouse events inside labels.** It works in pointer events,
  so a `mousedown` listener on a canvas in a label never fires; `pointerdown`
  and `click` do.
- **Dragged positions are lost** on any rebuild; layout pins are not done.

### What the description was missing

`describe()` was enough for structure. The inspection layer's document adds
what the UI needed:

- Source file and line, and the class's own docstring.
- Per-port rate. It comes from the DSP `stream_ports` convention, so a block
  outside that convention (the DDC `Stimulus`) shows "rate unknown".
- Bytes per buffer, worked out from the description's resolved shape and
  dtype.
- Whether an output holds data yet. Output storage is unspecified before the
  first run and after reset, so the spike shows "not run yet" from its own
  tick count; the framework does not track this.
- A display hint. A registry in the layer says that `Spectrum.psd` and
  `ConstellationViewer.points` are worth drawing inside the block.

## Not covered

Large models (the biggest here has 23 blocks), a VS Code webview, dark theme,
keyboard access, touch input, Firefox and Safari. For the scopes: a time
history longer than one frame, cursors shared between windows, a channel
selector (the address takes `channel=`, the page always asks for 0), screens
with a pixel ratio above 1, and frames holding NaN.
