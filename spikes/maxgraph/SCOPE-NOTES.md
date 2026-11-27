# Scope component notes

`src/scope.ts` is an interactive scope with four views: time, array, spectrum and
constellation. `scope-demo.html` / `src/scope-demo.ts` host it with synthetic
signals. `node spikes/maxgraph/scope.check.mjs` builds the demo, serves this
folder on a random local port and drives headless Chrome. Last run: 23 of 23
checks passed.

## uPlot 1.6.32: what the source actually does

I read these facts in `node_modules/uplot/dist/uPlot.esm.js` and `uPlot.d.ts`.

- `setData(data)` defaults to `resetScales = true`. With the default auto x
  scale it calls `autoScaleX()`, which drops any user zoom. So a plain
  `setData` on every frame would undo every zoom.
- `setData(data, false)` swaps the arrays and stops there. It does not re-range
  any scale or commit, so nothing redraws. The scope calls
  `u.batch(() => { setData(d, false); setScale('x', range) })` on every update.
- `setScale('x', …)` re-ranges every scale whose `auto` is true, using only the
  visible index range `i0..i1`. "Fit y to the visible data" therefore comes for
  free. `scale.auto` may be a function that uPlot reads on every rescale, so the
  Auto toggle is `auto: () => autoscale` with no rebuild. With auto false, y is
  left exactly where it is when x changes.
- `setScale` silently ignores spans below 1e-16 (with more than one point) and
  swaps min and max if they are reversed. The scope clamps zoom to at least two
  sample spacings.
- Every mutation is deferred to a microtask commit. `u.batch(fn)` commits
  synchronously. The scope relies on that so `lastUpdateMs` includes the draw
  and tests can read state right after a call. `new uPlot(...)` still draws in a
  microtask.
- The `setScale` hook fires per scale key only when min or max actually changed.
  Other hooks: init, addSeries, delSeries, setCursor, setLegend, setSelect,
  setSeries, setData, setSize, drawClear, drawAxes, drawSeries, draw, ready,
  destroy, syncRect.
- Drag zoom: `cursor.drag = {x: true, y: false, dist: 0, setScale: true, uni}`.
  On mouseup with a selection it sets x from `posToVal` of the selection edges.
  With the default `dist: 0`, a 1 px jitter zooms into a 1 px window, so the
  scope uses 4. Mode 2 turns `drag.y` on by default.
- Double-click is built in. It runs `autoScaleX()`, which is first..last of
  `data[0]` with no padding, and it only listens to button 0 on `.u-over`.
- After a drag, uPlot catches the `click` on `.u-wrap` in the capture phase and
  stops it when the mouse moved. A plain click does reach `.u-over`, so markers
  can listen there. A double-click first produces two clicks.
- `cursor.bind.mousedown` replaces the default filter (button 0 and target is
  `.u-over`). The scope uses it to leave shift+left free for panning. uPlot
  already ignores the middle button.
- Coordinates: `posToVal(px, key)` and `valToPos(v, key)` work in CSS px
  relative to the plotting area. `valToPos(v, key, true)` returns canvas px.
  `posToIdx(px)` searches only the visible `i0..i1`. `bbox` is in canvas px
  (×pxRatio). `u.rect` is a cached `getBoundingClientRect()` of `.u-over`. It is
  invalidated on window resize, on scroll (a capture listener on window) and on
  mouseenter.
- The linear path builder decimates on its own. When the visible points number
  at least 4 × the plot width in canvas pixels, it builds a min, max, first and
  last value per pixel column and draws one vertical run per column
  (`drawAcc`). It still walks every point (O(n)) but emits only a few `lineTo`
  calls per column. Below 4 points per pixel it draws every point. The threshold
  is in canvas pixels, so on a 2× screen it takes twice as many points. Because
  of this, the scope hands full-resolution arrays to uPlot and does not use
  `minMaxDecimate`. Markers, peak search and readouts read those same arrays.
- TypedArrays go in as they are, with no copy. With `legend: {show: false}`,
  the `height` option is exactly the plot plus the axes.
- Mode 2 (faceted or scatter) gives per-series `[x[], y[]]` data and per-facet
  scales, but no default path builder. You write the paths yourself, and the
  stock `points` builder draws each point as a Path2D arc. Mode 2 has no equal
  aspect ratio.
- Surprise: uPlot leaves its last axis `textAlign` on the shared 2D context.
  Marker labels drawn in the `draw` hook landed outside their boxes, invisible,
  until the hook set `textAlign = 'left'` itself. The screenshot caught this.

## What uPlot could not do, and what I wrote instead

- **Wheel zoom:** uPlot has none. The scope has its own `wheel` listener on
  `.u-over` (non-passive) that zooms x about the value under the cursor, by a
  factor of `exp(deltaY · 0.0015)`, clamped to the data.
- **Pan:** uPlot has none. Shift+left-drag or middle-drag pans x, with the
  mouse followed on `window`.
- **Markers:** uPlot has no annotations. Markers are stored as sample indices
  and drawn in the `draw` hook (a dashed vertical line, a diamond at the sample
  and an M1/M2 label), so they follow zoom, pan, resize and new data. Clicks,
  peak search, delta and readout are all my own code.
- **Double-click versus marker click:** the first click of a double-click places
  a marker. The scope snapshots the markers before each click and restores them
  on `dblclick`.
- **Resizing:** uPlot does not resize itself. A ResizeObserver on the plot div
  calls `setSize`. The uPlot root is absolutely positioned, so resizing it never
  feeds back into the observer.
- **Constellation:** a plain canvas, not mode 2. Mode 2 would need a hand-written
  path builder, has no equal aspect ratio, and none of its interactions apply
  (wheel zoom on both axes, drag pan), so it would only add weight. The canvas
  draws 2×2 `fillRect`s, keeps an equal aspect ratio and draws a light grid with
  edge labels. It has its own wheel zoom about the cursor, drag pan, double-click
  reset and Auto toggle.
- **Toolbar, readout, centred message, spectrum Hz/kHz/MHz ticks:** all my own
  code.

## Gestures

- **uPlot views:**
  - Drag: zoom x.
  - Wheel: zoom x about the cursor.
  - Shift+drag or middle-drag: pan x.
  - Double-click: reset zoom.
  - Plain click (moved less than 4 px): marker. A third click moves the marker
    placed longest ago.
  - Peak: moves the active (most recent) marker to the maximum of series 0 in
    the visible range, or creates it.
- **Constellation:** wheel zooms about the cursor, drag pans (left or middle
  button), double-click resets.
- Puppeteer drives all of these with real mouse input in the check, including
  `keyboard.down('Shift')` for the pan and `{button: 'middle'}` for the
  middle-drag.

## Measured update cost

`lastUpdateMs` is transform plus synchronous uPlot draw. These are medians over
90 updates at 30 Hz in headless Chrome, at device pixel ratio 1, from the last
run:

| Scope | Median | Max |
|---|---|---|
| time, 4096 complex | 0.5 ms | 1.7 ms |
| spectrum, 4096 complex (FFT included) | 0.4 ms | 2.2 ms |
| array, 512 | 0.1 ms | 0.3 ms |
| constellation, 1024 | 0.3 ms | 0.7 ms |
| tall time, 4096 | 0.3 ms | 0.6 ms |

The figures vary by ±0.1 ms between runs. `performance.now()` is quantised to
0.1 ms on this page (it is not cross-origin isolated). The figures also cover
only JS and canvas command submission, not GPU raster or compositing.

## Decisions and deviations

- When the signal's length or rate changes, the scope resets the zoom and also
  clears the markers, because the axis meaning changed.
- When the number of series changes (real to complex) or the spectrum's
  `hasRate` changes, the uPlot instance is rebuilt, which also resets the zoom.
- In the time view, markers and peak search read series 0 (re). For complex
  data, the hover readout shows both re and im.
- The readout reserves two lines (28 px), not one. Two markers plus a delta,
  for example `M1 3.9961 kHz 28.34 dB   M2 … Δ 1.9805 kHz -82.22 dB`, do not
  fit on one 11 px monospace line at 356 px. A fixed height avoids resize
  jitter when the text changes.
- The array view shows complex input as its magnitude, computed into a reused
  buffer.

## Not done or not tested

- NaN or Inf samples are not tested. uPlot treats only `null` as a gap, so a
  NaN could disturb the y autoscale.
- HiDPI (pxRatio 2) is untested. Headless runs at DPR 1.
- There is no touch or keyboard interaction.
- The constellation is measured only at 1024 points. In the check, the tall
  scope's switch to constellation showed the real-only message, because the
  chirp is real.
- These notes cover the component on its own. Its use in the side panel and
  in a popped-out window is checked by `check.mjs`.
- Rebuild cost when the series count changes is not measured.
- Markers on the constellation were not required and are not implemented.
