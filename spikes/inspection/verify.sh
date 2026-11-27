#!/usr/bin/env bash
# Full check of the spike. Run from the repository root:
#
#   bash spikes/inspection/verify.sh
#
# Layer checks in Python; the page's own transforms against NumPy; the scope
# component alone; then the page in headless Chrome against each live model, a
# recording with no model, and a model that cannot run.
set -uo pipefail

PORT=${PORT:-8791}
UI=spikes/maxgraph
SHOTS=$UI/screenshots
mkdir -p "$SHOTS"
status=0
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

serve() {
  uv run --locked python -m spikes.inspection.server --port "$PORT" "$@" >/dev/null 2>&1 &
  server=$!
  for _ in $(seq 1 40); do
    curl -s -o /dev/null "http://127.0.0.1:$PORT/api/document" && return 0
    sleep 0.25
  done
  echo "server did not start: $*"
  return 1
}

stop() {
  kill "$server" 2>/dev/null
  wait "$server" 2>/dev/null
}

echo "== build"
npm --prefix "$UI" run build 2>&1 | grep -E "error|Done" || status=1

echo "== layer checks (Python, no browser)"
uv run --locked python -m spikes.inspection.check | grep -E "FAIL|passed" || status=1

echo "== the page's transforms against NumPy (Node, no browser)"
uv run --locked python -m spikes.inspection.dsp_fixture > "$work/dsp.json" || status=1
node "$UI/dsp.check.mjs" "$work/dsp.json" | grep -E "FAIL|passed" || status=1

echo "== scope component alone, synthetic signals"
node "$UI/scope.check.mjs" | grep -E "FAIL|INFO|passed" || status=1

recording=""
for model in ber ddc; do
  echo "== live $model"
  serve --model "$model" || { status=1; continue; }
  output=$(cd "$UI" && node check.mjs "http://127.0.0.1:$PORT/" screenshots) || status=1
  echo "$output" | grep -E "FAIL|INFO|SKIP|passed"
  recording=$(echo "$output" | sed -n 's/^RECORDING //p')
  stop
done

echo "== replay of the $model recording, no model loaded"
if [ -n "$recording" ] && serve --replay "$recording"; then
  (cd "$UI" && node check.mjs "http://127.0.0.1:$PORT/" screenshots replay) | grep -E "FAIL|passed" || status=1
  stop
else
  echo "no recording to replay"
  status=1
fi

echo "== a model with an unbound input"
if serve --model unbound; then
  (cd "$UI" && node check.mjs "http://127.0.0.1:$PORT/" screenshots unbound) | grep -E "FAIL|passed" || status=1
  stop
else
  status=1
fi
exit $status
