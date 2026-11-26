#!/usr/bin/env bash
# Full check of the spike: layer checks in Python, then the page in headless
# Chrome against each live model, then against a recording with no model.
# Run from the repository root:  bash spikes/inspection/verify.sh
set -uo pipefail

PORT=${PORT:-8791}
UI=spikes/maxgraph
SHOTS=$UI/screenshots
mkdir -p "$SHOTS"
status=0

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

echo "== layer checks (Python, no browser)"
uv run --locked python -m spikes.inspection.check | tail -1 || status=1

recording=""
for model in ber ddc; do
  echo "== live $model"
  serve --model "$model" || { status=1; continue; }
  output=$(cd "$UI" && node check.mjs "http://127.0.0.1:$PORT/" screenshots) || status=1
  echo "$output" | grep -E "FAIL|INFO|passed"
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
exit $status
