#!/usr/bin/env bash
# Start Copper Ensemble web UI + Cloudflare quick tunnel.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
PORT="${PORT:-8060}"
CF_BIN="/tmp/cloudflared"
LOG="/tmp/copper-cloudflared.log"
URL_FILE="$SCRIPT_DIR/PUBLIC_URL"
TMUX_CONF="${TMUX_CONF:-/exec-daemon/tmux.portal.conf}"
APP_SESSION="copper-ensemble-web"
TUNNEL_SESSION="copper-ensemble-tunnel"

tmux_cmd() { tmux -f "$TMUX_CONF" "$@"; }

if ! curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  tmux_cmd has-session -t "=$APP_SESSION" 2>/dev/null || \
    tmux_cmd new-session -d -s "$APP_SESSION" -c "$SCRIPT_DIR" -- "${SHELL:-bash}" -l
  tmux_cmd send-keys -t "$APP_SESSION:0.0" \
    "cd '$SCRIPT_DIR' && export PATH=\"\$HOME/.local/bin:\$PATH\" && PORT=${PORT} python3 -m copper_ensemble.web" C-m
  for _ in $(seq 1 40); do
    curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break
    sleep 0.5
  done
fi

if ! curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "App failed to start on :${PORT}" >&2
  exit 1
fi

if [ ! -x "$CF_BIN" ]; then
  curl -sL "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64" -o "$CF_BIN"
  chmod +x "$CF_BIN"
fi

# Kill only tunnels aimed at this port
pkill -f "cloudflared tunnel --url http://127.0.0.1:${PORT}" 2>/dev/null || true
sleep 1
: > "$LOG"

tmux_cmd has-session -t "=$TUNNEL_SESSION" 2>/dev/null || \
  tmux_cmd new-session -d -s "$TUNNEL_SESSION" -c "$SCRIPT_DIR" -- "${SHELL:-bash}" -l
tmux_cmd send-keys -t "$TUNNEL_SESSION:0.0" \
  "$CF_BIN tunnel --url http://127.0.0.1:${PORT} --no-autoupdate 2>&1 | tee $LOG" C-m

URL=""
for _ in $(seq 1 45); do
  URL=$(rg -o 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG" 2>/dev/null | head -1 || true)
  if [ -n "$URL" ]; then
    echo "$URL" > "$URL_FILE"
    echo "$URL"
    exit 0
  fi
  sleep 1
done
echo "Timed out waiting for tunnel URL; see $LOG" >&2
exit 1
