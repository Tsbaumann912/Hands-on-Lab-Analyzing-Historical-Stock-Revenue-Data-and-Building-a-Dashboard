#!/usr/bin/env bash
# ── Start QuantTerminal + Cloudflare public tunnel ────────────────────────────
# Starts the Dash server and a Cloudflare quick tunnel so you can open the app
# from any browser.
#
# Usage: ./start-public.sh
#
# Canonical URL (while tunnel is running):
#   https://pts-instructor-almost-temperatures.trycloudflare.com
#
# Note: trycloudflare.com URLs are tied to the running cloudflared process.
# If you restart the tunnel, run ./expose.sh to get the new URL and update PUBLIC_URL.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${PORT:-8050}"
CF_BIN="/tmp/cloudflared"
LOG="/tmp/cloudflared.log"
TMUX_CONF="${TMUX_CONF:-/exec-daemon/tmux.portal.conf}"
APP_SESSION="quant-terminal-dev"
TUNNEL_SESSION="cloudflared-tunnel"

tmux_cmd() {
  tmux -f "$TMUX_CONF" "$@"
}

start_app() {
  if curl -sf "http://127.0.0.1:${PORT}/" >/dev/null 2>&1; then
    echo "App already running on port ${PORT}"
    return
  fi

  if ! tmux_cmd has-session -t "=$APP_SESSION" 2>/dev/null; then
    tmux_cmd new-session -d -s "$APP_SESSION" -c "$SCRIPT_DIR" -- "${SHELL:-bash}" -l
  fi

  tmux_cmd send-keys -t "$APP_SESSION:0.0" "cd '$SCRIPT_DIR' && python3 wsgi.py" C-m
  echo "Starting app on port ${PORT}..."

  for _ in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:${PORT}/" >/dev/null 2>&1; then
      echo "App ready at http://127.0.0.1:${PORT}"
      return
    fi
    sleep 1
  done

  echo "Error: app did not start on port ${PORT}" >&2
  exit 1
}

start_tunnel() {
  if [ ! -x "$CF_BIN" ]; then
    echo "Downloading cloudflared..."
    curl -sL "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64" -o "$CF_BIN"
    chmod +x "$CF_BIN"
  fi

  # Reuse existing tunnel if still running
  if pgrep -f "cloudflared tunnel --url http://127.0.0.1:${PORT}" >/dev/null 2>&1; then
    URL=$(rg -o 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG" 2>/dev/null | head -1 || true)
    if [ -z "$URL" ] && [ -f PUBLIC_URL ]; then
      URL=$(tr -d '[:space:]' < PUBLIC_URL)
    fi
    if [ -n "$URL" ]; then
      echo "Tunnel already running: $URL"
      return
    fi
  fi

  pkill -f "cloudflared tunnel --url http://127.0.0.1:${PORT}" 2>/dev/null || true
  sleep 1

  if ! tmux_cmd has-session -t "=$TUNNEL_SESSION" 2>/dev/null; then
    tmux_cmd new-session -d -s "$TUNNEL_SESSION" -c "$SCRIPT_DIR" -- "${SHELL:-bash}" -l
  fi

  tmux_cmd send-keys -t "$TUNNEL_SESSION:0.0" \
    "'$CF_BIN' tunnel --url http://127.0.0.1:${PORT} --no-autoupdate 2>&1 | tee '$LOG'" C-m

  for _ in $(seq 1 30); do
    URL=$(rg -o 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG" 2>/dev/null | head -1 || true)
    if [ -n "$URL" ]; then
      echo "$URL" > PUBLIC_URL
      echo ""
      echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
      echo "  QuantTerminal is live at:"
      echo "  $URL"
      echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
      return
    fi
    sleep 1
  done

  echo "Error: timed out waiting for tunnel URL. Check $LOG" >&2
  exit 1
}

start_app
start_tunnel
