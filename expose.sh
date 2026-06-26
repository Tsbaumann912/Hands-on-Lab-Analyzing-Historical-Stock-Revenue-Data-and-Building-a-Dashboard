#!/usr/bin/env bash
# ── Expose QuantTerminal to the public internet ───────────────────────────────
# Creates a temporary Cloudflare quick tunnel to your local Dash server.
#
# Prerequisites:
#   1. App running:  python3 app.py   (or ./run.sh)
#   2. cloudflared:  auto-downloaded to /tmp/cloudflared on first run
#
# Usage: ./expose.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${PORT:-8050}"
CF_BIN="/tmp/cloudflared"
LOG="/tmp/cloudflared.log"

if ! curl -sf "http://127.0.0.1:${PORT}/" >/dev/null; then
  echo "Error: nothing listening on http://127.0.0.1:${PORT}"
  echo "Start the app first:  python3 app.py"
  exit 1
fi

if [ ! -x "$CF_BIN" ]; then
  echo "Downloading cloudflared..."
  curl -sL "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64" -o "$CF_BIN"
  chmod +x "$CF_BIN"
fi

# Stop any existing tunnel
pkill -f "cloudflared tunnel --url http://127.0.0.1:${PORT}" 2>/dev/null || true
sleep 1

echo "Starting public tunnel → http://127.0.0.1:${PORT}"
"$CF_BIN" tunnel --url "http://127.0.0.1:${PORT}" --no-autoupdate >"$LOG" 2>&1 &
TUNNEL_PID=$!

for _ in $(seq 1 30); do
  URL=$(rg -o 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG" 2>/dev/null | head -1 || true)
  if [ -n "$URL" ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  QuantTerminal is live at:"
    echo "  $URL"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Open that URL in any browser. Tunnel PID: $TUNNEL_PID"
    echo "Press Ctrl+C to stop the tunnel."
    wait "$TUNNEL_PID"
    exit 0
  fi
  sleep 1
done

echo "Timed out waiting for tunnel URL. Check $LOG"
kill "$TUNNEL_PID" 2>/dev/null || true
exit 1
