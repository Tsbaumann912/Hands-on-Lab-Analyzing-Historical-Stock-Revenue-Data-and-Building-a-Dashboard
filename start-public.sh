#!/usr/bin/env bash
# ── Start QuantTerminal with a continuously supervised public URL ─────────────
# Launches keep-public.sh in tmux so the Dash app + Cloudflare tunnel restart
# automatically if they die. Writes the live URL to PUBLIC_URL.
#
# Usage: ./start-public.sh
#
# Optional (stable hostname — recommended for production):
#   export CLOUDFLARE_TUNNEL_TOKEN=...   # named tunnel token from Zero Trust
#   export PUBLIC_BASE_URL=https://quant.example.com
#
# Without a named-tunnel token, Cloudflare issues a new trycloudflare.com
# hostname whenever the quick tunnel process is recreated. keep-public.sh
# keeps *a* public URL alive continuously and updates PUBLIC_URL.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

TMUX_CONF="${TMUX_CONF:-/exec-daemon/tmux.portal.conf}"
KEEP_SESSION="${KEEP_SESSION:-quant-keep-public}"

tmux_cmd() {
  if [ -f "$TMUX_CONF" ]; then
    tmux -f "$TMUX_CONF" "$@"
  else
    tmux "$@"
  fi
}

chmod +x "$SCRIPT_DIR/keep-public.sh"

# Restart supervisor cleanly
if tmux_cmd has-session -t "=$KEEP_SESSION" 2>/dev/null; then
  # Stop previous keep-public loop (session may hold the supervisor)
  OLD_PID="$(tmux_cmd list-panes -t "$KEEP_SESSION:0.0" -F '#{pane_pid}' 2>/dev/null || true)"
  tmux_cmd kill-session -t "$KEEP_SESSION" 2>/dev/null || true
  sleep 1
  if [ -n "${OLD_PID:-}" ]; then
    # Only kill the supervisor shell tree, not unrelated processes
    kill "$OLD_PID" 2>/dev/null || true
  fi
fi

tmux_cmd new-session -d -s "$KEEP_SESSION" -c "$SCRIPT_DIR" -- bash -l
tmux_cmd send-keys -t "$KEEP_SESSION:0.0" \
  "cd '$SCRIPT_DIR' && exec ./keep-public.sh" C-m

echo "Supervisor started in tmux session: $KEEP_SESSION"
echo "Waiting for PUBLIC_URL..."

for _ in $(seq 1 60); do
  if [ -f PUBLIC_URL ]; then
    URL="$(tr -d '[:space:]' < PUBLIC_URL)"
    if [ -n "$URL" ]; then
      # Prefer health via forced public DNS (VM local DNS can miss trycloudflare)
      HOST="${URL#https://}"
      IP="$(dig +short "$HOST" @1.1.1.1 2>/dev/null | head -1 || true)"
      if [ -n "$IP" ]; then
        CODE="$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 --resolve "$HOST:443:$IP" "$URL/health" || true)"
        if [ "$CODE" = "200" ]; then
          echo ""
          echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
          echo "  QuantTerminal is live (supervised):"
          echo "  $URL"
          echo "  Academy: ${URL}/academy"
          echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
          echo ""
          echo "URL is kept alive by ./keep-public.sh (restarts app/tunnel on failure)."
          echo "Follow logs:  tmux -f $TMUX_CONF attach -t $KEEP_SESSION"
          exit 0
        fi
      fi
      # Fall back: local health + printed URL (external DNS may still be propagating)
      if curl -sf --max-time 3 "http://127.0.0.1:${PORT:-8050}/health" >/dev/null; then
        echo ""
        echo "App is up locally. Public URL (propagating): $URL"
        echo "Academy: ${URL}/academy"
        exit 0
      fi
    fi
  fi
  sleep 1
done

echo "Timed out waiting for a healthy public URL. Check:"
echo "  tmux -f $TMUX_CONF attach -t $KEEP_SESSION"
echo "  tail -50 /tmp/cloudflared.log /tmp/quantterminal-app.log"
exit 1
