#!/usr/bin/env bash
# ── Keep QuantTerminal + Cloudflare tunnel running forever ───────────────────
# Restarts the Dash app and quick tunnel if either exits, and refreshes
# PUBLIC_URL whenever a new trycloudflare hostname is issued.
#
# Usage (foreground):  ./keep-public.sh
# Usage (detached):    ./start-public.sh   # launches this in tmux
#
# Note: Cloudflare *quick* tunnels get a new hostname after each tunnel restart.
# For a forever-stable hostname you need a named Cloudflare Tunnel token
# (CLOUDFLARE_TUNNEL_TOKEN) or a PaaS deploy (Railway / Hugging Face Spaces).

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${PORT:-8050}"
CF_BIN="${CF_BIN:-/tmp/cloudflared}"
LOG="${LOG:-/tmp/cloudflared.log}"
APP_LOG="${APP_LOG:-/tmp/quantterminal-app.log}"
PUBLIC_FILE="${PUBLIC_FILE:-$SCRIPT_DIR/PUBLIC_URL}"
POLL_SECS="${POLL_SECS:-5}"
APP_PID=""
TUNNEL_PID=""

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

ensure_cloudflared() {
  if [ -x "$CF_BIN" ]; then
    return
  fi
  log "Downloading cloudflared → $CF_BIN"
  curl -sL "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64" -o "$CF_BIN"
  chmod +x "$CF_BIN"
}

app_healthy() {
  curl -sf --max-time 3 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1
}

start_app() {
  if app_healthy; then
    # Adopt existing listener
    APP_PID="$(pgrep -f 'python3 wsgi.py' | head -1 || true)"
    log "App already healthy (pid=${APP_PID:-unknown})"
    return
  fi

  # Stop stale python wsgi by PID only
  for pid in $(pgrep -f 'python3 wsgi.py' || true); do
    log "Stopping stale app pid=$pid"
    kill "$pid" 2>/dev/null || true
  done
  sleep 1

  log "Starting QuantTerminal on :${PORT}"
  nohup python3 wsgi.py >"$APP_LOG" 2>&1 &
  APP_PID=$!

  for _ in $(seq 1 40); do
    if app_healthy; then
      log "App ready (pid=$APP_PID)"
      return
    fi
    # Process died?
    if ! kill -0 "$APP_PID" 2>/dev/null; then
      log "App exited early — see $APP_LOG"
      tail -20 "$APP_LOG" || true
      return 1
    fi
    sleep 1
  done
  log "App start timed out — see $APP_LOG"
  return 1
}

extract_tunnel_url() {
  rg -o 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG" 2>/dev/null | tail -1 || true
}

start_tunnel() {
  ensure_cloudflared

  # Prefer named tunnel token for a stable hostname
  if [ -n "${CLOUDFLARE_TUNNEL_TOKEN:-}" ]; then
    for pid in $(pgrep -f 'cloudflared tunnel' || true); do
      kill "$pid" 2>/dev/null || true
    done
    sleep 1
    : >"$LOG"
    log "Starting named Cloudflare tunnel (token auth)"
    nohup "$CF_BIN" tunnel run --token "$CLOUDFLARE_TUNNEL_TOKEN" >"$LOG" 2>&1 &
    TUNNEL_PID=$!
    if [ -n "${PUBLIC_BASE_URL:-}" ]; then
      echo "$PUBLIC_BASE_URL" >"$PUBLIC_FILE"
      log "Stable public URL: $PUBLIC_BASE_URL"
    else
      log "Named tunnel running (pid=$TUNNEL_PID). Set PUBLIC_BASE_URL to publish the hostname."
    fi
    return
  fi

  for pid in $(pgrep -f 'cloudflared tunnel --url' || true); do
    log "Stopping old quick tunnel pid=$pid"
    kill "$pid" 2>/dev/null || true
  done
  sleep 1

  : >"$LOG"
  log "Starting Cloudflare quick tunnel → http://127.0.0.1:${PORT}"
  nohup "$CF_BIN" tunnel --url "http://127.0.0.1:${PORT}" --no-autoupdate >"$LOG" 2>&1 &
  TUNNEL_PID=$!

  local url=""
  for _ in $(seq 1 45); do
    url="$(extract_tunnel_url)"
    if [ -n "$url" ]; then
      echo "$url" >"$PUBLIC_FILE"
      log "Public URL: $url"
      log "Academy:    ${url}/academy"
      return
    fi
    if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
      log "Tunnel exited early — see $LOG"
      tail -30 "$LOG" || true
      return 1
    fi
    sleep 1
  done
  log "Timed out waiting for tunnel URL — see $LOG"
  return 1
}

tunnel_alive() {
  if [ -n "${CLOUDFLARE_TUNNEL_TOKEN:-}" ]; then
    pgrep -f 'cloudflared tunnel' >/dev/null 2>&1
    return $?
  fi
  pgrep -f 'cloudflared tunnel --url' >/dev/null 2>&1
}

cleanup() {
  log "Shutting down keep-public (leaving app/tunnel running)"
  exit 0
}

trap cleanup INT TERM

log "QuantTerminal keep-public supervisor starting"
start_app || true
start_tunnel || true

while true; do
  if ! app_healthy; then
    log "App unhealthy — restarting"
    start_app || true
  fi
  if ! tunnel_alive; then
    log "Tunnel down — restarting"
    start_tunnel || true
  else
    # Refresh PUBLIC_URL if log gained a newer hostname
    url="$(extract_tunnel_url)"
    if [ -n "$url" ]; then
      prev=""
      [ -f "$PUBLIC_FILE" ] && prev="$(tr -d '[:space:]' <"$PUBLIC_FILE")"
      if [ "$url" != "$prev" ]; then
        echo "$url" >"$PUBLIC_FILE"
        log "PUBLIC_URL updated → $url"
      fi
    fi
  fi
  sleep "$POLL_SECS"
done
