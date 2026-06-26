#!/usr/bin/env bash
# ── QuantTerminal Launcher ─────────────────────────────────────────────────
# Usage: ./run.sh
# Opens the trading terminal at http://127.0.0.1:8050

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtualenv if present
if [ -d ".venv/bin" ]; then
    source .venv/bin/activate
elif [ -d "venv/bin" ]; then
    source venv/bin/activate
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  QuantTerminal — Quantitative Futures Trading Terminal"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  → http://127.0.0.1:8050"
echo "  → https://pts-instructor-almost-temperatures.trycloudflare.com (public)"
echo "  Press Ctrl+C to stop"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python3 wsgi.py
