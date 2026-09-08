"""Flask web dashboard for the standalone copper ensemble CTA."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

from flask import Flask, Response, jsonify, render_template_string, request

from copper_ensemble.data import dataframe_to_bars, load_yfinance_hg, make_synthetic_hg
from copper_ensemble.engine import BacktestEngine
from copper_ensemble.models import load_config
from copper_ensemble.validation import validate_ensemble

logger = logging.getLogger(__name__)

app = Flask(__name__)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default.yaml"

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Copper Ensemble CTA</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    :root {
      --bg0: #0f1419;
      --bg1: #1a222d;
      --bg2: #243041;
      --ink: #e8eef5;
      --muted: #8b9bb0;
      --accent: #c45c26;
      --accent2: #2a9d8f;
      --line: #2e3b4d;
      --good: #3dba7a;
      --bad: #e35d6a;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background:
        radial-gradient(1200px 600px at 10% -10%, #2a1a12 0%, transparent 55%),
        radial-gradient(900px 500px at 100% 0%, #123028 0%, transparent 50%),
        var(--bg0);
      color: var(--ink);
      min-height: 100vh;
    }
    header {
      padding: 2rem 1.5rem 1rem;
      max-width: 1100px;
      margin: 0 auto;
    }
    .brand {
      font-family: "IBM Plex Serif", Georgia, serif;
      font-size: clamp(2rem, 4vw, 2.8rem);
      letter-spacing: -0.02em;
      margin: 0 0 0.35rem;
      color: #f3ebe3;
    }
    .tag {
      color: var(--muted);
      font-size: 1rem;
      max-width: 42rem;
      line-height: 1.45;
      margin: 0;
    }
    main {
      max-width: 1100px;
      margin: 0 auto;
      padding: 0 1.5rem 3rem;
      display: grid;
      gap: 1.25rem;
    }
    .panel {
      background: linear-gradient(180deg, var(--bg1), #161d26);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 1.1rem 1.2rem;
    }
    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 0.75rem;
      align-items: end;
    }
    label {
      display: grid;
      gap: 0.3rem;
      font-size: 0.8rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    input, select, button {
      font: inherit;
      border-radius: 8px;
      border: 1px solid var(--line);
      background: var(--bg2);
      color: var(--ink);
      padding: 0.55rem 0.75rem;
    }
    button {
      background: linear-gradient(135deg, var(--accent), #a3461c);
      border: none;
      font-weight: 600;
      cursor: pointer;
      padding: 0.65rem 1.1rem;
    }
    button:disabled { opacity: 0.55; cursor: wait; }
    .metrics {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 0.75rem;
    }
    .metric {
      background: var(--bg0);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 0.85rem;
    }
    .metric .k { color: var(--muted); font-size: 0.75rem; text-transform: uppercase; }
    .metric .v { font-size: 1.35rem; font-weight: 650; margin-top: 0.2rem; font-variant-numeric: tabular-nums; }
    .good { color: var(--good); }
    .bad { color: var(--bad); }
    #status { color: var(--muted); font-size: 0.9rem; min-height: 1.2rem; }
    h2 { margin: 0 0 0.75rem; font-size: 1.05rem; font-weight: 600; }
    table { width: 100%; border-collapse: collapse; font-size: 0.92rem; }
    th, td { text-align: left; padding: 0.45rem 0.35rem; border-bottom: 1px solid var(--line); }
    th { color: var(--muted); font-weight: 500; font-size: 0.78rem; text-transform: uppercase; }
    .pass { color: var(--good); font-weight: 650; }
    .fail { color: var(--bad); font-weight: 650; }
    footer { color: var(--muted); font-size: 0.8rem; text-align: center; padding: 0 1rem 2rem; }
  </style>
</head>
<body>
  <header>
    <h1 class="brand">Copper Ensemble</h1>
    <p class="tag">Standalone COMEX HG CTA — TSMOM, carry, basis-momentum, inventory-trend, and macro-fade in one vol-targeted book. Independent of QuantTerminal.</p>
  </header>
  <main>
    <section class="panel">
      <div class="controls">
        <label>Data source
          <select id="source">
            <option value="yfinance_2008" selected>Yahoo HG 2008→now</option>
            <option value="synthetic">Synthetic HG</option>
            <option value="yfinance">Yahoo HG (5y)</option>
          </select>
        </label>
        <label>Days (synthetic)
          <input id="days" type="number" value="1500" min="400" max="5000" step="100" />
        </label>
        <button id="runBtn" type="button">Run backtest + validate</button>
      </div>
      <p id="status">Ready. First market run loads Yahoo HG from 2008 (may take 1–3 min).</p>
    </section>

    <section class="panel">
      <h2>Performance</h2>
      <div class="metrics" id="metrics"></div>
    </section>

    <section class="panel">
      <h2>Equity curve</h2>
      <div id="equityChart" style="height:360px;"></div>
    </section>

    <section class="panel">
      <h2>Sleeve ablation (Sharpe)</h2>
      <div id="ablationChart" style="height:280px;"></div>
    </section>

    <section class="panel">
      <h2>Promotion gates</h2>
      <div id="gates"></div>
    </section>
  </main>
  <footer>copper_ensemble · research / paper trading only · not financial advice</footer>
  <script>
    const fmt = (x, d=2) => (x === null || x === undefined || Number.isNaN(x)) ? '—' : Number(x).toFixed(d);
    const pct = (x) => fmt(100 * x, 2) + '%';

    function renderMetrics(m, v) {
      const items = [
        ['Sharpe', fmt(m.sharpe, 3), m.sharpe >= 0 ? 'good' : 'bad'],
        ['Total return', pct(m.total_return), m.total_return >= 0 ? 'good' : 'bad'],
        ['Max DD', pct(m.max_drawdown), 'bad'],
        ['End equity', '$' + fmt(m.end_equity, 0), ''],
        ['Fills', fmt(m.n_fills, 0), ''],
        ['DSR', fmt(v.deflated_sharpe, 3), v.deflated_sharpe >= 0.95 ? 'good' : ''],
        ['OOS retention', pct(v.oos_retention), ''],
        ['Gates', v.passed ? 'PASSED' : 'FAILED', v.passed ? 'good' : 'bad'],
      ];
      document.getElementById('metrics').innerHTML = items.map(([k, val, cls]) =>
        `<div class="metric"><div class="k">${k}</div><div class="v ${cls}">${val}</div></div>`
      ).join('');
    }

    function renderEquity(dates, equity) {
      Plotly.newPlot('equityChart', [{
        x: dates, y: equity, type: 'scatter', mode: 'lines',
        line: { color: '#c45c26', width: 2 }, fill: 'tozeroy',
        fillcolor: 'rgba(196,92,38,0.12)'
      }], {
        paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { t: 10, r: 20, b: 40, l: 55 },
        xaxis: { color: '#8b9bb0', gridcolor: '#2e3b4d' },
        yaxis: { color: '#8b9bb0', gridcolor: '#2e3b4d', tickprefix: '$' },
        font: { color: '#e8eef5' }
      }, {responsive: true, displayModeBar: false});
    }

    function renderAblation(ablations) {
      const keys = Object.keys(ablations).filter(k => k === 'all' || k.startsWith('only_'));
      const labels = keys.map(k => k.replace('only_', '').replace('all', 'ENSEMBLE'));
      const vals = keys.map(k => ablations[k].sharpe);
      Plotly.newPlot('ablationChart', [{
        x: labels, y: vals, type: 'bar',
        marker: { color: labels.map(l => l === 'ENSEMBLE' ? '#c45c26' : '#2a9d8f') }
      }], {
        paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { t: 10, r: 20, b: 50, l: 45 },
        xaxis: { color: '#8b9bb0', gridcolor: '#2e3b4d' },
        yaxis: { color: '#8b9bb0', gridcolor: '#2e3b4d', title: 'Sharpe' },
        font: { color: '#e8eef5' }
      }, {responsive: true, displayModeBar: false});
    }

    function renderGates(v) {
      const rows = (v.walk_forward || []).map(w =>
        `<tr><td>W${w.window}</td><td>${fmt(w.is_sharpe,3)}</td><td>${fmt(w.oos_sharpe,3)}</td><td>${pct(w.oos_max_dd)}</td></tr>`
      ).join('');
      document.getElementById('gates').innerHTML = `
        <p>Status: <span class="${v.passed ? 'pass' : 'fail'}">${v.passed ? 'PASSED' : 'FAILED'}</span></p>
        <ul>${(v.notes || []).map(n => `<li>${n}</li>`).join('')}</ul>
        <table>
          <thead><tr><th>Window</th><th>IS Sharpe</th><th>OOS Sharpe</th><th>OOS Max DD</th></tr></thead>
          <tbody>${rows || '<tr><td colspan="4">No WFA windows</td></tr>'}</tbody>
        </table>`;
    }

    async function run() {
      const btn = document.getElementById('runBtn');
      const status = document.getElementById('status');
      btn.disabled = true;
      status.textContent = 'Running ensemble backtest and validation…';
      try {
        const body = {
          source: document.getElementById('source').value,
          days: Number(document.getElementById('days').value || 1500),
        };
        const res = await fetch('/api/run', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'request failed');
        renderMetrics(data.metrics, data.validation);
        renderEquity(data.dates, data.equity);
        renderAblation(data.validation.ablations);
        renderGates(data.validation);
        status.textContent = `Done · ${data.n_bars} bars · ${data.date_start || ''} → ${data.date_end || ''} · ${data.metrics.n_fills} fills · ${data.source}`;
      } catch (err) {
        status.textContent = 'Error: ' + err.message;
      } finally {
        btn.disabled = false;
      }
    }

    document.getElementById('runBtn').addEventListener('click', run);
    // Do not auto-run Yahoo 2008 on every page load (slow); wait for click
    // unless synthetic is selected — still auto-run synthetic for quick demo.
    if (document.getElementById('source').value === 'synthetic') {
      run();
    } else {
      document.getElementById('status').textContent =
        'Select Run to load Yahoo HG from 2008→now and walk-forward validate.';
    }
  </script>
</body>
</html>
"""


def _load_bars(source: str, days: int) -> tuple[Any, list, str]:
    cfg = load_config(CONFIG_PATH)
    label = source
    if source == "yfinance_2008":
        try:
            df = load_yfinance_hg(cfg.contract.yfinance_ticker, start="2008-01-01")
            label = f"yfinance HG=F {df.index.min().date()}→{df.index.max().date()}"
        except Exception as exc:  # noqa: BLE001
            logger.exception("yfinance 2008 load failed; falling back to synthetic")
            df = make_synthetic_hg(n_days=days, seed=42)
            label = f"synthetic-fallback ({exc})"
    elif source == "yfinance":
        try:
            df = load_yfinance_hg(cfg.contract.yfinance_ticker, period="5y")
            label = f"yfinance HG=F 5y ({df.index.min().date()}→{df.index.max().date()})"
        except Exception as exc:  # noqa: BLE001
            logger.exception("yfinance load failed; falling back to synthetic")
            df = make_synthetic_hg(n_days=days, seed=42)
            label = f"synthetic-fallback ({exc})"
    else:
        df = make_synthetic_hg(n_days=days, seed=42)
        label = "synthetic"
    bars = dataframe_to_bars(df, symbol=cfg.contract.symbol)
    return cfg, bars, label


@app.get("/health")
def health() -> Response:
    return Response("ok", mimetype="text/plain")


@app.get("/")
def index() -> str:
    return render_template_string(DASHBOARD_HTML)


@app.post("/api/run")
def api_run() -> Any:
    payload: Dict[str, Any] = request.get_json(silent=True) or {}
    source = str(payload.get("source", "yfinance_2008"))
    days = int(payload.get("days", 1500))
    days = max(400, min(days, 5000))

    try:
        cfg, bars, label = _load_bars(source, days)
        engine = BacktestEngine(cfg)
        result = engine.run(bars)
        report = validate_ensemble(cfg, bars)
        dates = [
            b.timestamp.isoformat() if hasattr(b.timestamp, "isoformat") else str(b.timestamp)
            for b in bars
        ]
        validation = {
            "passed": report.passed,
            "deflated_sharpe": report.deflated_sharpe,
            "oos_retention": report.oos_retention,
            "notes": report.notes,
            "ablations": report.ablations,
            "walk_forward": [
                {
                    "window": w.window,
                    "is_sharpe": w.is_sharpe,
                    "oos_sharpe": w.oos_sharpe,
                    "oos_max_dd": w.oos_max_dd,
                }
                for w in report.walk_forward
            ],
        }
        return jsonify(
            {
                "metrics": result.metrics,
                "equity": result.equity_curve.tolist(),
                "dates": dates,
                "n_bars": len(bars),
                "date_start": dates[0] if dates else None,
                "date_end": dates[-1] if dates else None,
                "validation": validation,
                "source": label,
            }
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("api_run failed")
        return jsonify({"error": str(exc)}), 500


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    port = int(__import__("os").environ.get("PORT", "8060"))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
