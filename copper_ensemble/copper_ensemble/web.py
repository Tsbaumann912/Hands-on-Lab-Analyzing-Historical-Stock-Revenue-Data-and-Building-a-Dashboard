"""Flask web dashboard for the standalone copper ensemble CTA."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

from flask import Flask, Response, jsonify, render_template_string, request, send_file

from copper_ensemble import __version__
from copper_ensemble.data import dataframe_to_bars, load_yfinance_hg, make_synthetic_hg
from copper_ensemble.engine import BacktestEngine, calendar_year_returns
from copper_ensemble.models import load_config
from copper_ensemble.validation import validate_ensemble

logger = logging.getLogger(__name__)

app = Flask(__name__)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "default.yaml"
RELEASE_ZIP = ROOT / "releases" / f"copper-ensemble-allgreen-dd30-v{__version__}.zip"
RELEASE_SHA = RELEASE_ZIP.with_suffix(RELEASE_ZIP.suffix + ".sha256")

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
    .header-actions {
      margin-top: 1rem;
      display: flex;
      flex-wrap: wrap;
      gap: 0.75rem;
      align-items: center;
    }
    a.btn-download {
      display: inline-block;
      text-decoration: none;
      background: linear-gradient(135deg, var(--accent2), #1f7a6e);
      color: #f3fffb;
      font-weight: 600;
      border-radius: 8px;
      padding: 0.65rem 1.1rem;
    }
    a.btn-download:hover { filter: brightness(1.08); }
    .ver { color: var(--muted); font-size: 0.85rem; }
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
    <p class="tag">Standalone COMEX HG CTA — TSMOM + Fast/Slow MA + Stochastic RSI with calendar-year profit lock (flatten once YTD is green).</p>
    <div class="header-actions">
      <a class="btn-download" href="/download">Download strategy package (zip)</a>
      <span class="ver">v{{ version }} · all-green years · max DD ≤ 30%</span>
    </div>
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
      <h2>Calendar-year returns</h2>
      <div id="yearlyTable"></div>
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
        ['UPI (full)', fmt(m.upi, 3), m.upi >= 0 ? 'good' : 'bad'],
        ['Composite OOS UPI', fmt(v.composite_oos_upi, 3), v.composite_oos_upi >= 0 ? 'good' : 'bad'],
        ['Rolling OOS UPI', fmt(v.mean_oos_upi, 3), v.mean_oos_upi >= 0 ? 'good' : 'bad'],
        ['Anchored OOS UPI', fmt(v.mean_anchored_oos_upi, 3), v.mean_anchored_oos_upi >= 0 ? 'good' : 'bad'],
        ['Sharpe', fmt(m.sharpe, 3), m.sharpe >= 0 ? 'good' : 'bad'],
        ['Ulcer Index', fmt(m.ulcer_index, 2) + '%', ''],
        ['Total return', pct(m.total_return), m.total_return >= 0 ? 'good' : 'bad'],
        ['Max DD', pct(m.max_drawdown), 'bad'],
        ['Profitable years', fmt(m.n_profitable_years, 0) + ' / ' + fmt(m.n_calendar_years, 0),
          (m.n_losing_years === 0 ? 'good' : 'bad')],
        ['Min year return', pct(m.min_year_return), m.min_year_return > 0 ? 'good' : 'bad'],
        ['Fills', fmt(m.n_fills, 0), ''],
        ['Gates', v.passed ? 'PASSED' : 'FAILED', v.passed ? 'good' : 'bad'],
      ];
      document.getElementById('metrics').innerHTML = items.map(([k, val, cls]) =>
        `<div class="metric"><div class="k">${k}</div><div class="v ${cls}">${val}</div></div>`
      ).join('');
    }

    function renderYearly(yearly) {
      const years = Object.keys(yearly || {}).sort();
      if (!years.length) {
        document.getElementById('yearlyTable').innerHTML = '<p>No yearly data</p>';
        return;
      }
      const rows = years.map(y => {
        const r = yearly[y];
        const cls = r > 0 ? 'good' : (r < 0 ? 'bad' : '');
        return `<tr><td>${y}</td><td class="${cls}">${pct(r)}</td></tr>`;
      }).join('');
      document.getElementById('yearlyTable').innerHTML = `
        <table>
          <thead><tr><th>Year</th><th>Return</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>`;
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
        `<tr><td>R${w.window}</td><td>${fmt(w.is_upi,3)}</td><td>${fmt(w.oos_upi,3)}</td><td>${fmt(w.oos_sharpe,3)}</td><td>${pct(w.oos_max_dd)}</td></tr>`
      ).join('');
      const arows = (v.walk_forward_anchored || []).map(w =>
        `<tr><td>A${w.window}</td><td>${fmt(w.is_upi,3)}</td><td>${fmt(w.oos_upi,3)}</td><td>${fmt(w.oos_sharpe,3)}</td><td>${pct(w.oos_max_dd)}</td></tr>`
      ).join('');
      document.getElementById('gates').innerHTML = `
        <p>Status: <span class="${v.passed ? 'pass' : 'fail'}">${v.passed ? 'PASSED' : 'FAILED'}</span>
        · Composite OOS UPI: <b>${fmt(v.composite_oos_upi,3)}</b></p>
        <ul>${(v.notes || []).map(n => `<li>${n}</li>`).join('')}</ul>
        <h3 style="margin-top:1rem;font-size:0.95rem;color:#8b9bb0">Rolling WFA</h3>
        <table>
          <thead><tr><th>Win</th><th>IS UPI</th><th>OOS UPI</th><th>OOS Sharpe</th><th>OOS Max DD</th></tr></thead>
          <tbody>${rows || '<tr><td colspan="5">No rolling windows</td></tr>'}</tbody>
        </table>
        <h3 style="margin-top:1rem;font-size:0.95rem;color:#8b9bb0">Anchored WFA</h3>
        <table>
          <thead><tr><th>Win</th><th>IS UPI</th><th>OOS UPI</th><th>OOS Sharpe</th><th>OOS Max DD</th></tr></thead>
          <tbody>${arows || '<tr><td colspan="5">No anchored windows</td></tr>'}</tbody>
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
        renderYearly(data.yearly_returns || {});
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
    return render_template_string(DASHBOARD_HTML, version=__version__)


@app.get("/download")
def download_package() -> Any:
    """Serve the pre-built strategy zip for offline install."""
    if not RELEASE_ZIP.is_file():
        return (
            jsonify(
                {
                    "error": "Release zip not found. Run: bash scripts/build_release.sh",
                    "expected": str(RELEASE_ZIP),
                }
            ),
            404,
        )
    return send_file(
        RELEASE_ZIP,
        as_attachment=True,
        download_name=RELEASE_ZIP.name,
        mimetype="application/zip",
    )


@app.get("/download/sha256")
def download_sha256() -> Any:
    if not RELEASE_SHA.is_file():
        return jsonify({"error": "sha256 file missing", "expected": str(RELEASE_SHA)}), 404
    return send_file(RELEASE_SHA, as_attachment=True, download_name=RELEASE_SHA.name)


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
            "mean_oos_sharpe": report.mean_oos_sharpe,
            "mean_is_sharpe": report.mean_is_sharpe,
            "mean_oos_upi": report.mean_oos_upi,
            "mean_anchored_oos_upi": report.mean_anchored_oos_upi,
            "composite_oos_upi": report.composite_oos_upi,
            "target_oos_met": report.target_oos_met,
            "notes": report.notes,
            "ablations": report.ablations,
            "walk_forward": [
                {
                    "window": w.window,
                    "scheme": w.scheme,
                    "is_sharpe": w.is_sharpe,
                    "oos_sharpe": w.oos_sharpe,
                    "is_upi": w.is_upi,
                    "oos_upi": w.oos_upi,
                    "oos_ulcer_index": w.oos_ulcer_index,
                    "oos_max_dd": w.oos_max_dd,
                }
                for w in report.walk_forward
            ],
            "walk_forward_anchored": [
                {
                    "window": w.window,
                    "scheme": w.scheme,
                    "is_sharpe": w.is_sharpe,
                    "oos_sharpe": w.oos_sharpe,
                    "is_upi": w.is_upi,
                    "oos_upi": w.oos_upi,
                    "oos_ulcer_index": w.oos_ulcer_index,
                    "oos_max_dd": w.oos_max_dd,
                }
                for w in report.walk_forward_anchored
            ],
        }
        return jsonify(
            {
                "metrics": result.metrics,
                "equity": result.equity_curve.tolist(),
                "dates": dates,
                "yearly_returns": {
                    str(y): float(r)
                    for y, r in calendar_year_returns(
                        result.equity_curve, [b.timestamp for b in bars]
                    ).items()
                },
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
