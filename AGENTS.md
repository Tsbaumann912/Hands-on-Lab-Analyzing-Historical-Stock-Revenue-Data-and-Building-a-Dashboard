# AGENTS.md

## Cursor Cloud specific instructions

### What this repo is
**QuantTerminal** — a quantitative futures trading terminal. The main app is a
Plotly **Dash** desktop web app (entry point `wsgi.py`) backed by a decoupled
quant stack: `core/` (config, models, enums), `data/` (Databento + yfinance +
tick aggregation), `indicators/`, `strategies/`, `engine/` (backtest, metrics,
walk-forward optimizer), `risk/`, `portfolio/`, `brokers/`, and the Dash UI under
`app/` (pages, components, data_service bridge). Tuneable parameters live in
`config/default.yaml`; optimisation search bounds in `config/optuna.yaml`.

The repo also contains a legacy IBM/Coursera notebook `Final Assignment.ipynb`
("Extracting and Visualizing Stock Data"). It is an assignment template with
intentionally-empty answer cells, so running it top-to-bottom will fail
(e.g. `NameError: name 'tesla_revenue' is not defined`). That is expected and
unrelated to the terminal.

### Environment
- Python deps are installed system-wide with `pip install --break-system-packages`.
  **The default cloud install script only installs the notebook deps** (yfinance,
  pandas, jupyter, kaleido, …) — it does NOT install the terminal's web/quant
  stack. To run or test the terminal you must install:
  `pip install --break-system-packages dash dash-bootstrap-components plotly optuna vaderSentiment feedparser gunicorn pytest pytest-asyncio`
  (or `pip install --break-system-packages -r requirements.txt`).
- There is no virtualenv; `python3-venv` is not preinstalled. `pip` installs
  console scripts to `~/.local/bin`, which is not on `PATH` by default — prefix
  with `export PATH="$HOME/.local/bin:$PATH"` (e.g. for `pytest`, `jupyter`).
- Network access to Yahoo Finance works from the VM, so `yfinance` fetches live
  futures history (real daily bars back to ~2000). Databento data requires
  `DATABENTO_API_KEY` (set via Cloud Agent Secrets) — without it the Strategies
  tab falls back to yfinance + synthetic intraday.

### Running the app (Dash)
- Start the dev server: `export PATH="$HOME/.local/bin:$PATH"; python3 wsgi.py`
  → serves at `http://127.0.0.1:8050`. Health check: `GET /health` → `ok`.
- Key pages: `/` (Overview), `/charts`, `/futures`, `/indicators`,
  `/strategy-lab` (Strategies — backtest / optimize / walk-forward), `/risk`.
- Production entry: `gunicorn wsgi:server` (Procfile / Dockerfile use this).
- Headless static Plotly export works via `kaleido` (`fig.write_image(...)`);
  interactive charts render in the browser.

### Lint / test
- Test suite: `export PATH="$HOME/.local/bin:$PATH"; python3 -m pytest tests/ -q`
  (≈160 tests). Run after changes to the quant stack or data service.
- There is no enforced lint config; `ruff`/`mypy` are listed in `requirements.txt`
  but not wired into CI.
