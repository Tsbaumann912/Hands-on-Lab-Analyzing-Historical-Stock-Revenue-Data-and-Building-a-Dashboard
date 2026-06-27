# AGENTS.md

## Cursor Cloud specific instructions

### What this repo is
A single IBM/Coursera data-science assignment notebook, `Final Assignment.ipynb`
("Extracting and Visualizing Stock Data"). It pulls stock prices with `yfinance`,
scrapes quarterly revenue tables with `requests`/`BeautifulSoup`, and plots them
with `plotly`. The notebook is an **assignment template**: most answer cells are
intentionally empty, so executing it top-to-bottom will fail (e.g.
`NameError: name 'tesla_revenue' is not defined`). That is expected and not an
environment problem.

### Environment
- Python deps are installed system-wide with `pip install --break-system-packages`
  (the startup update script handles this). There is no `requirements.txt` or
  virtualenv. `python3-venv` is **not** preinstalled on the base image, which is
  why we avoid venvs.
- `pip` installs console scripts to `~/.local/bin`, which is not on `PATH` by
  default. Prefix commands or run with `export PATH="$HOME/.local/bin:$PATH"`
  (e.g. for the `jupyter` CLI).
- Network access to Yahoo Finance and the IBM S3 course data URLs works from the
  VM, so `yfinance` and the webscrape cells fetch live data.

### Running the app (Jupyter)
- Start the dev server: `export PATH="$HOME/.local/bin:$PATH"; jupyter lab --no-browser --port 8888 --ip 0.0.0.0`
  (add `--ServerApp.token=...` for a known token). It serves `Final Assignment.ipynb`.
- Headless static export with `plotly` + `kaleido` works (a browser is present);
  `fig.write_image(...)` succeeds. Interactive `fig.show()` needs a browser, so in
  headless contexts prefer `write_image`/`write_html`.

### Lint / test
- There is no lint config and no automated test suite in this repo.
