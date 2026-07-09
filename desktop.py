"""
QuantTerminal — Desktop Application Launcher

Runs the Dash server locally and opens QuantTerminal in its own desktop
window (no browser tabs, no address bar), like a native application.

Window backends, tried in order:
  1. pywebview        — a true native OS window (pip install pywebview)
  2. Chrome / Edge / Chromium app-mode window (--app=URL)
  3. The system default browser (plain tab, last resort)

Run directly:
    python3 desktop.py

Or install a double-clickable desktop shortcut (Windows / macOS / Linux):
    python3 install_desktop_app.py
"""

from __future__ import annotations

import logging
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from time import monotonic, sleep

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.config import AppConfig, Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
log = logging.getLogger("quantterminal.desktop")

_HEALTH_POLL_INTERVAL_SECONDS: float = 0.25

# Chromium-family executables that support --app mode, per platform.
_BROWSER_CANDIDATES: dict[str, list[str]] = {
    "win32": [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ],
    "darwin": [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    ],
    "linux": [
        "google-chrome",
        "google-chrome-stable",
        "chromium",
        "chromium-browser",
        "microsoft-edge",
        "brave-browser",
    ],
}


@dataclass
class ServerHandle:
    """A running local QuantTerminal server that can be shut down."""

    url: str
    thread: threading.Thread
    _shutdown: object  # werkzeug BaseWSGIServer.shutdown

    def stop(self) -> None:
        try:
            self._shutdown()  # type: ignore[operator]
        except Exception:  # noqa: BLE001 — best-effort teardown on exit
            log.exception("Server shutdown raised")


def find_free_port(host: str, preferred: int) -> int:
    """Return ``preferred`` if bindable, otherwise an OS-assigned free port."""
    for candidate in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, candidate))
                return int(sock.getsockname()[1])
            except OSError:
                continue
    raise RuntimeError(f"No free TCP port available on {host}")


def start_server(cfg: AppConfig) -> ServerHandle:
    """Start the Dash/Flask server on a background thread and return a handle."""
    from werkzeug.serving import make_server

    import wsgi  # heavy import — deferred so --help stays fast

    port = find_free_port(cfg.host, cfg.port)
    httpd = make_server(cfg.host, port, wsgi.server, threaded=True)
    thread = threading.Thread(
        target=httpd.serve_forever, name="quantterminal-server", daemon=True
    )
    thread.start()
    url = f"http://{cfg.host}:{port}"
    log.info("QuantTerminal server started at %s", url)
    return ServerHandle(url=url, thread=thread, _shutdown=httpd.shutdown)


def wait_for_health(url: str, timeout_seconds: float) -> bool:
    """Poll ``/health`` until the server responds or the timeout elapses."""
    deadline = monotonic() + timeout_seconds
    health_url = f"{url}/health"
    while monotonic() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        sleep(_HEALTH_POLL_INTERVAL_SECONDS)
    return False


def find_app_mode_browser() -> str | None:
    """Locate a Chromium-family browser executable that supports --app mode."""
    platform_key = "linux" if sys.platform.startswith("linux") else sys.platform
    for candidate in _BROWSER_CANDIDATES.get(platform_key, []):
        if os.path.sep in candidate:
            if Path(candidate).exists():
                return candidate
        else:
            resolved = shutil.which(candidate)
            if resolved:
                return resolved
    return None


def _open_native_window(url: str, cfg: AppConfig) -> bool:
    """Backend 1 — pywebview native OS window. Returns True if it ran."""
    try:
        import webview  # type: ignore[import-not-found]
    except ImportError:
        return False
    try:
        webview.create_window(
            cfg.window_title,
            url,
            width=cfg.window_width,
            height=cfg.window_height,
        )
        webview.start()
        return True
    except Exception:  # noqa: BLE001 — e.g. missing GTK/QT runtime on Linux
        log.exception("pywebview failed; falling back to app-mode browser")
        return False


def _open_app_mode_window(url: str, cfg: AppConfig) -> bool:
    """Backend 2 — Chromium app-mode window. Blocks until closed."""
    browser = find_app_mode_browser()
    if browser is None:
        return False
    # A dedicated profile dir ties the process lifetime to this window, so
    # closing the window reliably ends the process (and the app with it).
    with tempfile.TemporaryDirectory(prefix="quantterminal-") as profile_dir:
        cmd = [
            browser,
            f"--app={url}",
            f"--window-size={cfg.window_width},{cfg.window_height}",
            f"--user-data-dir={profile_dir}",
            "--no-first-run",
            "--no-default-browser-check",
        ]
        log.info("Opening app window via %s", browser)
        try:
            subprocess.run(cmd, check=False)
            return True
        except OSError:
            log.exception("App-mode browser launch failed")
            return False


def _open_browser_tab(url: str) -> None:
    """Backend 3 — plain default-browser tab; blocks until Ctrl+C."""
    import webbrowser

    webbrowser.open(url)
    log.info("Opened %s in the default browser — press Ctrl+C to quit", url)
    try:
        while True:
            sleep(1)
    except KeyboardInterrupt:
        pass


def run() -> int:
    """Launch the server, open the desktop window, and clean up on close."""
    cfg = Config.from_yaml(ROOT / "config" / "default.yaml").app
    handle = start_server(cfg)

    if not wait_for_health(handle.url, cfg.startup_timeout_seconds):
        log.error(
            "Server did not become healthy within %ss", cfg.startup_timeout_seconds
        )
        handle.stop()
        return 1

    try:
        if not _open_native_window(handle.url, cfg):
            if not _open_app_mode_window(handle.url, cfg):
                _open_browser_tab(handle.url)
    finally:
        log.info("Window closed — shutting down QuantTerminal server")
        handle.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
