"""Tests for the desktop launcher (desktop.py) and shortcut installer."""

from __future__ import annotations

import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

import desktop
import install_desktop_app as installer
from core.config import AppConfig, Config

ROOT = Path(__file__).resolve().parent.parent


# ── Config wiring ─────────────────────────────────────────────────────────────

class TestAppConfig:
    def test_defaults(self) -> None:
        cfg = AppConfig()
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 8050
        assert cfg.window_title == "QuantTerminal"

    def test_loaded_from_default_yaml(self) -> None:
        cfg = Config.from_yaml(ROOT / "config" / "default.yaml")
        assert cfg.app.host == "127.0.0.1"
        assert cfg.app.port == 8050
        assert cfg.app.window_width > 0
        assert cfg.app.window_height > 0
        assert cfg.app.startup_timeout_seconds > 0


# ── Port selection ────────────────────────────────────────────────────────────

class TestFindFreePort:
    def test_returns_preferred_when_free(self) -> None:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            free_port = probe.getsockname()[1]
        assert desktop.find_free_port("127.0.0.1", free_port) == free_port

    def test_falls_back_when_preferred_taken(self) -> None:
        blocker = socket.socket()
        blocker.bind(("127.0.0.1", 0))
        blocker.listen(1)
        taken = blocker.getsockname()[1]
        try:
            port = desktop.find_free_port("127.0.0.1", taken)
            assert port != taken
            assert port > 0
        finally:
            blocker.close()


# ── Health polling ────────────────────────────────────────────────────────────

class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 — stdlib naming
        status = 200 if self.path == "/health" else 404
        self.send_response(status)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def health_server():
    httpd = HTTPServer(("127.0.0.1", 0), _HealthHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


class TestWaitForHealth:
    def test_healthy_server(self, health_server: str) -> None:
        assert desktop.wait_for_health(health_server, timeout_seconds=5) is True

    def test_dead_server_times_out(self) -> None:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            dead_port = probe.getsockname()[1]
        url = f"http://127.0.0.1:{dead_port}"
        assert desktop.wait_for_health(url, timeout_seconds=0.5) is False


# ── Browser discovery ─────────────────────────────────────────────────────────

class TestFindAppModeBrowser:
    def test_returns_path_or_none(self) -> None:
        result = desktop.find_app_mode_browser()
        assert result is None or isinstance(result, str)

    def test_none_when_nothing_installed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(desktop.shutil, "which", lambda _name: None)
        monkeypatch.setattr(desktop.Path, "exists", lambda _self: False)
        assert desktop.find_app_mode_browser() is None


# ── End-to-end: real server boots and serves /health ─────────────────────────

class TestStartServer:
    def test_server_boots_and_health_responds(self) -> None:
        cfg = Config.from_yaml(ROOT / "config" / "default.yaml").app
        handle = desktop.start_server(cfg)
        try:
            assert desktop.wait_for_health(handle.url, timeout_seconds=30) is True
        finally:
            handle.stop()


# ── Shortcut installer builders ───────────────────────────────────────────────

class TestShortcutBuilders:
    def test_linux_desktop_entry(self) -> None:
        entry = installer.build_linux_desktop_entry("/usr/bin/python3")
        assert entry.startswith("[Desktop Entry]")
        assert "Name=QuantTerminal" in entry
        assert f'Exec=/usr/bin/python3 "{ROOT / "desktop.py"}"' in entry
        assert f"Icon={ROOT / 'app' / 'assets' / 'icon.png'}" in entry
        assert "Terminal=false" in entry

    def test_windows_shortcut_script(self, tmp_path: Path) -> None:
        shortcut = tmp_path / "QuantTerminal.lnk"
        script = installer.build_windows_shortcut_script(shortcut, r"C:\Python\pythonw.exe")
        assert str(shortcut) in script
        assert "pythonw.exe" in script
        assert "desktop.py" in script
        assert "$s.Save()" in script

    def test_macos_launcher_script(self) -> None:
        script = installer.build_macos_launcher_script("/usr/bin/python3")
        assert script.startswith("#!/bin/bash")
        assert f'cd "{ROOT}"' in script
        assert "desktop.py" in script

    def test_macos_info_plist_roundtrip(self) -> None:
        import plistlib

        plist = plistlib.loads(installer.build_macos_info_plist())
        assert plist["CFBundleName"] == "QuantTerminal"
        assert plist["CFBundleExecutable"] == "QuantTerminal"

    def test_install_linux_writes_shortcut(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(installer.Path, "home", classmethod(lambda _cls: tmp_path))
        monkeypatch.setattr(installer, "_desktop_dir", lambda: tmp_path / "Desktop")
        shortcut = installer.install_linux()
        assert shortcut.exists()
        assert shortcut.read_text().startswith("[Desktop Entry]")
        assert (tmp_path / ".local/share/applications/quantterminal.desktop").exists()


# ── Icon assets committed ─────────────────────────────────────────────────────

class TestIconAssets:
    def test_icon_files_exist(self) -> None:
        assert (ROOT / "app" / "assets" / "icon.png").exists()
        assert (ROOT / "app" / "assets" / "favicon.ico").exists()
