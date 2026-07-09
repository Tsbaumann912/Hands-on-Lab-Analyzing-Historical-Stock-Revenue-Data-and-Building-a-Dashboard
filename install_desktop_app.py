"""
QuantTerminal — Desktop Shortcut Installer

Creates a double-clickable "QuantTerminal" icon on your desktop that launches
the app in its own window (via desktop.py). Works on Windows, macOS and Linux.

Usage (run once, from the repo root):
    python3 install_desktop_app.py

What it creates:
    Windows — QuantTerminal.lnk on the Desktop (runs pythonw, no console box)
    macOS   — QuantTerminal.app bundle on the Desktop
    Linux   — QuantTerminal.desktop on the Desktop + app-menu entry
"""

from __future__ import annotations

import logging
import plistlib
import stat
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP_NAME = "QuantTerminal"
APP_COMMENT = "Quantitative Futures Trading Terminal"
LAUNCHER = ROOT / "desktop.py"
ICON_PNG = ROOT / "app" / "assets" / "icon.png"
ICON_ICO = ROOT / "app" / "assets" / "favicon.ico"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("quantterminal.install")


def _desktop_dir() -> Path:
    """Best-effort resolution of the user's Desktop folder."""
    if sys.platform.startswith("linux"):
        try:
            out = subprocess.run(
                ["xdg-user-dir", "DESKTOP"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
            if out:
                return Path(out)
        except (OSError, subprocess.CalledProcessError):
            pass
    return Path.home() / "Desktop"


def _windows_python() -> str:
    """Prefer pythonw.exe so no console window flashes when launching."""
    exe = Path(sys.executable)
    pythonw = exe.with_name("pythonw.exe")
    return str(pythonw) if pythonw.exists() else str(exe)


# ── Shortcut content builders (pure functions — unit tested) ─────────────────

def build_linux_desktop_entry(python: str) -> str:
    return (
        "[Desktop Entry]\n"
        "Type=Application\n"
        f"Name={APP_NAME}\n"
        f"Comment={APP_COMMENT}\n"
        f'Exec={python} "{LAUNCHER}"\n'
        f"Path={ROOT}\n"
        f"Icon={ICON_PNG}\n"
        "Terminal=false\n"
        "Categories=Office;Finance;\n"
    )


def build_windows_shortcut_script(shortcut_path: Path, python: str) -> str:
    return (
        "$ws = New-Object -ComObject WScript.Shell; "
        f"$s = $ws.CreateShortcut('{shortcut_path}'); "
        f"$s.TargetPath = '{python}'; "
        f"$s.Arguments = '\"{LAUNCHER}\"'; "
        f"$s.WorkingDirectory = '{ROOT}'; "
        f"$s.IconLocation = '{ICON_ICO}'; "
        f"$s.Description = '{APP_COMMENT}'; "
        "$s.Save()"
    )


def build_macos_launcher_script(python: str) -> str:
    return (
        "#!/bin/bash\n"
        f'cd "{ROOT}"\n'
        f'exec "{python}" "{LAUNCHER}"\n'
    )


def build_macos_info_plist() -> bytes:
    return plistlib.dumps(
        {
            "CFBundleName": APP_NAME,
            "CFBundleDisplayName": APP_NAME,
            "CFBundleIdentifier": "com.quantterminal.app",
            "CFBundleVersion": "1.0.0",
            "CFBundlePackageType": "APPL",
            "CFBundleExecutable": APP_NAME,
            "CFBundleIconFile": "icon.icns",
            "NSHighResolutionCapable": True,
        }
    )


# ── Platform installers ──────────────────────────────────────────────────────

def install_linux() -> Path:
    entry = build_linux_desktop_entry(sys.executable)
    filename = f"{APP_NAME.lower()}.desktop"

    applications_dir = Path.home() / ".local" / "share" / "applications"
    applications_dir.mkdir(parents=True, exist_ok=True)
    (applications_dir / filename).write_text(entry)

    desktop_dir = _desktop_dir()
    desktop_dir.mkdir(parents=True, exist_ok=True)
    shortcut = desktop_dir / filename
    shortcut.write_text(entry)
    shortcut.chmod(shortcut.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    # Mark as trusted so GNOME shows an icon instead of a security prompt.
    try:
        subprocess.run(
            ["gio", "set", str(shortcut), "metadata::trusted", "true"],
            capture_output=True, check=False,
        )
    except OSError:
        pass
    return shortcut


def install_windows() -> Path:
    shortcut = _desktop_dir() / f"{APP_NAME}.lnk"
    script = build_windows_shortcut_script(shortcut, _windows_python())
    subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        check=True,
    )
    return shortcut


def install_macos() -> Path:
    bundle = _desktop_dir() / f"{APP_NAME}.app"
    macos_dir = bundle / "Contents" / "MacOS"
    resources_dir = bundle / "Contents" / "Resources"
    macos_dir.mkdir(parents=True, exist_ok=True)
    resources_dir.mkdir(parents=True, exist_ok=True)

    (bundle / "Contents" / "Info.plist").write_bytes(build_macos_info_plist())

    launcher = macos_dir / APP_NAME
    launcher.write_text(build_macos_launcher_script(sys.executable))
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    # Best-effort icon conversion (sips ships with macOS).
    try:
        subprocess.run(
            ["sips", "-s", "format", "icns", str(ICON_PNG),
             "--out", str(resources_dir / "icon.icns")],
            capture_output=True, check=False,
        )
    except OSError:
        pass
    return bundle


def main() -> int:
    if not LAUNCHER.exists():
        log.error("desktop.py not found next to this script — run from the repo root")
        return 1

    if sys.platform == "win32":
        shortcut = install_windows()
    elif sys.platform == "darwin":
        shortcut = install_macos()
    else:
        shortcut = install_linux()

    log.info("Created desktop shortcut: %s", shortcut)
    log.info("Double-click '%s' on your desktop to open the terminal.", APP_NAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
