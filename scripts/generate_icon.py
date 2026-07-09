"""
Generate the QuantTerminal app icon set (PNG + ICO) into app/assets/.

Outputs:
    app/assets/icon.png     — 512×512 desktop / dock icon
    app/assets/favicon.ico  — multi-size ICO (Dash serves it automatically,
                              and the Windows shortcut installer reuses it)

Run once when the icon design changes:
    python3 scripts/generate_icon.py

Requires Pillow (build-time only — not a runtime dependency):
    pip install pillow
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT / "app" / "assets"

CANVAS_SIZE = 1024
CORNER_RADIUS = 224
BACKGROUND = (13, 27, 42, 255)       # deep navy
GRID_LINE = (34, 52, 74, 255)
BULL = (52, 199, 89, 255)            # green candle
BEAR = (255, 59, 48, 255)            # red candle
ACCENT = (0, 113, 227, 255)          # trend line blue

ICO_SIZES = [16, 24, 32, 48, 64, 128, 256]
PNG_SIZE = 512


@dataclass
class Candle:
    """One candlestick, in fractional canvas coordinates (0-1)."""

    x: float
    body_top: float
    body_bottom: float
    wick_top: float
    wick_bottom: float
    bullish: bool


CANDLES = [
    Candle(0.22, 0.58, 0.74, 0.52, 0.80, False),
    Candle(0.40, 0.44, 0.66, 0.38, 0.72, True),
    Candle(0.58, 0.34, 0.52, 0.27, 0.58, True),
    Candle(0.76, 0.24, 0.42, 0.18, 0.48, True),
]

BODY_WIDTH = 0.10
WICK_WIDTH = 0.022
GRID_ROWS = 4
GRID_MARGIN = 0.14
TREND_POINTS = [(0.14, 0.78), (0.40, 0.60), (0.60, 0.46), (0.88, 0.22)]
TREND_WIDTH = 0.030


def _px(fraction: float) -> int:
    return int(round(fraction * CANVAS_SIZE))


def draw_icon() -> Image.Image:
    """Render the master 1024×1024 icon."""
    img = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    draw.rounded_rectangle(
        [0, 0, CANVAS_SIZE - 1, CANVAS_SIZE - 1],
        radius=CORNER_RADIUS,
        fill=BACKGROUND,
    )

    for row in range(1, GRID_ROWS):
        y = _px(GRID_MARGIN + row * (1 - 2 * GRID_MARGIN) / GRID_ROWS)
        draw.line(
            [(_px(GRID_MARGIN), y), (_px(1 - GRID_MARGIN), y)],
            fill=GRID_LINE,
            width=_px(0.006),
        )

    draw.line(
        [(_px(x), _px(y)) for x, y in TREND_POINTS],
        fill=ACCENT,
        width=_px(TREND_WIDTH),
        joint="curve",
    )

    for candle in CANDLES:
        colour = BULL if candle.bullish else BEAR
        cx = _px(candle.x)
        half_wick = _px(WICK_WIDTH) // 2
        draw.rectangle(
            [cx - half_wick, _px(candle.wick_top), cx + half_wick, _px(candle.wick_bottom)],
            fill=colour,
        )
        half_body = _px(BODY_WIDTH) // 2
        draw.rounded_rectangle(
            [cx - half_body, _px(candle.body_top), cx + half_body, _px(candle.body_bottom)],
            radius=_px(0.015),
            fill=colour,
        )

    return img


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    master = draw_icon()

    png_path = ASSETS_DIR / "icon.png"
    master.resize((PNG_SIZE, PNG_SIZE), Image.LANCZOS).save(png_path)
    print(f"wrote {png_path}")

    ico_path = ASSETS_DIR / "favicon.ico"
    master.save(ico_path, sizes=[(s, s) for s in ICO_SIZES])
    print(f"wrote {ico_path}")


if __name__ == "__main__":
    main()
