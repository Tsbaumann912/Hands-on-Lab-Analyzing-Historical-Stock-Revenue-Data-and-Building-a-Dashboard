"""Academy curriculum catalog — loads markdown modules from docs/academy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
ACADEMY_DIR = _REPO_ROOT / "docs" / "academy"


@dataclass(frozen=True)
class AcademyModule:
    """One curriculum module with filesystem slug and UI metadata."""

    slug: str
    title: str
    summary: str
    filename: str
    strategy_ids: tuple[str, ...] = ()

    @property
    def path(self) -> Path:
        return ACADEMY_DIR / self.filename

    def read_markdown(self) -> str:
        if not self.path.is_file():
            return f"# Missing module\n\nExpected file: `{self.path}`"
        return self.path.read_text(encoding="utf-8")


MODULES: tuple[AcademyModule, ...] = (
    AcademyModule(
        slug="syllabus",
        title="Syllabus & Strategy Index",
        summary="Course overview and the full S01–S30 taxonomy.",
        filename="README.md",
    ),
    AcademyModule(
        slug="foundations",
        title="00 · Market Foundations",
        summary="Futures vs FX, margin, rolls, sessions.",
        filename="00-foundations.md",
    ),
    AcademyModule(
        slug="risk",
        title="01 · Risk & Execution",
        summary="Sizing, costs, drawdown limits, process.",
        filename="01-risk-and-execution.md",
    ),
    AcademyModule(
        slug="trend",
        title="02 · Trend & Momentum",
        summary="S01–S04 directional trend engines.",
        filename="02-trend-and-momentum.md",
        strategy_ids=("S01", "S02", "S03", "S04"),
    ),
    AcademyModule(
        slug="mean-reversion",
        title="03 · Mean Reversion & Range",
        summary="S05–S08 fades, VWAP, pairs.",
        filename="03-mean-reversion-and-range.md",
        strategy_ids=("S05", "S06", "S07", "S08"),
    ),
    AcademyModule(
        slug="breakout",
        title="04 · Breakout & Volatility",
        summary="S09–S11 ORB, ATR, squeeze.",
        filename="04-breakout-and-volatility.md",
        strategy_ids=("S09", "S10", "S11"),
    ),
    AcademyModule(
        slug="futures-spreads",
        title="05 · Futures Spreads & Curve",
        summary="S12–S16 calendars, crack/crush, carry, seasonality.",
        filename="05-futures-spreads-and-curve.md",
        strategy_ids=("S12", "S13", "S14", "S15", "S16"),
    ),
    AcademyModule(
        slug="forex-macro",
        title="06 · Forex Carry & Macro",
        summary="S17–S20 carry, FX momentum, PPP, rates.",
        filename="06-forex-carry-and-macro.md",
        strategy_ids=("S17", "S18", "S19", "S20"),
    ),
    AcademyModule(
        slug="news-sentiment",
        title="07 · News & Sentiment",
        summary="S21–S22 events, COT, risk sentiment.",
        filename="07-news-sentiment-and-event.md",
        strategy_ids=("S21", "S22"),
    ),
    AcademyModule(
        slug="discretionary",
        title="08 · Price Action & Flow",
        summary="S23–S25 discretionary and microstructure.",
        filename="08-discretionary-and-microstructure.md",
        strategy_ids=("S23", "S24", "S25"),
    ),
    AcademyModule(
        slug="quant-portfolio",
        title="09 · Quant Systems & Portfolios",
        summary="S26–S30 WFO, vol targeting, multi-strategy.",
        filename="09-systematic-quant-and-portfolio.md",
        strategy_ids=("S26", "S27", "S28", "S29", "S30"),
    ),
    AcademyModule(
        slug="resources",
        title="Resource Index (Videos & Links)",
        summary="Master list of CME, Babypips, research, and video hubs.",
        filename="resources.md",
    ),
)


def module_by_slug(slug: str) -> AcademyModule:
    for module in MODULES:
        if module.slug == slug:
            return module
    return MODULES[0]


def dropdown_options() -> list[dict[str, str]]:
    return [{"label": m.title, "value": m.slug} for m in MODULES]
