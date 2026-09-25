"""
Academy — educational curriculum for futures & forex profit strategies.
"""

from __future__ import annotations

from dash import Input, Output, callback, dcc, html

from app.academy_catalog import MODULES, dropdown_options, module_by_slug
from app.components import page_header, section_card


def _meta_block(slug: str) -> list:
    module = module_by_slug(slug)
    children: list = [html.P(module.summary, className="academy-summary")]
    if module.strategy_ids:
        children.append(
            html.Div(
                [
                    html.Span("Strategies: ", className="academy-strat-label"),
                    html.Span(", ".join(module.strategy_ids), className="academy-strat-ids"),
                ],
                className="academy-strat-row",
            )
        )
    return children


def layout() -> html.Div:
    first = MODULES[0]
    return html.Div(
        [
            page_header(
                "Academy",
                "Futures & forex strategy taxonomy — 30 named edges, videos, and web resources.",
                badge="Education",
            ),
            html.Div(
                [
                    section_card(
                        "Curriculum",
                        "Choose a module. Lessons load from docs/academy/.",
                        [
                            dcc.RadioItems(
                                id="academy-module-select",
                                options=dropdown_options(),
                                value=first.slug,
                                className="academy-radio",
                                inputClassName="academy-radio-input",
                                labelClassName="academy-radio-label",
                            ),
                            html.Div(
                                _meta_block(first.slug),
                                id="academy-module-meta",
                                className="academy-meta",
                            ),
                            html.Div(
                                [
                                    html.Span("Also explore: ", className="academy-jump-label"),
                                    dcc.Link(
                                        "Strategies lab",
                                        href="/strategy-lab",
                                        className="academy-jump",
                                    ),
                                    html.Span(" · "),
                                    dcc.Link("Risk console", href="/risk", className="academy-jump"),
                                    html.Span(" · "),
                                    dcc.Link(
                                        "Indicators", href="/indicators", className="academy-jump"
                                    ),
                                ],
                                className="academy-jumps",
                            ),
                        ],
                        className="academy-sidebar-card",
                    ),
                    section_card(
                        "Lesson",
                        None,
                        [
                            dcc.Markdown(
                                id="academy-markdown",
                                children=first.read_markdown(),
                                link_target="_blank",
                                className="academy-markdown",
                            ),
                        ],
                        className="academy-lesson-card",
                    ),
                ],
                className="academy-grid",
            ),
            html.P(
                "Educational content only — not investment advice. Futures and FX trading "
                "involve substantial risk of loss.",
                className="academy-disclaimer",
            ),
        ],
        className="page academy-page",
    )


@callback(
    Output("academy-markdown", "children"),
    Output("academy-module-meta", "children"),
    Input("academy-module-select", "value"),
)
def _load_module(slug: str):
    module = module_by_slug(slug or "syllabus")
    return module.read_markdown(), _meta_block(module.slug)
