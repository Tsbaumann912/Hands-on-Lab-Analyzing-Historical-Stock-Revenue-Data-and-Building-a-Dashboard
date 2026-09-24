"""
Academy — educational curriculum for futures & forex profit strategies.
"""

from __future__ import annotations

from dash import Input, Output, State, callback, dcc, html, ALL, ctx

from app.academy_catalog import MODULES, module_by_slug
from app.components import page_header, section_card


def _module_buttons(active_slug: str) -> list:
    buttons: list = []
    for module in MODULES:
        is_active = module.slug == active_slug
        buttons.append(
            html.Button(
                [
                    html.Span(module.title, className="academy-mod-title"),
                    html.Span(module.summary, className="academy-mod-summary"),
                ],
                id={"type": "academy-mod-btn", "slug": module.slug},
                n_clicks=0,
                className=(
                    "academy-mod-btn active" if is_active else "academy-mod-btn"
                ),
                type="button",
            )
        )
    return buttons


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
            dcc.Store(id="academy-active-slug", data=first.slug),
            html.Div(
                [
                    section_card(
                        "Curriculum",
                        "Choose a module. Lessons load from docs/academy/.",
                        [
                            html.Div(
                                _module_buttons(first.slug),
                                id="academy-module-list",
                                className="academy-module-list",
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
    Output("academy-active-slug", "data"),
    Input({"type": "academy-mod-btn", "slug": ALL}, "n_clicks"),
    State("academy-active-slug", "data"),
    prevent_initial_call=True,
)
def _select_module(n_clicks: list[int], current: str) -> str:
    if not ctx.triggered_id or not any(n_clicks or []):
        return current or MODULES[0].slug
    slug = ctx.triggered_id.get("slug")
    return slug or current or MODULES[0].slug


@callback(
    Output("academy-markdown", "children"),
    Output("academy-module-meta", "children"),
    Output("academy-module-list", "children"),
    Input("academy-active-slug", "data"),
)
def _render_module(slug: str):
    module = module_by_slug(slug or "syllabus")
    return (
        module.read_markdown(),
        _meta_block(module.slug),
        _module_buttons(module.slug),
    )
