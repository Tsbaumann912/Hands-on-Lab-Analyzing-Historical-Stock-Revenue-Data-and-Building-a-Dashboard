"""Reusable Apple-inspired UI building blocks for QuantTerminal."""

from __future__ import annotations

from dash import html
import dash_bootstrap_components as dbc


def page_header(title: str, subtitle: str, badge: str | None = None) -> html.Div:
    """Hero-style page header with optional badge."""
    children = []
    if badge:
        children.append(html.Span(badge, className="page-badge"))
    children.extend([
        html.H1(title, className="page-title"),
        html.P(subtitle, className="page-subtitle"),
    ])
    return html.Div(children, className="page-header")


def section_card(
    title: str,
    subtitle: str | None,
    children,
    *,
    className: str = "",
) -> html.Div:
    """Content card with Apple-style rounded surface."""
    header = [
        html.H3(title, className="section-title"),
    ]
    if subtitle:
        header.append(html.P(subtitle, className="section-subtitle"))
    return html.Div(
        [html.Div(header, className="section-header"), html.Div(children, className="section-body")],
        className=f"section-card {className}".strip(),
    )


def metric_tile(
    label: str,
    value: str,
    *,
    delta: str | None = None,
    delta_cls: str = "neutral",
    value_cls: str = "neutral",
    accent: str | None = None,
) -> html.Div:
    """Single KPI tile for watchlists and stats."""
    body = [
        html.Div(label, className="metric-label"),
    ]
    if accent:
        body.append(html.Div(accent, className="metric-accent"))
    body.append(html.Div(value, className=f"metric-value {value_cls}"))
    if delta:
        body.append(html.Div(delta, className=f"metric-delta {delta_cls}"))
    return html.Div(body, className="metric-tile")


def data_table(headers: list[str], rows: list, *, className: str = "") -> html.Table:
    """Clean data table with Apple-style row separators."""
    return html.Table(
        [
            html.Thead(html.Tr([html.Th(h) for h in headers])),
            html.Tbody(rows),
        ],
        className=f"data-table {className}".strip(),
    )


def control_group(label: str, control, *, hint: str | None = None) -> html.Div:
    """Form control wrapper with label."""
    children = [html.Label(label, className="control-label")]
    if hint:
        children.append(html.Span(hint, className="control-hint"))
    children.append(html.Div(control, className="control-input"))
    return html.Div(children, className="control-group")


def primary_button(text: str, button_id: str, **kwargs) -> dbc.Button:
    """Apple-style primary action button."""
    extra_class = kwargs.pop("className", "")
    classes = "btn-primary-apple " + extra_class
    return dbc.Button(
        text,
        id=button_id,
        className=classes.strip(),
        **kwargs,
    )


def status_pill(text: str, variant: str = "paper") -> html.Span:
    """Small status indicator pill."""
    return html.Span([
        html.Span(className=f"status-dot {variant}"),
        text,
    ], className=f"status-pill status-pill-{variant}")
