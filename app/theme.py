"""Shared Plotly chart theme — Apple-inspired light aesthetic."""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

CHART_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {
            "color": "#1d1d1f",
            "family": '-apple-system, BlinkMacSystemFont, "SF Pro Display", "Helvetica Neue", sans-serif',
            "size": 13,
        },
        "title": {
            "font": {
                "color": "#1d1d1f",
                "size": 17,
                "family": '-apple-system, BlinkMacSystemFont, "SF Pro Display", sans-serif',
            }
        },
        "xaxis": {
            "gridcolor": "rgba(0,0,0,0.06)",
            "linecolor": "rgba(0,0,0,0.08)",
            "tickcolor": "#86868b",
            "tickfont": {"color": "#6e6e73", "size": 11},
            "zerolinecolor": "rgba(0,0,0,0.06)",
        },
        "yaxis": {
            "gridcolor": "rgba(0,0,0,0.06)",
            "linecolor": "rgba(0,0,0,0.08)",
            "tickcolor": "#86868b",
            "tickfont": {"color": "#6e6e73", "size": 11},
            "zerolinecolor": "rgba(0,0,0,0.06)",
        },
        "legend": {
            "bgcolor": "rgba(255,255,255,0.9)",
            "bordercolor": "rgba(0,0,0,0.08)",
            "borderwidth": 1,
            "font": {"color": "#6e6e73", "size": 12},
        },
        "margin": {"l": 48, "r": 16, "t": 48, "b": 40},
        "hoverlabel": {
            "bgcolor": "#ffffff",
            "bordercolor": "#0071e3",
            "font": {"color": "#1d1d1f", "family": '-apple-system, BlinkMacSystemFont, sans-serif'},
        },
    }
}

COLORS = {
    "blue":   "#0071e3",
    "green":  "#34c759",
    "red":    "#ff3b30",
    "gold":   "#ff9f0a",
    "purple": "#af52de",
    "cyan":   "#5ac8fa",
    "orange": "#ff9500",
    "pink":   "#ff2d55",
    "muted":  "#86868b",
    "text":   "#1d1d1f",
}

pio.templates["quant_apple"] = go.layout.Template(**CHART_TEMPLATE)
pio.templates.default = "quant_apple"
