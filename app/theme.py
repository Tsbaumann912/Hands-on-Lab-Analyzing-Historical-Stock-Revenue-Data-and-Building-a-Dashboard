"""Shared Plotly chart theme for the dark terminal aesthetic."""

from __future__ import annotations

CHART_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "#111827",
        "plot_bgcolor": "#111827",
        "font": {"color": "#f1f5f9", "family": "Inter, Segoe UI, system-ui, sans-serif", "size": 12},
        "title": {"font": {"color": "#f1f5f9", "size": 14, "family": "Inter, sans-serif"}},
        "xaxis": {
            "gridcolor": "#1e293b",
            "linecolor": "#1e293b",
            "tickcolor": "#64748b",
            "tickfont": {"color": "#64748b", "size": 11},
            "zerolinecolor": "#1e293b",
        },
        "yaxis": {
            "gridcolor": "#1e293b",
            "linecolor": "#1e293b",
            "tickcolor": "#64748b",
            "tickfont": {"color": "#64748b", "size": 11},
            "zerolinecolor": "#1e293b",
        },
        "legend": {
            "bgcolor": "rgba(17,24,39,0.8)",
            "bordercolor": "#1e293b",
            "borderwidth": 1,
            "font": {"color": "#94a3b8", "size": 11},
        },
        "margin": {"l": 50, "r": 20, "t": 40, "b": 40},
        "hoverlabel": {
            "bgcolor": "#0d1117",
            "bordercolor": "#3b82f6",
            "font": {"color": "#f1f5f9"},
        },
    }
}

COLORS = {
    "blue":   "#3b82f6",
    "green":  "#10b981",
    "red":    "#ef4444",
    "gold":   "#f59e0b",
    "purple": "#8b5cf6",
    "cyan":   "#06b6d4",
    "orange": "#f97316",
    "pink":   "#ec4899",
    "muted":  "#64748b",
    "text":   "#f1f5f9",
}

import plotly.graph_objects as go
import plotly.io as pio

# Register as a named template so all charts can use it
pio.templates["quant_dark"] = go.layout.Template(**CHART_TEMPLATE)
pio.templates.default = "quant_dark"
