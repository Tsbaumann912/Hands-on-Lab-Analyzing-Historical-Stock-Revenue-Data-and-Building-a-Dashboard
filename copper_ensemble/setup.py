from __future__ import annotations

from setuptools import find_packages, setup

setup(
    name="copper-ensemble",
    version="0.2.0",
    description=(
        "Standalone COMEX Copper (HG) ensemble CTA — all-green years, "
        "max DD ≤ 30% production book (independent of QuantTerminal)"
    ),
    long_description=open("DOWNLOAD.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "pyyaml>=6.0",
        "scipy>=1.10",
        "yfinance>=0.2.40",
        "flask>=3.0",
        "plotly>=5.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "optuna>=3.0"],
    },
    entry_points={
        "console_scripts": [
            "copper-ensemble=copper_ensemble.cli:main",
            "copper-ensemble-web=copper_ensemble.web:main",
        ],
    },
    include_package_data=True,
)
