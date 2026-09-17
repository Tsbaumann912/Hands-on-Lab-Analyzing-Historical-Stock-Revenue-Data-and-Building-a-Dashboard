from __future__ import annotations

from setuptools import find_packages, setup

setup(
    name="silver-ensemble",
    version="0.1.0",
    description="Standalone COMEX Silver (SI) ensemble CTA — independent of QuantTerminal",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "pyyaml>=6.0",
        "scipy>=1.10",
        "yfinance>=0.2.40",
        "optuna>=3.0",
    ],
    entry_points={
        "console_scripts": [
            "silver-ensemble=silver_ensemble.cli:main",
        ],
    },
    include_package_data=True,
)
