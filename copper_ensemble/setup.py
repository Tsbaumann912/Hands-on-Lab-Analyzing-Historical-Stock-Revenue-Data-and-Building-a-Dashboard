from __future__ import annotations

from setuptools import find_packages, setup

setup(
    name="copper-ensemble",
    version="0.1.0",
    description="Standalone COMEX Copper (HG) ensemble CTA — independent of QuantTerminal",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "pyyaml>=6.0",
        "scipy>=1.10",
        "yfinance>=0.2.40",
    ],
    entry_points={
        "console_scripts": [
            "copper-ensemble=copper_ensemble.cli:main",
        ],
    },
    include_package_data=True,
)
