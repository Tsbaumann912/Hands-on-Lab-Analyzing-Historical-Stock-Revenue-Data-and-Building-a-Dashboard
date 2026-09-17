from __future__ import annotations

from setuptools import find_packages, setup

setup(
    name="platinum-ensemble",
    version="0.1.0",
    description="Standalone NYMEX Platinum (PL) ensemble CTA — independent of QuantTerminal",
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
            "platinum-ensemble=platinum_ensemble.cli:main",
        ],
    },
    include_package_data=True,
)
