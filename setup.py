"""Package setup for the futures trading terminal."""

from __future__ import annotations

from setuptools import find_packages, setup

setup(
    name="futures-terminal",
    version="0.1.0",
    description="Quantitative futures trading strategy research and development terminal",
    packages=find_packages(
        exclude=["tests*", "data/raw*", "data/cache*", "logs*"]
    ),
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.26.0",
        "pandas>=2.1.0",
        "polars>=0.20.0",
        "pyarrow>=14.0.0",
        "pyyaml>=6.0.1",
        "python-dotenv>=1.0.0",
        "optuna>=3.5.0",
        "websockets>=12.0",
    ],
    extras_require={
        "data": ["databento>=0.35.0"],
        "broker": ["alpaca-py>=0.20.0"],
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.23.0",
            "pytest-cov>=4.1.0",
            "mypy>=1.8.0",
            "ruff>=0.3.0",
        ],
    },
)
