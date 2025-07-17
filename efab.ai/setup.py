"""
Setup configuration for Beverly Knits AI Supply Chain Optimization Planner
"""

from setuptools import setup, find_packages
import os

# Read README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="beverly-knits-ai-planner",
    version="1.0.0",
    author="Beverly Knits AI Team",
    author_email="ai-team@beverlyknits.com",
    description="AI-driven supply chain optimization for textile manufacturing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/beverlyknits/ai-supply-chain-planner",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Manufacturing",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.12",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-cov>=4.0.0",
            "black>=24.0.0",
            "flake8>=7.0.0",
            "pre-commit>=3.0.0",
        ],
        "ai": [
            "tensorflow>=2.15.0",
            "torch>=2.0.0",
            "transformers>=4.30.0",
        ],
        "db": [
            "sqlalchemy>=2.0.0",
            "psycopg2-binary>=2.9.0",
            "alembic>=1.12.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "beverly-knits-planner=main:main",
            "beverly-knits-demo=demo:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.json", "*.yml", "*.yaml"],
        "config": ["*.json", "*.yml"],
        "data": ["*.csv", "*.json"],
        "tests": ["*.py"],
    },
    zip_safe=False,
    keywords="supply-chain, ai, optimization, textile, manufacturing, procurement, planning",
    project_urls={
        "Documentation": "https://github.com/beverlyknits/ai-supply-chain-planner/wiki",
        "Source": "https://github.com/beverlyknits/ai-supply-chain-planner",
        "Tracker": "https://github.com/beverlyknits/ai-supply-chain-planner/issues",
    },
)