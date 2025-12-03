#!/usr/bin/env python3
"""
Setup script for pySCENT: GPU-Accelerated SCENT for Single-Cell Enhancer Target Prediction
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements_path = this_directory / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
else:
    requirements = [
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "pyarrow>=12.0.0",
        "h5py>=3.8.0",
        "anndata>=0.9.0",
        "scanpy>=1.9.0",
        "tqdm>=4.65.0",
    ]

setup(
    name="pySCENT",
    version="0.1.0",
    author="pySCENT Contributors",
    author_email="",
    description="GPU-Accelerated SCENT for Single-Cell Enhancer Target Prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YOUR_USERNAME/pySCENT",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="single-cell genomics enhancer target prediction GPU CUDA",
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "gpu": [
            # Optional GPU-specific dependencies
        ],
    },
    entry_points={
        "console_scripts": [
            "pyscent=scripts.run_scent_gpu:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    project_urls={
        "Bug Reports": "https://github.com/YOUR_USERNAME/pySCENT/issues",
        "Source": "https://github.com/YOUR_USERNAME/pySCENT",
        "Documentation": "https://github.com/YOUR_USERNAME/pySCENT#readme",
    },
)
