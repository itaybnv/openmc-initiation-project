from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (
    (this_directory / "README.md").read_text(encoding="utf-8")
    if (this_directory / "README.md").exists()
    else ""
)

# Read requirements
requirements = []
if (this_directory / "requirements.txt").exists():
    with open(this_directory / "requirements.txt") as f:
        requirements = [
            line.strip() for line in f if line.strip() and not line.startswith("#")
        ]

setup(
    name="initiation-project",
    version="0.1.0",
    author="Itay Benvenisti",
    author_email="itaybnv@gmail.com",
    description="OpenMC Monte Carlo particle transport simulations for nuclear physics research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/itaybnv/openmc-initiation-project",
    packages=find_packages(),
    install_requires=[
        "openmc>=0.14.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "h5py>=3.8.0",
    ],
    extras_require={
        "jupyter": [
            "jupyter>=1.0.0",
            "ipykernel>=6.25.0",
            "pandas>=2.0.0",
            "seaborn>=0.12.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    keywords="openmc monte-carlo nuclear-physics simulation neutron-transport",
)
