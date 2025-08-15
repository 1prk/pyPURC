from setuptools import setup, find_packages

setup(
    name="pyPURC",
    version="0.1.0",
    description="Perturbed Utility Route Choice (PURC) implementation",
    author="PURC Team",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "networkx>=2.6.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "pyproj>=3.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=4.0.0",
            "mypy>=0.910",
        ]
    },
    entry_points={
        "console_scripts": [
            "purc=purc.cli:main",
        ]
    },
)