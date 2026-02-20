"""Setup configuration for ml-from-scratch-playground."""

from setuptools import find_packages, setup

setup(
    name="ml-from-scratch-playground",
    version="0.1.0",
    description="Machine learning algorithms from scratch using NumPy",
    author="Your Name",
    python_requires=">=3.10",
    install_requires=["numpy", "pandas", "matplotlib", "jupyter"],
    packages=find_packages(),
)
