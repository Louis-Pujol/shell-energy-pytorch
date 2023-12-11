# Basic setup
from setuptools import setup, find_packages

setup(
    name="pytorch-shell",
    version="0.0.1",
    author="Louis Pujol",
    description="Shell energy in PyTorch",
    packages=find_packages(),
    install_requires=["libigl", "numpy", "torch"],
)
