# -*- coding: utf-8 -*-
"""
This is a setup.py script to install ShakeNBreak
"""
from distutils.command import install_headers
import os
from setuptools import setup, find_packages
from distutils.cmd import Command
import warnings

path_to_file = os.path.dirname(os.path.abspath(__file__))

setup(
    name="compchem_toolkit",
    long_description_content_type="text/markdown",
    author="Irea Mosquera-Lois",
    readme="README.md",  # PyPI readme
    license="MIT",
    license_files=("LICENSE",),
    packages=find_packages(),
    install_requires=[
        "numpy",  # Requirement of hiphive
        "pymatgen",
        "matplotlib",
        "ase",
        "pandas",
        "seaborn",
        "monty",
        "click",
    ],
    entry_points={
        "console_scripts": [
            "analyse-procar = cli.procar:analyse_procar",
        ],
    },

)

