# ----------------------------------------------------------------------------
#  File:        setup.py
#  Project:     Celaya Solutions Ollama Simulator
#  Created by:  Celaya Solutions, 2025
#  Author:      Christopher Celaya <chris@celayasolutions.com>
#  Description: Setup script for the Ollama Simulator package
#  Version:     1.0.0
#  License:     MIT (SPDX-Identifier: MIT)
#  Last Update: June 22, 2025
# ----------------------------------------------------------------------------

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = []
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            # Skip faiss-cpu as it requires swig to be installed
            if "faiss-cpu" not in line:
                requirements.append(line)

setup(
    name="ollama_simulator",
    version="1.0.0",
    author="Christopher Celaya",
    author_email="chris@celayasolutions.com",
    description="A simulation environment for Mother and Baby LLMs using Ollama",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/while-basic/ollama-simulator",
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "vector": ["faiss-cpu==1.7.4"],
    },
    entry_points={
        "console_scripts": [
            "ollama-simulator=ollama_simulator.main:main",
        ],
    },
) 