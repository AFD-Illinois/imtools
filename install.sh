#!/bin/bash

git submodule update --recursive --init

# Colors, array plots
cd external/ehtplot
pip install -e .
cd ../..

# Library and set management
cd external/hallmark
pip install -e .
cd ../..

# Install imtools
pip install -e .
