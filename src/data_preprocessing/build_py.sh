#!/bin/bash

# Activate the conda environment
eval "$(conda shell.bash hook)"
conda activate my-rdkit-env

# Run Python scripts
python data_repository.py
python confident_learning.py
