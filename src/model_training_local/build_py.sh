#!/bin/bash

# Activate the conda environment
eval "$(conda shell.bash hook)"
conda activate tf-wsl

# Run Python scripts
python data_repository.py
python model_training_ultis.py