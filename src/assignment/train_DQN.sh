#!/bin/bash

# Activate the Conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate RL

# Set the PYTHONPATH to include the parent directory
export PYTHONPATH=$(dirname $(pwd))

# Navigate to the directory
cd /mnt/c/Users/dmgtr/OneDrive\ -\ Ecole\ Polytechnique/4A/MVA\ -\ S1/RL/mva-rl-assignment-ThomasROBERTparis/src/assignment

# Run the Python script
python training.py