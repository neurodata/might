#!/bin/bash
#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=01:00:00
#PBS -N my_script_job
#PBS -j oe

# Is it possible to make two environments:
# 1. W/ scikit-tree from pip because cluster can't use spin
# 2. W/ hyppo it's possible to install from the branch

# Load necessary modules
module load anaconda3

# TODO: change this accordingly
# Root directory
ROOT_DIR="/path/to/root/directory"

# Execute the Python script with the necessary arguments
python make_comight_datav2.py "$ROOT_DIR"
