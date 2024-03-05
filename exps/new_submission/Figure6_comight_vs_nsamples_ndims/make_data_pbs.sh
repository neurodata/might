#!/bin/bash
#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=01:00:00
#PBS -N my_script_job
#PBS -j oe

# Load necessary modules
module load anaconda3

# Root directory
ROOT_DIR="/path/to/root/directory"

# Execute the Python script with the necessary arguments
python make_comight_data.py "$ROOT_DIR"
