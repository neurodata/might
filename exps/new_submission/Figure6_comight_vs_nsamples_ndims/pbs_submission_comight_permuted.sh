#!/bin/bash
# what language (only thing that is required)
# PBS -S /bin/bash

# output files
#PBS -o /data/home/scurti15/scripts/might/pbs

# error files
#PBS -e /data/home/scurti15/scripts/might/pbs

# select the node you want to submit to (optional)
#PBS -l host=kvn01

# select how many cpus you want to use (default=1)
#PBS -l ncpus=1

# select how many jobs you want to run concurrently (default is as many as possible)
# PBS -W max_run_subjobs=2

# Root directory
ROOT_DIR="/path/to/root/directory"

# Path to the Python script that extracts parameters
PYTHON_SCRIPT="./comight-perm_sa98_vs_nsamples_ndims.py"

# Read parameters from text file and submit Python job
line=$(sed -n "${PBS_ARRAY_INDEX}p" ./parameters.txt)
seed=$(echo "$line" | awk '{print $1}')
n_samples=$(echo "$line" | awk '{print $2}')
n_dims_1=$(echo "$line" | awk '{print $3}')
sim_name=$(echo "$line" | awk '{print $4}')

# Submit Python job with parameters
python "$PYTHON_SCRIPT" "$seed" "$n_samples" "$n_dims_1" "$sim_name" "$ROOT_DIR"

# TO SUBMIT THIS:
# qsub -t 1-4200 your_pbs_job_script.sh
