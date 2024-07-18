#!/bin/bash
# what language (only thing that is required)
# PBS -S /bin/bash

# output files
#PBS -o /data/home/ali39/pbs

# error files
#PBS -e /data/home/ali39/pbs

# select how many cpus you want to use (default=1)
#PBS -l ncpus=1

# select how many jobs you want to run concurrently (default is as many as possible)
# PBS -W max_run_subjobs=150

# Root directory
ROOT_DIR="/data/home/ali39/output/"
path=~/miniconda3/envs/ms/bin

PYTHON_FILE="comight_sa98_cluster.py"

# Path to the Python script that extracts parameters
PYTHON_SCRIPT="~/might/exps/new_submission/Figure6_comight_vs_nsamples_ndims/$PYTHON_FILE"

# Read parameters from text file and submit Python job
line=$(sed -n "${PBS_ARRAY_INDEX}p" ./parameters_comight.txt)
seed=$(echo "$line" | awk '{print $1}')
n_samples=$(echo "$line" | awk '{print $2}')
n_dims_1=$(echo "$line" | awk '{print $3}')
sim_name=$(echo "$line" | awk '{print $4}')

# Submit Python job with parameters
echo $PYTHON_SCRIPT
# echo "Submitting Python job with parameters: $seed $n_samples $n_dims_1 $sim_name"
# echo "Command: $path/python $PYTHON_SCRIPT $seed $n_samples $n_dims_1 $sim_name $ROOT_DIR"
$path/python "$PYTHON_SCRIPT" "$seed" "$n_samples" "$n_dims_1" "$sim_name" "$ROOT_DIR"

# TO SUBMIT THIS:
# qsub -J 1-1400 pbs_submission_comight.sh

# To get an interactive session:
# qsub -I -l select=1:ncpus=1:mpiprocs=1:mem=5gb,walltime=00:15:00 -j oe