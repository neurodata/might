#!/bin/bash

# Define parameters for the job
param1_value="value1"
param2_value="value2"
param3_value="value3"

# Submit job to Sun Grid Engine
qsub << EOF
#!/bin/bash

# Request resources (adjust as needed)
#$ -N MyPythonJob           # Job name
#$ -l h_rt=01:00:00         # Runtime limit (1 hour)
#$ -l h_vmem=4G             # Memory limit (4 GB)
#$ -cwd                     # Use current working directory

# Load Python module (if needed)
module load python

# Run Python script with parameters
python your_script.py $param1_value $param2_value $param3_value

EOF