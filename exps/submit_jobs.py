import os

# Define parameter ranges
sim_types = ['direct-indirect', 'log_collider', 'confounder', 'independent']
n_samples_list = [64, 128, 256, 512, 1024]
n_features_2_list = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
n_features_2 = 4096
idx_list = list(range(100))  # 0 to 100

# Submit jobs for each parameter combination
for sim_type in sim_types:
    for n_samples in n_samples_list:
        for idx in idx_list:
            # Construct command to submit job
            command = f"qsub -N {sim_type}_{n_samples}_{n_features_2}_{idx} -l h_rt=72:00:00 -l h_vmem=4G -cwd python parallel_script_comight.py --sim_type {sim_type} --n_samples {n_samples} --n_features_2 {n_features_2} --idx {idx}"
            
            # Submit job to Sun Grid Engine
            os.system(command)

# Submit jobs for each parameter combination
n_samples = 512
for sim_type in sim_types:
    for n_features_2 in n_features_2_list:
        for idx in idx_list:
            # Construct command to submit job
            command = f"qsub -N {sim_type}_{n_samples}_{n_features_2}_{idx} -l h_rt=01:00:00 -l h_vmem=4G -cwd python parallel_script_comight.py --sim_type {sim_type} --n_samples {n_samples} --n_features_2 {n_features_2} --idx {idx}"
            
            # Submit job to Sun Grid Engine
            os.system(command)
