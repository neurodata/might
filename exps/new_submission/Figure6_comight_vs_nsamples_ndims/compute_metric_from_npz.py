from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

n_dims_2_ = 6


def recompute_metric(
    root_dir,
    sim_name,
    model_name,
    output_model_name,
    n_samples,
    n_dims_1,
    n_dims_2,  
    overwrite=False
):
    output_file = root_dir / 'output' / output_model_name / sim_name / f'{sim_name}_{n_samples}_{n_dims_1}_{n_dims_2}.npz'
    if output_file.exists():
        print(f"Output file: {output_file} exists")
        return


if __name__ == "__main__":
    root_dir = Path("/Volumes/Extreme Pro/cancer")
    # root_dir = Path('/home/hao/')
    # output_dir = Path('/data/adam/')
    output_dir = root_dir

    # sim_name = "multi_modal-5-102"
    sim_name = "mean_shiftv2"
    # sim_name = "multi_modalv2"
    # sim_name = 'multi_equal'

    # model_name = "might_viewone"
    # model_name = "might_viewtwo"
    # model_name = "ksg"
    # model_name = 'knn_viewone'
    # model_name = 'knn_viewtwo'
    # model_name = 'knn'
    # model_name = 'comight-cmi'
    # model_name = "comight-perm"
    model_name = "comight"
    # model_name = 'cdcorr'
    # param_name = "sas98"
    param_name = "sas98"
    # param_name = 'cmi'
    # param_name = 'cdcorr_pvalue'
    figname = "cmi"  # TODO: change
    figname = "sas98"  # TODO: change

    n_samples_list = [2**x for x in range(8, 13)]
    n_dims_1 = 1024 - 6
    n_dims_1 = 4096 - 6
    n_repeats = 100
    print(n_samples_list)

    n_dims_2 = 6

    # save the dataframe to a csv file over n-samples
    for idx in range(n_repeats):
        for n_samples in n_samples_list:
            metric = recompute_metric(
                root_dir,
                sim_name,
                model_name,
                n_samples,
                n_dims_1,
                n_dims_2, 
            )
