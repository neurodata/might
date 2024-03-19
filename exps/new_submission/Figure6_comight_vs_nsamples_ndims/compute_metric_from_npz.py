from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sktree.stats.utils import _mutual_information
n_dims_2_ = 6


def recompute_metric_n_samples(
    root_dir,
    sim_name,
    n_dims_1,
    n_dims_2,
    n_repeats,
    overwrite=False
):
    output_model_name = 'comight-cmi'
    n_samples_list = [2**x for x in range(8, 13)]

    fname = f"results_vs_nsamples_{sim_name}_{output_model_name}_cmi-observed_{n_dims_1}_{n_repeats}.csv"
    output_file = root_dir / 'output' / fname
    output_file.parent.mkdir(exist_ok=True, parents=True)

    if output_file.exists() and not overwrite:
        print(f"Output file: {output_file} exists")
        return
    
    # loop through directory and extract all the posteriors
    # for comight and comight-perm -> cmi_observed
    # then for comight-perm and its combinations -> cmi_permuted
    result = defaultdict(list)
    for idx in range(n_repeats):
        for n_samples in n_samples_list:
            comight_fname = root_dir / 'output' / 'comight' / sim_name / f"{sim_name}_{n_samples}_{n_dims_1}_{n_dims_2}_{idx}.npz"
            comight_perm_fname = root_dir / 'output' / 'comight-perm' / sim_name / f"{sim_name}_{n_samples}_{n_dims_1}_{n_dims_2}_{idx}.npz"
            comight_data = np.load(comight_fname)
            comight_perm_data = np.load(comight_perm_fname)

            obs_posteriors = comight_data['posterior_arr']
            obs_y = comight_data['y']
            perm_posteriors = comight_perm_data['posterior_arr']
            perm_y = comight_perm_data['y']

            # mutual information for both
            y_pred_proba = np.nanmean(obs_posteriors, axis=0)
            I_XZ_Y = _mutual_information(obs_y, y_pred_proba)

            y_pred_proba = np.nanmean(perm_posteriors, axis=0)
            I_Z_Y = _mutual_information(perm_y, y_pred_proba)

            result['cmi'].append(I_XZ_Y - I_Z_Y)
            result['idx'].append(idx)
            result['n_samples'].append(n_samples)
            result['n_dims_1'].append(n_dims_1)
            result['n_dims_2'].append(n_dims_2_)
        
    df = pd.DataFrame(result)
    df.to_csv(output_file, index=False)

    # now we do the same for comight-permuted
    fname = f"results_vs_nsamples_{sim_name}_{output_model_name}_cmi-permuted_{n_dims_1}_{n_repeats}.csv"
    output_file = root_dir / 'output' /  fname
    if output_file.exists() and not overwrite:
        print(f"Output file: {output_file} exists")
        return
    
    # loop through directory and extract all the posteriors
    # for comight and comight-perm -> cmi_observed
    # then for comight-perm and its combinations -> cmi_permuted
    result = defaultdict(list)
    for idx in range(n_repeats):
        for n_samples in n_samples_list:
            perm_idx = idx + 1 if idx <= n_repeats - 2 else 0
            comight_fname = root_dir / 'output' / 'comight-perm' / sim_name / f"{sim_name}_{n_samples}_{n_dims_1}_{n_dims_2}_{idx}.npz"
            comight_perm_fname = root_dir / 'output' / 'comight-perm' / sim_name / f"{sim_name}_{n_samples}_{n_dims_1}_{n_dims_2}_{perm_idx}.npz"
            comight_data = np.load(comight_fname)
            comight_perm_data = np.load(comight_perm_fname)

            obs_posteriors = comight_data['posterior_arr']
            obs_y = comight_data['y']
            perm_posteriors = comight_perm_data['posterior_arr']
            perm_y = comight_perm_data['y']

            # mutual information for both
            y_pred_proba = np.nanmean(obs_posteriors, axis=0)
            I_XZ_Y = _mutual_information(obs_y, y_pred_proba)

            y_pred_proba = np.nanmean(perm_posteriors, axis=0)
            I_Z_Y = _mutual_information(perm_y, y_pred_proba)

            result['cmi'].append(I_XZ_Y - I_Z_Y)
            result['idx'].append(idx)
            result['n_samples'].append(n_samples)
            result['n_dims_1'].append(n_dims_1)
            result['n_dims_2'].append(n_dims_2_)
    
    df = pd.DataFrame(result)
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    root_dir = Path("/Volumes/Extreme Pro/cancer")
    # root_dir = Path('/home/hao/')
    # output_dir = Path('/data/adam/')
    output_dir = root_dir

    # sim_name = "multi_modal-5-102"
    sim_name = "mean_shiftv2"
    # sim_name = "multi_modalv2"
    # sim_name = 'multi_equal'
    
    n_dims_1 = 4096 - 6
    n_dims_2 = 6
    n_repeats = 100

    recompute_metric_n_samples(
        root_dir,
        sim_name,
        n_dims_1,
        n_dims_2,
        n_repeats,
        overwrite=True
    )