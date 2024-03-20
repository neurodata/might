from collections import defaultdict
from pathlib import Path
from numpy.testing import assert_array_equal
import numpy as np
import pandas as pd
from sktree.stats.utils import (
    _mutual_information,
    _compute_null_distribution_coleman,
    POSITIVE_METRICS,
    METRIC_FUNCTIONS,
)

n_dims_2_ = 6


def _estimate_pvalue(
    y,
    orig_forest_proba,
    perm_forest_proba,
    metric,
    n_repeats,
    seed,
    n_jobs,
    **metric_kwargs,
):
    metric_func = METRIC_FUNCTIONS[metric]
    y = y[:, np.newaxis]
    print(y.shape, orig_forest_proba.shape, perm_forest_proba.shape)
    metric_star, metric_star_pi = _compute_null_distribution_coleman(
        y,
        orig_forest_proba,
        perm_forest_proba,
        metric,
        n_repeats=n_repeats,
        seed=seed,
        n_jobs=n_jobs,
        **metric_kwargs,
    )

    y_pred_proba_orig = np.nanmean(orig_forest_proba, axis=0)
    y_pred_proba_perm = np.nanmean(perm_forest_proba, axis=0)
    observe_stat = metric_func(y, y_pred_proba_orig, **metric_kwargs)
    permute_stat = metric_func(y, y_pred_proba_perm, **metric_kwargs)

    # metric^\pi - metric = observed test statistic, which under the
    # null is normally distributed around 0
    observe_test_stat = permute_stat - observe_stat

    # metric^\pi_j - metric_j, which is centered at 0
    null_dist = metric_star_pi - metric_star

    # compute pvalue
    if metric in POSITIVE_METRICS:
        pvalue = (1 + (null_dist <= observe_test_stat).sum()) / (1 + n_repeats)
    else:
        pvalue = (1 + (null_dist >= observe_test_stat).sum()) / (1 + n_repeats)
    return pvalue


def recompute_metric_n_samples(
    root_dir, sim_name, n_dims_1, n_dims_2, n_repeats, n_jobs=None, overwrite=False
):
    output_model_name = "comight-power"
    n_samples_list = [2**x for x in range(8, 11)]

    fname = f"results_vs_nsamples_{sim_name}_{output_model_name}_cmi-observed_{n_dims_1}_{n_repeats}.csv"
    output_file = root_dir / "output" / fname
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
            comight_fname = (
                root_dir
                / "output"
                / "comight"
                / sim_name
                / f"{sim_name}_{n_samples}_{n_dims_1}_{n_dims_2}_{idx}.npz"
            )
            comight_perm_fname = (
                root_dir
                / "output"
                / "comight-perm"
                / sim_name
                / f"{sim_name}_{n_samples}_{n_dims_1}_{n_dims_2}_{idx}.npz"
            )
            comight_data = np.load(comight_fname)
            comight_perm_data = np.load(comight_perm_fname)

            obs_posteriors = comight_data["posterior_arr"]
            obs_y = comight_data["y"]
            perm_posteriors = comight_perm_data["posterior_arr"]
            perm_y = comight_perm_data["y"]

            # mutual information for both
            y_pred_proba = np.nanmean(obs_posteriors, axis=0)
            I_XZ_Y = _mutual_information(obs_y, y_pred_proba)

            y_pred_proba = np.nanmean(perm_posteriors, axis=0)
            I_Z_Y = _mutual_information(perm_y, y_pred_proba)

            assert_array_equal(obs_y, perm_y)
            pvalue_sas98 = _estimate_pvalue(
                obs_y,
                obs_posteriors,
                perm_posteriors,
                's@98',
                10_000,
                idx,
                n_jobs,
            )

            pvalue_cmi = _estimate_pvalue(
                obs_y,
                obs_posteriors,
                perm_posteriors,
                'mi',
                10_000,
                idx,
                n_jobs,
            )

            result["sas98_pvalue"].append(pvalue_sas98)
            result["cmi_pvalue"].append(pvalue_cmi)
            result["cmi"].append(I_XZ_Y - I_Z_Y)
            result["idx"].append(idx)
            result["n_samples"].append(n_samples)
            result["n_dims_1"].append(n_dims_1)
            result["n_dims_2"].append(n_dims_2_)


if __name__ == "__main__":
    root_dir = Path("/Volumes/Extreme Pro/cancer")
    # root_dir = Path('/home/hao/')
    # output_dir = Path('/data/adam/')
    output_dir = root_dir

    # sim_name = "multi_modal-5-102"
    # sim_name = "mean_shiftv2"
    sim_name = "multi_modalv2"
    sim_name = 'multi_equal'

    sim_names = ["mean_shiftv3", "multi_modalv2", "multi_equal"]

    n_dims_1 = 4096 - 6
    n_dims_1 = 512 - 6
    n_dims_2 = 6
    n_repeats = 100
    n_jobs = -1

    for sim_name in sim_names:
        recompute_metric_n_samples(
            root_dir, sim_name, n_dims_1, n_dims_2, n_repeats, n_jobs=n_jobs, overwrite=True
        )
