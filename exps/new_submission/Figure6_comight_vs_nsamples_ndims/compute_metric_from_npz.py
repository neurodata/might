from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from sklearn.metrics import roc_curve
from treeple.stats.utils import (METRIC_FUNCTIONS, POSITIVE_METRICS,
                                 _compute_null_distribution_coleman,
                                 _mutual_information)

n_dims_2_ = 6


def _estimate_threshold(y_true, y_score, target_specificity=0.98, pos_label=1):
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=pos_label)

    # Find the threshold corresponding to the target specificity
    index = np.argmax(fpr >= (1 - target_specificity))
    threshold_at_specificity = thresholds[index]

    return threshold_at_specificity


def sensitivity_at_specificity(
    y_true, y_score, target_specificity=0.98, pos_label=1, threshold=None
):
    n_trees, n_samples, n_classes = y_score.shape

    # Compute nan-averaged y_score along the trees axis
    y_score_avg = np.nanmean(y_score, axis=0)

    # Extract true labels and nan-averaged predicted scores for the positive class
    y_true = y_true.ravel()
    y_score_binary = y_score_avg[:, 1]

    # Identify rows with NaN values in y_score_binary
    nan_rows = np.isnan(y_score_binary)

    # Remove NaN rows from y_score_binary and y_true
    y_score_binary = y_score_binary[~nan_rows]
    y_true = y_true[~nan_rows]

    if threshold is None:
        # Find the threshold corresponding to the target specificity
        threshold_at_specificity = _estimate_threshold(
            y_true, y_score_binary, target_specificity=0.98, pos_label=1
        )
    else:
        threshold_at_specificity = threshold

    # Use the threshold to classify predictions
    y_pred_at_specificity = (y_score_binary >= threshold_at_specificity).astype(int)

    # Compute sensitivity at the chosen specificity
    sensitivity = np.sum((y_pred_at_specificity == 1) & (y_true == 1)) / np.sum(
        y_true == 1
    )

    return sensitivity


def _estimate_sas98(y, posterior_arr, threshold=None, target_specificity=0.98):
    # Compute nan-averaged y_score along the trees axis
    y_score_avg = np.nanmean(posterior_arr, axis=0)

    # Extract true labels and nan-averaged predicted scores for the positive class
    y_true = y.ravel()
    y_score_binary = y_score_avg[:, 1]

    # Identify rows with NaN values in y_score_binary
    nan_rows = np.isnan(y_score_binary)

    # Remove NaN rows from y_score_binary and y_true
    y_score_binary = y_score_binary[~nan_rows]
    y_true = y_true[~nan_rows]

    threshold_at_specificity = _estimate_threshold(
        y_true, y_score_binary, target_specificity=0.98, pos_label=1
    )

    # generate S@S98 from posterior array
    sas98 = sensitivity_at_specificity(
        y,
        posterior_arr,
        target_specificity=target_specificity,
        threshold=threshold_at_specificity,
    )
    return sas98


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

    fname = (
        f"results_vs_nsamples_{sim_name}_{output_model_name}_{n_dims_1}_{n_repeats}.csv"
    )
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
            # pvalue_sas98 = _estimate_pvalue(
            #     obs_y,
            #     obs_posteriors,
            #     perm_posteriors,
            #     's@98',
            #     10_000,
            #     idx,
            #     n_jobs,
            # )

            # pvalue_cmi = _estimate_pvalue(
            #     obs_y,
            #     obs_posteriors,
            #     perm_posteriors,
            #     'mi',
            #     10_000,
            #     idx,
            #     n_jobs,
            # )

            # result["sas98_pvalue"].append(pvalue_sas98)
            # result["cmi_pvalue"].append(pvalue_cmi)

            # compute sas98 diffs
            sas98_obs = _estimate_sas98(obs_y, obs_posteriors)
            sas98_perm = _estimate_sas98(perm_y, perm_posteriors)

            result["sas98"].append(sas98_obs - sas98_perm)
            result["cmi"].append(I_XZ_Y - I_Z_Y)
            result["idx"].append(idx)
            result["n_samples"].append(n_samples)
            result["n_dims_1"].append(n_dims_1)
            result["n_dims_2"].append(n_dims_2_)

    df = pd.DataFrame(result)
    df.to_csv(output_file, index=False)

    # now we do the same for comight-permuted
    output_model_name = "comightperm-power"
    fname = (
        f"results_vs_nsamples_{sim_name}_{output_model_name}_{n_dims_1}_{n_repeats}.csv"
    )
    output_file = root_dir / "output" / fname
    if output_file.exists() and not overwrite:
        print(f"Output file: {output_file} exists")
        return

    result = defaultdict(list)
    for idx in range(n_repeats):
        for n_samples in n_samples_list:
            perm_idx = idx + 1 if idx <= n_repeats - 2 else 0
            comight_fname = (
                root_dir
                / "output"
                / "comight-perm"
                / sim_name
                / f"{sim_name}_{n_samples}_{n_dims_1}_{n_dims_2}_{idx}.npz"
            )
            comight_perm_fname = (
                root_dir
                / "output"
                / "comight-perm"
                / sim_name
                / f"{sim_name}_{n_samples}_{n_dims_1}_{n_dims_2}_{perm_idx}.npz"
            )
            comight_data = np.load(comight_fname)
            comight_perm_data = np.load(comight_perm_fname)

            obs_posteriors = comight_data["posterior_arr"]
            obs_y = comight_data["y"]
            perm_posteriors = comight_perm_data["posterior_arr"]
            perm_y = comight_perm_data["y"]

            assert_array_equal(obs_y, perm_y)

            # mutual information for both
            y_pred_proba = np.nanmean(obs_posteriors, axis=0)
            I_XZ_Y = _mutual_information(obs_y, y_pred_proba)

            y_pred_proba = np.nanmean(perm_posteriors, axis=0)
            I_Z_Y = _mutual_information(perm_y, y_pred_proba)

            # compute sas98 diffs
            sas98_obs = _estimate_sas98(obs_y, obs_posteriors)
            sas98_perm = _estimate_sas98(perm_y, perm_posteriors)

            result["sas98"].append(sas98_obs - sas98_perm)
            result["cmi"].append(I_XZ_Y - I_Z_Y)
            result["idx"].append(idx)
            result["n_samples"].append(n_samples)
            result["n_dims_1"].append(n_dims_1)
            result["n_dims_2"].append(n_dims_2_)

    df = pd.DataFrame(result)
    df.to_csv(output_file, index=False)


def recompute_metric_n_dims(
    root_dir, sim_name, n_samples, n_dims_2, n_repeats, n_jobs=None, overwrite=False
):
    output_model_name = "comight-power"
    n_dims_list = [2**i - 6 for i in range(3, 11)]

    fname = (
        f"results_vs_ndims_{sim_name}_{output_model_name}_{n_samples}_{n_repeats}.csv"
    )
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
        for n_dims_1 in n_dims_list:
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
            # pvalue_sas98 = _estimate_pvalue(
            #     obs_y,
            #     obs_posteriors,
            #     perm_posteriors,
            #     's@98',
            #     10_000,
            #     idx,
            #     n_jobs,
            # )

            # pvalue_cmi = _estimate_pvalue(
            #     obs_y,
            #     obs_posteriors,
            #     perm_posteriors,
            #     'mi',
            #     10_000,
            #     idx,
            #     n_jobs,
            # )

            # result["sas98_pvalue"].append(pvalue_sas98)
            # result["cmi_pvalue"].append(pvalue_cmi)

            # compute sas98 diffs
            sas98_obs = _estimate_sas98(obs_y, obs_posteriors)
            sas98_perm = _estimate_sas98(perm_y, perm_posteriors)

            result["sas98"].append(sas98_obs - sas98_perm)
            result["cmi"].append(I_XZ_Y - I_Z_Y)
            result["idx"].append(idx)
            result["n_samples"].append(n_samples)
            result["n_dims_1"].append(n_dims_1)
            result["n_dims_2"].append(n_dims_2_)

    df = pd.DataFrame(result)
    df.to_csv(output_file, index=False)

    # now we do the same for comight-permuted
    output_model_name = "comightperm-power"
    fname = (
        f"results_vs_ndims_{sim_name}_{output_model_name}_{n_samples}_{n_repeats}.csv"
    )
    output_file = root_dir / "output" / fname
    if output_file.exists() and not overwrite:
        print(f"Output file: {output_file} exists")
        return

    result = defaultdict(list)
    for idx in range(n_repeats):
        for n_dims_1 in n_dims_list:
            perm_idx = idx + 1 if idx <= n_repeats - 2 else 0
            comight_fname = (
                root_dir
                / "output"
                / "comight-perm"
                / sim_name
                / f"{sim_name}_{n_samples}_{n_dims_1}_{n_dims_2}_{idx}.npz"
            )
            comight_perm_fname = (
                root_dir
                / "output"
                / "comight-perm"
                / sim_name
                / f"{sim_name}_{n_samples}_{n_dims_1}_{n_dims_2}_{perm_idx}.npz"
            )
            comight_data = np.load(comight_fname)
            comight_perm_data = np.load(comight_perm_fname)

            obs_posteriors = comight_data["posterior_arr"]
            obs_y = comight_data["y"]
            perm_posteriors = comight_perm_data["posterior_arr"]
            perm_y = comight_perm_data["y"]

            assert_array_equal(obs_y, perm_y)

            # mutual information for both
            y_pred_proba = np.nanmean(obs_posteriors, axis=0)
            I_XZ_Y = _mutual_information(obs_y, y_pred_proba)

            y_pred_proba = np.nanmean(perm_posteriors, axis=0)
            I_Z_Y = _mutual_information(perm_y, y_pred_proba)

            # compute sas98 diffs
            sas98_obs = _estimate_sas98(obs_y, obs_posteriors)
            sas98_perm = _estimate_sas98(perm_y, perm_posteriors)

            result["sas98"].append(sas98_obs - sas98_perm)
            result["cmi"].append(I_XZ_Y - I_Z_Y)
            result["idx"].append(idx)
            result["n_samples"].append(n_samples)
            result["n_dims_1"].append(n_dims_1)
            result["n_dims_2"].append(n_dims_2_)

    df = pd.DataFrame(result)
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    root_dir = Path("/Volumes/Extreme Pro/cancer")
    # root_dir = Path('/home/hao/')
    # output_dir = Path('/data/adam/')
    output_dir = root_dir

    sim_names = ["mean_shiftv4", "multi_modalv2", "multi_equal"]

    # n_dims_1 = 4096 - 6
    n_dims_1 = 512 - 6
    n_dims_2 = 6

    # n_samples = 1024
    n_samples = 512
    n_repeats = 100
    n_jobs = -1

    for sim_name in sim_names:
        # recompute_metric_n_samples(
        #     root_dir,
        #     sim_name,
        #     n_dims_1,
        #     n_dims_2,
        #     n_repeats,
        #     n_jobs=n_jobs,
        #     overwrite=True,
        # )

        recompute_metric_n_dims(
            root_dir,
            sim_name,
            n_samples,
            n_dims_2,
            n_repeats,
            n_jobs=n_jobs,
            overwrite=True,
        )
