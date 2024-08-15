from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from joblib import Parallel, delayed
from numpy.testing import assert_array_equal
from numpy.typing import ArrayLike
from scipy.stats import entropy
from sklearn.metrics import (balanced_accuracy_score, mean_absolute_error,
                             mean_squared_error, roc_auc_score, roc_curve)


def _mutual_information(y_true: ArrayLike, y_pred_proba: ArrayLike) -> float:
    """Compute estimate of mutual information for supervised classification setting.

    Parameters
    ----------
    y_true : ArrayLike of shape (n_samples,)
        The true labels.
    y_pred_proba : ArrayLike of shape (n_samples, n_outputs)
        Posterior probabilities.

    Returns
    -------
    float :
        The estimated MI.
    """
    if y_true.squeeze().ndim != 1:
        raise ValueError(f"y_true must be 1d, not {y_true.shape}")

    # entropy averaged over n_samples
    H_YX = np.mean(entropy(y_pred_proba, base=np.exp(1), axis=1))
    # empirical count of each class (n_classes)
    _, counts = np.unique(y_true, return_counts=True)
    H_Y = entropy(counts, base=np.exp(1))
    return H_Y - H_YX


def _cond_entropy(y_true: ArrayLike, y_pred_proba: ArrayLike) -> float:
    """Compute estimate of entropy for supervised classification setting.

    H(Y | X)

    Parameters
    ----------
    y_true : ArrayLike of shape (n_samples,)
        The true labels. Not used in computation of the entropy.
    y_pred_proba : ArrayLike of shape (n_samples, n_outputs)
        Posterior probabilities.

    Returns
    -------
    float :
        The estimated MI.
    """
    if y_true.squeeze().ndim != 1:
        raise ValueError(f"y_true must be 1d, not {y_true.shape}")

    # entropy averaged over n_samples
    H_YX = np.mean(entropy(y_pred_proba, base=np.exp(1), axis=1))
    return H_YX


# Define function to compute the sensitivity at 98% specificity
def _SA98(y_true: ArrayLike, y_pred_proba: ArrayLike, max_fpr=0.02) -> float:
    """Compute the sensitivity at 98% specificity.

    Parameters
    ----------
    y_true : ArrayLike of shape (n_samples,)
        The true labels.
    y_pred_proba : ArrayLike of shape (n_samples, n_outputs)
        Posterior probabilities.
    max_fpr : float, optional. Default=0.02.

    Returns
    -------
    float :
        The estimated SA98.
    """
    if y_true.squeeze().ndim != 1:
        raise ValueError(f"y_true must be 1d, not {y_true.shape}")
    if 0 in y_true or -1 in y_true:
        fpr, tpr, thresholds = roc_curve(
            y_true, y_pred_proba[:, 1], pos_label=1, drop_intermediate=False
        )
    else:
        fpr, tpr, thresholds = roc_curve(
            y_true, y_pred_proba[:, 1], pos_label=2, drop_intermediate=False
        )
    s98 = max([tpr for (fpr, tpr) in zip(fpr, tpr) if fpr <= max_fpr])
    return s98


METRIC_FUNCTIONS = {
    "mse": mean_squared_error,
    "mae": mean_absolute_error,
    "balanced_accuracy": balanced_accuracy_score,
    "auc": roc_auc_score,
    "mi": _mutual_information,
    "cond_entropy": _cond_entropy,
    "s@98": _SA98,
}


def _compute_null_distribution_coleman(
    y_test: ArrayLike,
    y_pred_proba_normal: ArrayLike,
    y_pred_proba_perm: ArrayLike,
    metric: str = "mse",
    n_repeats: int = 10_000,
    seed: Optional[int] = None,
    n_jobs: Optional[int] = None,
    **metric_kwargs,
) -> Tuple[ArrayLike, ArrayLike]:
    """Compute null distribution using Coleman method.

    The null distribution is comprised of two forests.

    Parameters
    ----------
    y_test : ArrayLike of shape (n_samples, n_outputs)
        The output matrix.
    y_pred_proba_normal : ArrayLike of shape (n_estimators_normal, n_samples, n_outputs)
        The predicted posteriors from the normal forest. Some of the trees
        may have nans predicted in them, which means the tree used these samples
        for training and not for prediction.
    y_pred_proba_perm : ArrayLike of shape (n_estimators_perm, n_samples, n_outputs)
        The predicted posteriors from the permuted forest. Some of the trees
        may have nans predicted in them, which means the tree used these samples
        for training and not for prediction.
    metric : str, optional
        The metric, which to compute the null distribution of statistics, by default 'mse'.
    n_repeats : int, optional
        The number of times to sample the null, by default 1000.
    seed : int, optional
        Random seed, by default None.
    metric_kwargs : dict, optional
        Keyword arguments to pass to the metric function.

    Returns
    -------
    metric_star : ArrayLike of shape (n_samples,)
        An array of the metrics computed on half the trees.
    metric_star_pi : ArrayLike of shape (n_samples,)
        An array of the metrics computed on the other half of the trees.
    """
    # sample two sets of equal number of trees from the combined forest these are the posteriors
    # (n_estimators * 2, n_samples, n_outputs)
    all_y_pred = np.concatenate((y_pred_proba_normal, y_pred_proba_perm), axis=0)

    n_estimators, _, _ = y_pred_proba_normal.shape
    n_samples_test = len(y_test)
    if all_y_pred.shape[1] != n_samples_test:
        raise RuntimeError(
            f"The number of samples in `all_y_pred` {len(all_y_pred)} "
            f"is not equal to 2 * n_samples_test {2 * n_samples_test}"
        )

    # create index array of [1, ..., 2N] to slice into `all_y_pred` the stacks of trees
    y_pred_ind_arr = np.arange((2 * n_estimators), dtype=int)

    metric_star = np.zeros((n_repeats,))
    metric_star_pi = np.zeros((n_repeats,))

    # compute non-nan-mask array
    # non_nan_mask = np.ma.masked_invalid(all_y_pred)
    # print(non_nan_mask.shape)
    # all_y_pred = np.nan_to_num(all_y_pred, nan=0)

    # generate the random seeds for the parallel jobs
    # ss = np.random.SeedSequence(seed)
    rng = np.random.default_rng(seed)
    # for i, seed in zip(range(n_repeats), rng.integers(0, 2**32, n_repeats)):
    #     now = time()
    #     _parallel_build_null_forests(
    #         y_pred_ind_arr,
    #         n_estimators,
    #         all_y_pred,
    #         y_test,
    #         # non_nan_mask,
    #         seed,
    #         metric,
    #         **metric_kwargs,
    #     )
    #     print(f"Time taken: {time() - now}")
    out = Parallel(n_jobs=n_jobs)(
        delayed(_parallel_build_null_forests)(
            y_pred_ind_arr,
            n_estimators,
            all_y_pred,
            y_test,
            seed,
            metric,
            **metric_kwargs,
        )
        for i, seed in zip(range(n_repeats), rng.integers(0, 2**32, n_repeats))
    )

    for idx, (first_half_metric, second_half_metric) in enumerate(out):
        metric_star[idx] = first_half_metric
        metric_star_pi[idx] = second_half_metric

    return metric_star, metric_star_pi


import numba as nb

# @nb.jit(cache=True, parallel=True, nogil=True)
# def nanmean3D_axis0(array):
#     output = np.empty((array.shape[1], array.shape[2]))
#     for i in nb.prange(array.shape[1]):
#         for j in range(array.shape[2]):
#             output[i, j] = np.nanmean(array[:, i, j])
#     return output


@nb.jit(cache=True, parallel=True, nogil=True)
def nanmean2D_axis0(array):
    output = np.empty(array.shape[1])
    for i in nb.prange(array.shape[1]):
        output[i] = np.nanmean(array[:, i])
    return output


def _parallel_build_null_forests(
    index_arr,
    n_estimators: int,
    all_y_pred: ArrayLike,
    y_test: ArrayLike,
    # non_nan_mask: ArrayLike,
    seed: int,
    metric: str,
    **metric_kwargs: dict,
):
    """Randomly sample two sets of forests and compute the metric on each."""
    rng = np.random.default_rng(seed)
    metric_func = METRIC_FUNCTIONS[metric]

    # two sets of random indices from 1 : 2N are sampled using Fisher-Yates
    # first_forest_inds = rng.choice(len(index_arr), size=n_estimators, replace=False)
    # second_forest_inds = np.setdiff1d(index_arr, first_forest_inds)
    rng.shuffle(index_arr)

    # get random half of the posteriors from two sets of trees
    first_forest_inds = index_arr[: n_estimators // 2]
    second_forest_inds = index_arr[n_estimators // 2 :]

    # get random half of the posteriors as one forest
    # print(non_nan_mask.shape)
    first_forest_pred = all_y_pred[first_forest_inds, ...]
    second_forest_pred = all_y_pred[second_forest_inds, ...]

    # determine if there are any nans in the final posterior array, when
    # averaged over the trees
    # first_forest_samples = _non_nan_samples(first_forest_pred)
    # second_forest_samples = _non_nan_samples(second_forest_pred)
    # Find the row indices with NaN values along the specified axis
    # nan_indices = np.isnan(posterior_arr).any(axis=2).all(axis=0)

    # # Invert the boolean mask to get indices without NaN values
    # nonnan_indices = np.where(~nan_indices)[0]

    # todo: is this step necessary?
    # non_nan_samples = np.intersect1d(
    #     first_forest_samples, second_forest_samples, assume_unique=True
    # )
    # now average the posteriors over the trees for the non-nan samples
    # y_pred_first_half = np.nanmean(first_forest_pred[:, non_nan_samples, :], axis=0)
    # y_pred_second_half = np.nanmean(second_forest_pred[:, non_nan_samples, :], axis=0)
    # compute two instances of the metric from the sampled trees
    # first_half_metric = metric_func(y_test[non_nan_samples, :], y_pred_first_half)
    # second_half_metric = metric_func(y_test[non_nan_samples, :], y_pred_second_half)

    # XXX: slice it only on the first output and the second output is by definition 1 - first_output
    # (n_samples, n_outputs) and (n_samples, n_outputs)
    # y_pred_first_half = np.nanmean(first_forest_pred[:, :, 0], axis=0).reshape(-1, 1)
    # y_pred_second_half = np.nanmean(second_forest_pred[:, :, 0], axis=0).reshape(-1, 1)

    y_pred_first_half = nanmean2D_axis0(first_forest_pred[:, :, 0]).reshape(-1, 1)
    y_pred_second_half = nanmean2D_axis0(second_forest_pred[:, :, 0]).reshape(-1, 1)

    # make first axis: 1 - second axis
    y_pred_first_half = np.concatenate(
        (y_pred_first_half, 1 - y_pred_first_half), axis=1
    )
    y_pred_second_half = np.concatenate(
        (y_pred_second_half, 1 - y_pred_second_half), axis=1
    )

    # y_pred_first_half = nanmean3D_axis0(first_forest_pred)
    # y_pred_second_half = nanmean3D_axis0(second_forest_pred)

    # y_pred_first_half = np.mean(first_forest_pred, axis=0)
    # y_pred_second_half = np.mean(second_forest_pred, axis=0)
    # y_pred_first_half = np.sum(first_forest_pred, axis=0) / len(y_test)
    # y_pred_second_half = np.sum(second_forest_pred, axis=0) / len(y_test)

    # print(y_pred_first_half.shape, y_pred_second_half.shape)
    # if np.isnan(y_pred_first_half).any() or np.isnan(y_pred_second_half).any():
    #     raise RuntimeError("NaNs in the first half of the posteriors.")

    # figure out if any sample indices have nans after averaging over trees
    # and just slice them out in both y_test and y_pred.

    # compute two instances of the metric from the sampled trees
    first_half_metric = metric_func(y_test, y_pred_first_half, **metric_kwargs)
    second_half_metric = metric_func(y_test, y_pred_second_half, **metric_kwargs)
    return first_half_metric, second_half_metric
    # return y_pred_first_half, y_pred_second_half
    # return 0, 0


def run_pvalue(
    root_dir,
    model_name,
    perm_model_name,
    sim_name,
    n_samples,
    n_dims_1,
    n_dims_2_,
    idx,
    metric,
    n_permutations,
    overwrite=False,
):
    output_fname = (
        root_dir
        / "output"
        / "coleman_pvalues"
        / sim_name
        / f"{sim_name}_{n_samples}_{n_dims_1}_{n_dims_2_}_{idx}.npz"
    )
    output_fname.parent.mkdir(parents=True, exist_ok=True)
    if output_fname.exists() and not overwrite:
        return

    norm_output_fname = (
        root_dir
        / "output"
        / model_name
        / sim_name
        / f"{sim_name}_{n_samples}_{n_dims_1}_{n_dims_2_}_{idx}.npz"
    )

    # load data
    norm_data = np.load(norm_output_fname)
    y = norm_data["y"]
    y_pred_normal = norm_data["posterior_arr"]

    perm_output_fname = (
        root_dir
        / "output"
        / perm_model_name
        / sim_name
        / f"{sim_name}_{n_samples}_{n_dims_1}_{n_dims_2_}_{idx}.npz"
    )
    perm_data = np.load(perm_output_fname)
    assert_array_equal(y, perm_data["y"])
    y_pred_perm = perm_data["posterior_arr"]

    # pr = cProfile.Profile()
    # pr.enable()
    metric_star, metric_star_pi = _compute_null_distribution_coleman(
        y,
        y_pred_normal,
        y_pred_perm,
        metric=metric,
        n_repeats=n_permutations,
        seed=idx,
        n_jobs=1,
    )
    # pr.disable()
    # stats = Stats(pr)
    # stats.sort_stats("tottime").print_stats(10)

    y_pred_proba_orig = np.nanmean(y_pred_normal, axis=0)
    y_pred_proba_perm = np.nanmean(y_pred_perm, axis=0)
    metric_func = METRIC_FUNCTIONS[metric]
    observe_stat = metric_func(y, y_pred_proba_orig)
    permute_stat = metric_func(y, y_pred_proba_perm)

    # metric^\pi - metric = observed test statistic, which under the
    # null is normally distributed around 0
    observe_test_stat = permute_stat - observe_stat

    # metric^\pi_j - metric_j, which is centered at 0
    null_dist = metric_star_pi - metric_star

    # compute pvalue
    pvalue = (1 + (null_dist <= observe_test_stat).sum()) / (1 + n_permutations)

    np.savez_compressed(
        output_fname,
        idx=idx,
        pvalue=pvalue,
        observe_test_stat=observe_test_stat,
        null_dist=null_dist,
        n_permutations=n_permutations,
        n_samples=n_samples,
        n_dims_1=n_dims_1,
        n_dims_2=n_dims_2_,
        sim_type=sim_name,
    )

    # results = defaultdict(list)
    # results["pvalues"].append(pvalue)
    # results["n_samples"].append(n_samples)
    # results["sim_name"].append(sim_name)
    # results["idx"].append(idx)
    # results["n_dims_1"].append(n_dims_1)
    # return results


if __name__ == "__main__":
    root_dir = Path("/Volumes/Extreme Pro/cancer")
    root_dir = Path("/data/adam/")
    n_repeats = 100
    n_permutations = 1_000
    metric = "mi"
    sims = [
        "multi_equal",
        "multi_modalv2",
        "mean_shiftv4",
    ]
    n_jobs = -2

    # Section: varying over sample-sizes
    model_name = "comight"
    perm_model_name = "comight-perm"
    n_dims_1 = 512 - 6
    n_samples_list = [2**x for x in range(8, 11)]
    n_samples_list = [2**8, 2**10]
    print(n_samples_list)

    n_dims_2_ = 6

    # Section: varying over samples
    results = defaultdict(list)
    Parallel(n_jobs=n_jobs)(
        delayed(run_pvalue)(
            root_dir,
            model_name,
            perm_model_name,
            sim_name,
            n_samples,
            n_dims_1,
            n_dims_2_,
            idx,
            metric,
            n_permutations,
        )
        for sim_name in sims
        for idx in range(n_repeats)
        for n_samples in n_samples_list
    )
    # for res in out:
    #     for key in res.keys():
    #         results[key].extend(res[key])
    # assert all(len(val) == len(results["pvalues"]) for key, val in results.items())

    # TODO: save coleman pvalues
    # df = pd.DataFrame(results)
    # df.to_csv(root_dir / "coleman_nsamples_pvalues.csv")

    # Section: varying over dimensions
    # n_dims_list = [2**i - 6 for i in range(3, 12)]
    # n_samples = 512
    # for sim in sims:
    #     pvalues = []

    #     for idx in range(n_repeats):
    #         for n_dims in n_dims_list:
    #             norm_output_fname = (
    #                 root_dir
    #                 / "output"
    #                 / model_name
    #                 / sim_name
    #                 / f"{sim_name}_{n_samples}_{n_dims_1}_{n_dims_2_}_{idx}.npz"
    #             )

    #             # load data
    #             norm_data = np.load(norm_output_fname)
    #             y = norm_data["y"]
    #             y_pred_normal = norm_data["posterior_arr"]

    #             perm_output_fname = (
    #                 root_dir
    #                 / "output"
    #                 / perm_model_name
    #                 / sim_name
    #                 / f"{sim_name}_{n_samples}_{n_dims_1}_{n_dims_2_}_{idx}.npz"
    #             )
    #             perm_data = np.load(perm_output_fname)
    #             assert_array_equal(y, perm_data["y"])
    #             y_pred_perm = perm_data["posterior_arr"]

    #             metric_star, metric_star_pi = _compute_null_distribution_coleman(
    #                 y,
    #                 y_pred_normal,
    #                 y_pred_perm,
    #                 metric=metric,
    #                 n_repeats=n_permutations,
    #                 seed=idx,
    #                 n_jobs=n_jobs,
    #             )

    #             y_pred_proba_orig = np.nanmean(y_pred_normal, axis=0)
    #             y_pred_proba_perm = np.nanmean(y_pred_perm, axis=0)
    #             metric_func = METRIC_FUNCTIONS[metric]
    #             observe_stat = metric_func(y, y_pred_proba_orig)
    #             permute_stat = metric_func(y, y_pred_proba_perm)

    #             # metric^\pi - metric = observed test statistic, which under the
    #             # null is normally distributed around 0
    #             observe_test_stat = permute_stat - observe_stat

    #             # metric^\pi_j - metric_j, which is centered at 0
    #             null_dist = metric_star_pi - metric_star

    #             # compute pvalue
    #             pvalue = (1 + (null_dist <= observe_test_stat).sum()) / (1 + n_permutations)

    #             pvalues.append(pvalue)
