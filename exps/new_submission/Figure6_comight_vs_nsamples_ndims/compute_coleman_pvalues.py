import numpy as np
from typing import Optional, Tuple
from pathlib import Path
import numpy as np
from numpy.testing import assert_array_equal
from joblib import Parallel, delayed
from numpy.typing import ArrayLike
from scipy.stats import entropy
from sklearn.metrics import (
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    roc_curve,
)


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

    # generate the random seeds for the parallel jobs
    ss = np.random.SeedSequence(seed)
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
        for i, seed in zip(range(n_repeats), ss.spawn(n_repeats))
    )

    for idx, (first_half_metric, second_half_metric) in enumerate(out):
        metric_star[idx] = first_half_metric
        metric_star_pi[idx] = second_half_metric

    return metric_star, metric_star_pi


def _parallel_build_null_forests(
    index_arr: ArrayLike,
    n_estimators: int,
    all_y_pred: ArrayLike,
    y_test: ArrayLike,
    seed: int,
    metric: str,
    **metric_kwargs: dict,
):
    """Randomly sample two sets of forests and compute the metric on each."""
    rng = np.random.default_rng(seed)
    metric_func = METRIC_FUNCTIONS[metric]

    # two sets of random indices from 1 : 2N are sampled using Fisher-Yates
    rng.shuffle(index_arr)

    # get random half of the posteriors from two sets of trees
    first_forest_inds = index_arr[: n_estimators // 2]
    second_forest_inds = index_arr[n_estimators // 2 :]

    # get random half of the posteriors as one forest
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
    # # compute two instances of the metric from the sampled trees
    # first_half_metric = metric_func(y_test[non_nan_samples, :], y_pred_first_half)
    # second_half_metric = metric_func(y_test[non_nan_samples, :], y_pred_second_half)

    # (n_samples, n_outputs) and (n_samples, n_outputs)
    y_pred_first_half = np.nanmean(first_forest_pred[:, :, :], axis=0)
    y_pred_second_half = np.nanmean(second_forest_pred[:, :, :], axis=0)
    if any(np.isnan(y_pred_first_half).any()) or any(
        np.isnan(y_pred_second_half).any()
    ):
        raise RuntimeError("NaNs in the first half of the posteriors.")

    # figure out if any sample indices have nans after averaging over trees
    # and just slice them out in both y_test and y_pred.

    # compute two instances of the metric from the sampled trees
    first_half_metric = metric_func(y_test, y_pred_first_half, **metric_kwargs)
    second_half_metric = metric_func(y_test, y_pred_second_half, **metric_kwargs)
    return first_half_metric, second_half_metric


if __name__ == "__main__":
    root_dir = Path("/")
    n_repeats = 100
    n_permutations = 10_000
    metric = "mi"
    sims = ["multi_equal", "multi_modalv2", "mean_shiftv4"]
    n_jobs = 1

    # Section: varying over sample-sizes
    model_name = "comight"
    perm_model_name = "comight-perm"
    n_dims_1 = 512 - 6
    n_samples_list = [2**x for x in range(8, 11)]

    n_dims_2_ = 6

    # Section: varying over samples
    for sim_name in sims:
        pvalues = []

        for idx in range(n_repeats):
            for n_samples in n_samples_list:
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

                pvalue = _compute_null_distribution_coleman(
                    y,
                    y_pred_normal,
                    y_pred_perm,
                    metric=metric,
                    n_repeats=n_permutations,
                    seed=idx,
                    n_jobs=n_jobs,
                )

                pvalues.append(pvalue)
                print("done")

    # TODO: save coleman pvalues

    # Section: varying over dimensions
    n_dims_list = [2**i - 6 for i in range(3, 12)]
    n_samples = 512
    for sim in sims:
        pvalues = []

        for idx in range(n_repeats):
            for n_dims in n_dims_list:
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

                pvalue = _compute_null_distribution_coleman(
                    y,
                    y_pred_normal,
                    y_pred_perm,
                    metric=metric,
                    n_repeats=n_permutations,
                    seed=idx,
                    n_jobs=n_jobs,
                )

                pvalues.append(pvalue)
