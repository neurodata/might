"""Generating data for CoMIGHT simulations with S@S98."""

# A : Control ~ N(0, 1), Cancer ~ N(1, 1)
# B:  Control ~ N(0, 1), Cancer ~ 0.75*N(1, 1) + 0.25*N(5, 1)
# C:  Control~ 0.75*N(1, 1) + 0.25*N(5, 1), Cancer ~ 0.75*N(1, 1) + 0.25*N(5, 1)
import sys
from pathlib import Path
import numpy as np
from sktree.datasets import make_trunk_classification, make_trunk_mixture_classification
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.model_selection import (
    StratifiedShuffleSplit,
)
from sklearn.metrics import roc_curve
from joblib import delayed, Parallel
from sktree import HonestForestClassifier
from sktree.tree import MultiViewDecisionTreeClassifier
from sktree.stats import (
    build_hyppo_oob_forest,
)
from sktree.datasets import make_trunk_classification


import numpy as np
from sklearn.datasets import make_blobs


def _generate_2d_rotation(theta=0):
    R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])

    return R


def generate_gaussian_parity(
    n_samples,
    centers=None,
    class_label=None,
    cluster_std=0.25,
    angle_params=None,
    random_state=None,
):
    """
    Generate 2-dimensional Gaussian XOR distribution.
    (Classic XOR problem but each point is the
    center of a Gaussian blob distribution)

    Parameters
    ----------
    n_samples : int
        Total number of points divided among the four
        clusters with equal probability.

    centers : array of shape [n_centers,2], optional (default=None)
        The coordinates of the ceneter of total n_centers blobs.

    class_label : array of shape [n_centers], optional (default=None)
        class label for each blob.

    cluster_std : float, optional (default=1)
        The standard deviation of the blobs.

    angle_params: float, optional (default=None)
        Number of radians to rotate the distribution by.

    random_state : int, RandomState instance, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.


    Returns
    -------
    X : array of shape [n_samples, 2]
        The generated samples.
    y : array of shape [n_samples]
        The integer labels for cluster membership of each sample.
    """

    if random_state != None:
        np.random.seed(random_state)

    if centers == None:
        centers = np.array([(-0.5, 0.5), (0.5, 0.5), (-0.5, -0.5), (0.5, -0.5)])

    if class_label == None:
        class_label = [0, 1, 1, 0]

    blob_num = len(class_label)

    # get the number of samples in each blob with equal probability
    samples_per_blob = np.random.multinomial(
        n_samples, 1 / blob_num * np.ones(blob_num)
    )

    X, y = make_blobs(
        n_samples=samples_per_blob,
        n_features=2,
        centers=centers,
        cluster_std=cluster_std,
    )

    for blob in range(blob_num):
        y[np.where(y == blob)] = class_label[blob]

    if angle_params != None:
        R = _generate_2d_rotation(angle_params)
        X = X @ R

    return X, y


def make_gaussian_parity(
    root_dir,
    n_samples=4096,
    n_dim_1=4090,
    mu_0=0,
    mu_1=1,
    seed=None,
    n_dim_2=6,
    return_params=False,
    overwrite=False,
):
    output_fname = (
        root_dir
        / "data"
        / "multi-modal-xor"
        / f"multi-modal-xor_{n_samples}_{n_dim_1}_{n_dim_2}_{seed}.npz"
    )
    output_fname.parent.mkdir(exist_ok=True, parents=True)
    if not overwrite and output_fname.exists():
        return
    rng = np.random.default_rng(seed)

    low_dimX, y = generate_gaussian_parity(
        n_samples,
        class_label=None,
        cluster_std=1.0,
        angle_params=None,
        random_state=seed,
    )

    assert low_dimX.shape[1] == 2
    view_1 = low_dimX[:, (0,)]
    view_1 = np.hstack(
        (view_1, rng.normal(loc=0, scale=1, size=(low_dimX.shape[0], n_dim_1 - 1)))
    )
    view_2 = low_dimX[:, 1:]

    # add noise to the second view so that view_2 = (n_samples, n_dim_2)
    view_2 = np.concatenate(
        (view_2, rng.standard_normal(size=(n_samples, n_dim_2 - view_2.shape[1]))),
        axis=1,
    )

    X = np.concatenate((view_1, view_2), axis=1)
    print(output_fname, X.shape, y.shape)
    np.savez_compressed(output_fname, X=X, y=y)


def make_mean_shift(
    root_dir,
    n_samples=4096,
    n_dim_1=4090,
    mu_0=0,
    mu_1=1,
    seed=None,
    n_dim_2=6,
    return_params=False,
    overwrite=False,
):
    """Make mean shifted binary classification data.

    X comprises of [view_1, view_2] where view_1 is the first ``n_dim_1`` dimensions
    and view_2 is the last ``n_dim_2`` dimensions.

    view_1 is generated, such that [A, B] corresponding to class labels [0, 1]
    are generated as follows:

    A ~ N(1, I)
    B ~ N(m_factor, I)

    view_2 is generated, such that [A, B] corresponding to class labels [0, 1]
    are generated as follows:

    A ~ N(1 / np.sqrt(2), I)
    B ~ N(1 / np.sqrt(2) * m_factor, I)

    Parameters
    ----------
    n_samples : int, optional
        The number of samples to generate, by default 1024.
    n_dim_1 : int, optional
        The number of dimensions in first view, by default 4090.
    mu_0 : int, optional
        The mean of the first class, by default -1.
    mu_1 : int, optional
        The mean of the second class, by default 1.
    seed : int, optional
        Random seed, by default None.
    n_dim_2 : int, optional
        The number of dimensions in second view, by default 6.
    return_params : bool
        Whether to return parameters of the generating model or not. Default is False.

    Returns
    -------
    X : ArrayLike of shape (n_samples, n_dim_1 + n_dim_2)
        Data.
    y : ArrayLike of shape (n_samples,)
        Labels.
    """
    output_fname = (
        root_dir
        / "data"
        / "mean_shift_compounding"
        / f"mean_shift_compounding_{n_samples}_{n_dim_1}_{n_dim_2}_{seed}.npz"
    )
    output_fname.parent.mkdir(exist_ok=True, parents=True)
    if not overwrite and output_fname.exists():
        return
    rng = np.random.default_rng(seed)
    default_n_informative = 1

    # X, y, means, cov = make_trunk_classification(
    #     n_samples=n_samples // 2,
    #     n_dim=n_dim_1 + 1,
    #     n_informative=default_n_informative,
    #     mu_0=mu_0,
    #     mu_1=mu_1,
    #     return_params=True,
    #     rho=0.5,
    #     seed=seed,
    # )

    method = "svd"
    mu_1_vec = np.array([-0.5, 0.5])
    mu_0_vec = np.array([0 / np.sqrt(i) for i in range(1, 3)])
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])

    X = np.vstack(
        (
            rng.multivariate_normal(mu_1_vec, cov, n_samples // 2, method=method),
            rng.multivariate_normal(mu_0_vec, cov, n_samples // 2, method=method),
        )
    )
    assert X.shape[1] == 2
    view_1 = X[:, (0,)]
    view_1 = np.hstack(
        (view_1, rng.normal(loc=0, scale=1, size=(X.shape[0], n_dim_1 - 1)))
    )
    view_2 = X[:, 1:]

    # # get the second informative dimension
    # view_1 = X[:, 1:]

    # # only take one informative dimension
    # view_2 = X[:, (0,)]

    # add noise to the second view so that view_2 = (n_samples, n_dim_2)
    view_2 = np.concatenate(
        (view_2, rng.standard_normal(size=(n_samples, n_dim_2 - view_2.shape[1]))),
        axis=1,
    )

    X = np.concatenate((view_1, view_2), axis=1)
    y = np.concatenate((np.zeros(n_samples // 2), np.ones(n_samples // 2)))

    np.savez_compressed(output_fname, X=X, y=y)


def make_multi_modal(
    root_dir,
    n_samples=4096,
    n_dim_1=4090,
    mu_0=0,
    mu_1=5,
    mix=0.75,
    seed=None,
    n_dim_2=6,
    return_params=False,
    overwrite=False,
):
    """Make multi-modal binary classification data.

    X comprises of [view_1, view_2] where view_1 is the first ``n_dim_1`` dimensions
    and view_2 is the last ``n_dim_2`` dimensions.

    view_1 is generated, such that [A, B] corresponding to class labels [0, 1]
    are generated as follows:

    A ~ N(0, I)
    B ~ mix * N(1, I) + (1 - mix) * N(m_factor, I)

    view_2 is generated, such that [A, B] corresponding to class labels [0, 1]
    are generated as follows:

    A ~ N(0, I)
    B ~ mix * N(1 / np.sqrt(2), I) + (1 - mix) * N(1 / np.sqrt(2) * m_factor, I)

    Parameters
    ----------
    n_samples : int, optional
        The number of samples to generate, by default 1024.
    n_dim_1 : int, optional
        The number of dimensions in first view, by default 4090.
    mu_0 : int, optional
        The mean of the first class, by default 1.
    mu_1 : int, optional
        The mean of the second class, by default -1.
    seed : int, optional
        Random seed, by default None.
    n_dim_2 : int, optional
        The number of dimensions in second view, by default 6.
    return_params : bool
        Whether to return parameters of the generating model or not. Default is False.

    Returns
    -------
    X : ArrayLike of shape (n_samples, n_dim_1 + n_dim_2)
        Data.
    y : ArrayLike of shape (n_samples,)
        Labels.
    """
    output_fname = (
        root_dir
        / "data"
        / "multi_modal_compounding"
        / f"multi_modal_compounding_{n_samples}_{n_dim_1}_{n_dim_2}_{seed}.npz"
    )
    output_fname.parent.mkdir(exist_ok=True, parents=True)
    if not overwrite and output_fname.exists():
        return

    rng = np.random.default_rng(seed)
    default_n_informative = 2

    # X, y, means, covs, X_mixture = make_trunk_mixture_classification(
    #     n_samples=n_samples,
    #     n_dim=n_dim_1 + 1,
    #     n_informative=default_n_informative,
    #     mu_0=mu_0,
    #     mu_1=mu_1,
    #     mix=mix,
    #     scaling_factor=1,
    #     seed=seed,
    #     rho=0.5,
    #     return_params=True,
    # )
    # get the second informative dimension
    # view_1 = X[:, 1:]

    # # only take one informative dimension
    # view_2 = X[:, (0,)]

    method = "svd"
    mu_1_vec = np.array([-1, 2])
    mu_0_vec = np.array([0 / np.sqrt(i) for i in range(1, 3)])
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])

    mixture_idx = rng.choice(2, n_samples // 2, replace=True, shuffle=True, p=[mix, 1 - mix])  # type: ignore

    norm_params = [[mu_0_vec, cov], [mu_1_vec, cov]]
    X_mixture = np.fromiter(
        (
            rng.multivariate_normal(*(norm_params[i]), size=1, method=method)
            for i in mixture_idx
        ),
        dtype=np.dtype((float, 2)),
    )

    X = np.vstack(
        (
            rng.multivariate_normal(np.zeros(2), cov, n_samples // 2, method=method),
            X_mixture.reshape(n_samples // 2, 2),
        )
    )

    assert X.shape[1] == 2
    view_1 = X[:, (0,)]
    view_1 = np.hstack(
        (view_1, rng.normal(loc=0, scale=1, size=(X.shape[0], n_dim_1 - 1)))
    )
    view_2 = X[:, 1:]

    # add noise to the second view so that view_2 = (n_samples, n_dim_2)
    view_2 = np.concatenate(
        (view_2, rng.standard_normal(size=(n_samples, n_dim_2 - view_2.shape[1]))),
        axis=1,
    )
    X = np.concatenate((view_1, view_2), axis=1)
    y = np.concatenate((np.zeros(n_samples // 2), np.ones(n_samples // 2)))
    np.savez_compressed(output_fname, X=X, y=y)


def make_multi_equal(
    root_dir,
    n_samples=4096,
    n_dim_1=4090,
    mu_0=1,
    mu_1=5,
    mix=0.75,
    seed=None,
    n_dim_2=6,
    return_params=False,
):
    """Make multi-modal binary classification data.

    X comprises of [view_1, view_2] where view_1 is the first ``n_dim_1`` dimensions
    and view_2 is the last ``n_dim_2`` dimensions.

    view_1 is generated, such that [A, B] corresponding to class labels [0, 1]
    are generated as follows:

    A ~ mix * N(1, I) + (1 - mix) * N(m_factor, I)
    B ~ mix * N(1, I) + (1 - mix) * N(m_factor, I)

    view_2 is generated, such that [A, B] corresponding to class labels [0, 1]
    are generated as follows:

    A ~ mix * N(1 / np.sqrt(2), I) + (1 - mix) * N(1 / np.sqrt(2) * m_factor, I)
    B ~ mix * N(1 / np.sqrt(2), I) + (1 - mix) * N(1 / np.sqrt(2) * m_factor, I)

    Parameters
    ----------
    n_samples : int, optional
        The number of samples to generate, by default 1024.
    n_dim_1 : int, optional
        The number of dimensions in first view, by default 4090.
    mu_0 : int, optional
        The mean of the first class, by default 1.
    mu_1 : int, optional
        The mean of the second class, by default -1.
    seed : int, optional
        Random seed, by default None.
    n_dim_2 : int, optional
        The number of dimensions in second view, by default 6.
    return_params : bool
        Whether to return parameters of the generating model or not. Default is False.

    Returns
    -------
    X : ArrayLike of shape (n_samples, n_dim_1 + n_dim_2)
        Data.
    y : ArrayLike of shape (n_samples,)
        Labels.
    """
    output_fname = (
        root_dir
        / "data"
        / "multi_equal"
        / f"multi_equal_{n_samples}_{n_dim_1}_{n_dim_2}_{seed}.npz"
    )
    output_fname.parent.mkdir(exist_ok=True, parents=True)

    rng = np.random.default_rng(seed)
    default_n_informative = 2

    X1, _ = make_trunk_mixture_classification(
        n_samples=n_samples,
        n_dim=n_dim_1 + 1,
        n_informative=default_n_informative,
        mu_0=mu_0,
        mu_1=mu_1,
        mix=mix,
        rho=0.5,
        seed=rng.integers(0, np.iinfo(np.int32).max),
        return_params=False,
    )
    # only keep the second half of samples, corresponding to the mixture
    X1 = X1[n_samples // 2 :, :]

    X2, _ = make_trunk_mixture_classification(
        n_samples=n_samples,
        n_dim=n_dim_1 + 1,
        n_informative=default_n_informative,
        mu_0=mu_0,
        mu_1=mu_1,
        mix=mix,
        rho=0.5,
        seed=rng.integers(0, np.iinfo(np.int32).max),
        return_params=False,
    )
    # only keep the second half of samples, corresponding to the mixture
    X2 = X2[n_samples // 2 :, :]

    X = np.vstack((X1, X2))
    y = np.hstack((np.zeros(n_samples // 2), np.ones(n_samples // 2)))

    # get the second informative dimension
    view_1 = X[:, 1:]

    # only take one informative dimension
    view_2 = X[:, (0,)]

    # add noise to the second view so that view_2 = (n_samples, n_dim_2)
    view_2 = np.concatenate(
        (view_2, rng.standard_normal(size=(n_samples, n_dim_2 - view_2.shape[1]))),
        axis=1,
    )
    X = np.concatenate((view_1, view_2), axis=1)
    np.savez_compressed(output_fname, X=X, y=y)


seed = 12345
rng = np.random.default_rng(seed)

### hard-coded parameters
n_estimators = 6000
max_features = 0.3
test_size = 0.2
n_jobs = -1


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


def _run_simulation(
    n_samples,
    n_dims_1,
    idx,
    root_dir,
    sim_name,
    model_name,
    overwrite=False,
    use_second_split_for_threshold=False,
):
    n_samples_ = 4096
    n_dims_2_ = 6
    n_dims_1_ = 4090
    overwrite = True
    target_specificity = 0.98

    fname = (
        root_dir
        / "data"
        / sim_name
        / f"{sim_name}_{n_samples_}_{n_dims_1_}_{n_dims_2_}_{idx}.npz"
    )
    if not fname.exists():
        raise RuntimeError(f"{fname} does not exist")
    print(f"Reading {fname}")
    data = np.load(fname, allow_pickle=True)
    X, y = data["X"], data["y"]
    print(X.shape, y.shape)
    if n_samples < X.shape[0]:
        _cv = StratifiedShuffleSplit(n_splits=1, train_size=n_samples)
        for train_idx, _ in _cv.split(X, y):
            continue
        X = X[train_idx, :]
        y = y[train_idx, ...].squeeze()
    if n_dims_1 < n_dims_1_:
        view_one = X[:, :n_dims_1]
        view_two = X[:, n_dims_1_:]
        assert view_two.shape[1] == n_dims_2_
        X = np.concatenate((view_one, view_two), axis=1)

    output_fname = (
        root_dir
        / "output"
        / model_name
        / sim_name
        / f"{sim_name}_{n_samples}_{n_dims_1}_{n_dims_2_}_{idx}.npz"
    )
    output_fname.parent.mkdir(exist_ok=True, parents=True)

    print(
        "Running analysis for: ",
        output_fname,
        overwrite,
        X.shape,
        n_samples,
        n_dims_1 + n_dims_2_,
    )
    if not output_fname.exists() or overwrite:
        might_kwargs = MODEL_NAMES["might"]
        feature_set_ends = [n_dims_1, n_dims_1 + n_dims_2_]  # [4090, 4096]
        assert X.shape[1] == feature_set_ends[1]

        est = HonestForestClassifier(feature_set_ends=feature_set_ends, **might_kwargs)

        est, posterior_arr = build_hyppo_oob_forest(
            est,
            X,
            y,
            verbose=False,
        )

        if use_second_split_for_threshold:
            # array-like of shape (n_estimators, n_samples, n_classes)
            honest_idx_posteriors = est.predict_proba_per_tree(
                X, indices=est.honest_indices_
            )

            # get the threshold for specified highest sensitivity at 0.98 specificity
            # Compute nan-averaged y_score along the trees axis
            y_score_avg = np.nanmean(honest_idx_posteriors, axis=0)

            # Extract true labels and nan-averaged predicted scores for the positive class
            y_true = y.ravel()
            y_score_binary = y_score_avg[:, 1]

            # Identify rows with NaN values in y_score_binary
            nan_rows = np.isnan(y_score_binary)

            # Remove NaN rows from y_score_binary and y_true
            y_score_binary = y_score_binary[~nan_rows]
            y_true = y_true[~nan_rows]
        else:
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

        np.savez_compressed(
            output_fname,
            idx=idx,
            n_samples=n_samples,
            n_dims_1=n_dims_1,
            n_dims_2=n_dims_2_,
            sas98=sas98,
            sim_type=sim_name,
            y=y,
            posterior_arr=posterior_arr,
            threshold=threshold_at_specificity,
        )


MODEL_NAMES = {
    "might": {
        "n_estimators": n_estimators,
        "honest_fraction": 0.5,
        "n_jobs": 1,
        "bootstrap": True,
        "stratify": True,
        "max_samples": 1.6,
        "tree_estimator": MultiViewDecisionTreeClassifier(),
    },
}

if __name__ == "__main__":
    root_dir = sys.argv[1]

    SIMULATIONS_NAMES = ["mean_shift", "multi_modal", "multi_equal"]
    model_name = "comight"
    overwrite = False

    n_repeats = 100

    # Section: Make data
    # root_dir = Path("/Volumes/Extreme Pro/cancer")
    # root_dir = Path("/data/adam/")

    n_repeats = 100
    Parallel(n_jobs=-1)(
        delayed(func)(
            Path(root_dir),
            seed=seed,
        )
        for seed in range(n_repeats)
        for func in [
            make_mean_shift,
            make_multi_modal,
            make_multi_equal,
        ]
    )

    # for seed in range(n_repeats):
    #     make_mean_shift(root_dir, seed=seed)
    #     make_multi_modal(root_dir, seed=seed)
    # make_multi_equal(root_dir, seed=seed)

    # Section: varying over sample-sizes
    # n_samples_list = [2**x for x in range(8, 13)]
    # n_dims_1 = 4090
    # print(n_samples_list)
    # results = Parallel(n_jobs=-2)(
    #     delayed(_run_simulation)(
    #         n_samples,
    #         n_dims_1,
    #         idx,
    #         root_dir,
    #         sim_name,
    #         model_name,
    #         overwrite=False,
    #     )
    #     for sim_name in SIMULATIONS_NAMES
    #     for n_samples in n_samples_list
    #     for idx in range(n_repeats)
    # )

    # Section: varying over dimensions
    # n_samples = 4096
    # n_dims_list = [2**i - 6 for i in range(3, 13)]
    # print(n_dims_list)
    # results = Parallel(n_jobs=-2)(
    #     delayed(_run_simulation)(
    #         n_samples,
    #         n_dims_1,
    #         idx,
    #         root_dir,
    #         sim_name,
    #         model_name,
    #         overwrite=False,
    #     )
    #     for sim_name in SIMULATIONS_NAMES
    #     for n_dims_1 in n_dims_list
    #     for idx in range(n_repeats)
    # )
