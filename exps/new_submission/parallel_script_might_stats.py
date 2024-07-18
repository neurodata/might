from collections import defaultdict
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedShuffleSplit
from sktree import HonestForestClassifier
from sktree.datasets import make_trunk_classification
from sktree.stats import build_hyppo_oob_forest

seed = 12345
rng = np.random.default_rng(seed)

### hard-coded parameters
n_estimators = 6000
max_features = 0.3
test_size = 0.2
n_jobs = -1

SIMULATIONS = {
    "trunk": {"mu_0": 1, "mu_1": -1},
    "trunk-overlap": {"mu_0": 1, "mu_1": 1},
    # "3": {"band_type": "ar", "rho": 0.5},
    # "4": {"band_type": "ar", "m_factor": 1, "rho": 0.5},
    # "5": {"mix": 0.5},
}


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
    # est,
    n_samples,
    n_dims,
    idx,
    root_dir,
    sim_name,
    model_name,
    overwrite=False,
    use_second_split_for_threshold=False,
):
    n_dims_ = 4096
    n_samples_ = 4096
    overwrite = True
    target_specificity = 0.98

    # fname = (
    #     root_dir / "data" / sim_name / f"{sim_name}_{n_samples_}_{n_dims_}_{idx}.npz"
    # )
    # if not fname.exists():
    #     raise RuntimeError(f"{fname} does not exist")
    # print(f"Reading {fname}")
    # data = np.load(fname, allow_pickle=True)
    # X, y = data["X"], data["y"]
    # if n_samples < X.shape[0]:
    #     _cv = StratifiedShuffleSplit(n_splits=1, train_size=n_samples)
    #     for train_idx, _ in _cv.split(X, y):
    #         continue
    #     X = X[train_idx, :]
    #     y = y[train_idx, ...].squeeze()
    # if n_dims < X.shape[1]:
    #     X = X[:, :n_dims]

    sim_params = SIMULATIONS[sim_name]
    X, y = make_trunk_classification(
        n_samples=n_samples, n_dim=n_dims, n_informative=100, seed=seed, **sim_params
    )

    output_fname = (
        root_dir
        / "output"
        / model_name
        / sim_name
        / f"{sim_name}_{n_samples}_{n_dims}_{idx}.npz"
    )
    output_fname.parent.mkdir(exist_ok=True, parents=True)

    print("Running analysis for: ", output_fname, overwrite)
    if not output_fname.exists() or overwrite:
        might_kwargs = MODEL_NAMES["might"]
        est = HonestForestClassifier(**might_kwargs)

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
            n_dims=n_dims,
            sas98=sas98,
            sim_type=sim_name,
            y=y,
            posterior_arr=posterior_arr,
            threshold=threshold_at_specificity,
        )


MODEL_NAMES = {
    "might": {
        "n_estimators": n_estimators,
        # "random_state": seed,
        "honest_fraction": 0.5,
        "n_jobs": 1,
        "bootstrap": True,
        "stratify": True,
        "max_samples": 1.6,
    },
}

if __name__ == "__main__":
    oob = True
    overwrite = False

    root_dir = Path("/Volumes/Extreme Pro/cancer")
    root_dir = Path("./output/")

    SIMULATIONS_NAMES = {
        "1": "trunk",
        "2": "trunk-overlap",
        # "3": "trunk-banded",
        # "4": "trunk-banded-overlap",
        # "5": "trunk-mix",
    }
    n_dims_list = [2**i for i in range(12)]
    n_dims_list[0] = 1
    n_dims = 2048
    n_samples = 524
    use_second_split_for_threshold = False
    model_name = "might-threshold-third-split"

    n_repeats = 2
    n_samples_list = [2**x for x in range(8, 13)]
    n_samples_ = 4096
    n_dims_ = 4096

    print(n_samples_list)
    print(n_dims_list)
    print(n_samples_)

    # Parallelize the simulations using joblib
    # train MIGHT
    # might_kwargs = MODEL_NAMES["might"]
    # est = HonestForestClassifier(**might_kwargs)

    results = Parallel(n_jobs=-2)(
        delayed(_run_simulation)(
            # est,
            n_samples,
            n_dims,
            idx,
            root_dir,
            sim_name,
            model_name,
            overwrite=False,
            use_second_split_for_threshold=use_second_split_for_threshold,
        )
        for sim_name in SIMULATIONS_NAMES.values()
        for n_samples in n_samples_list
        for idx in range(n_repeats)
    )
