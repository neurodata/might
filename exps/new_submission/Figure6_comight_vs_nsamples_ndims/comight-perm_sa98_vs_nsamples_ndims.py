"""Generating data for CoMIGHT simulations with S@S98."""

# A : Control ~ N(0, 1), Cancer ~ N(1, 1)
# B:  Control ~ N(0, 1), Cancer ~ 0.75*N(1, 1) + 0.25*N(5, 1)
# C:  Control~ 0.75*N(1, 1) + 0.25*N(5, 1), Cancer ~ 0.75*N(1, 1) + 0.25*N(5, 1)
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import roc_curve
from treeple.stats import PermutationHonestForestClassifier, build_oob_forest
from treeple.tree import MultiViewDecisionTreeClassifier

seed = 12345
rng = np.random.default_rng(seed)

### hard-coded parameters
n_estimators = 6000
max_features = 0.3


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
):
    n_samples_ = 4096
    n_dims_2_ = 6
    n_dims_1_ = 4090
    target_specificity = 0.98

    fname = (
        root_dir
        / "data"
        / sim_name
        / f"{sim_name}_{n_samples_}_{n_dims_1_}_{n_dims_2_}_{idx}.npz"
    )
    output_fname = (
        root_dir
        / "output"
        / model_name
        / sim_name
        / f"{sim_name}_{n_samples}_{n_dims_1}_{n_dims_2_}_{idx}.npz"
    )
    output_fname.parent.mkdir(exist_ok=True, parents=True)
    output_fname.parent.mkdir(exist_ok=True, parents=True)
    print(f"Output file: {output_fname} {output_fname.exists()}")
    if not overwrite and output_fname.exists():
        return
    if not fname.exists():
        raise RuntimeError(f"{fname} does not exist")
    print(f"Reading {fname}")
    data = np.load(fname, allow_pickle=True)
    X, y = data["X"], data["y"]
    print(X.shape, y.shape)
    if n_samples < X.shape[0]:
        # _cv = StratifiedShuffleSplit(n_splits=1, train_size=n_samples, random_state=seed)
        # for train_idx, _ in _cv.split(X, y):
        #     continue
        # X = X[train_idx, :]
        # y = y[train_idx, ...].squeeze()
        class_0_idx = np.arange(4096 // 2)
        class_1_idx = np.arange(4096 // 2, 4096)

        # vstack first class and second class?
        X = np.vstack(
            (X[class_0_idx[: n_samples // 2], :], X[class_1_idx[: n_samples // 2], :])
        )
        y = np.concatenate(
            (y[class_0_idx[: n_samples // 2]], y[class_1_idx[: n_samples // 2]])
        )
        assert np.sum(y) == n_samples // 2, f"{np.sum(y)}, {n_samples // 2}"
    if n_dims_1 < n_dims_1_:
        view_one = X[:, :n_dims_1]
        view_two = X[:, n_dims_1_:]
        assert view_two.shape[1] == n_dims_2_
        X = np.concatenate((view_one, view_two), axis=1)

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
        feature_set_ends = [
            n_dims_1,
            n_dims_1 + n_dims_2_,
        ]  # [4090, 4096] for varying samples
        assert X.shape[1] == feature_set_ends[1]
        est = PermutationHonestForestClassifier(
            seed=seed, feature_set_ends=feature_set_ends, **might_kwargs
        )
        covariate_index = np.arange(n_dims_1)  # permute the first view (i.e. X)

        est, posterior_arr = build_oob_forest(
            est,
            X,
            y,
            covariate_index=covariate_index,
            verbose=False,
        )

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
        "max_features": max_features,
        "max_samples": 1.6,
        "tree_estimator": MultiViewDecisionTreeClassifier(),
    },
}

if __name__ == "__main__":
    # idx = int(sys.argv[1])
    # n_samples = int(sys.argv[2])
    # n_dims_1 = int(sys.argv[3])
    # sim_name = sys.argv[4]
    # root_dir = sys.argv[5]

    # model_name = "comight-perm"
    # _run_simulation(
    #     n_samples,
    #     n_dims_1,
    #     idx,
    #     Path(root_dir),
    #     sim_name,
    #     model_name,
    #     overwrite=False,
    # )

    root_dir = Path("/Volumes/Extreme Pro/cancer")
    root_dir = Path("/data/adam/")

    SIMULATIONS_NAMES = [
        "mean_shiftv4",
        "multi_modalv2",
        "multi_equal",
    ]
    model_name = "comight-perm"
    overwrite = False

    n_start = 0
    n_repeats = 100
    n_jobs = -2

    # Section: varying over sample-sizes
    n_samples_list = [2**x for x in range(8, 11)]
    n_dims_1 = 512 - 6
    print(n_samples_list)
    results = Parallel(n_jobs=n_jobs)(
        delayed(_run_simulation)(
            n_samples,
            n_dims_1,
            idx,
            root_dir,
            sim_name,
            model_name,
            overwrite=False,
        )
        for sim_name in SIMULATIONS_NAMES
        for n_samples in n_samples_list
        for idx in range(n_start, n_repeats)
    )

    # Section: varying over dimensions
    n_samples = 512
    n_dims_list = [2**i - 6 for i in range(3, 12)]
    print(n_dims_list)
    results = Parallel(n_jobs=n_jobs, verbose=True)(
        delayed(_run_simulation)(
            n_samples,
            n_dims_1,
            idx,
            root_dir,
            sim_name,
            model_name,
            overwrite=False,
        )
        for sim_name in SIMULATIONS_NAMES
        for n_dims_1 in n_dims_list
        for idx in range(n_start, n_repeats)
    )
