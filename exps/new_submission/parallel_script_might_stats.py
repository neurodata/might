from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.model_selection import (
    StratifiedShuffleSplit,
)
from sklearn.metrics import roc_curve
from joblib import delayed, Parallel
from sktree import HonestForestClassifier
from sktree.stats import (
    build_hyppo_oob_forest,
)

seed = 12345
rng = np.random.default_rng(seed)

### hard-coded parameters
n_estimators = 500
max_features = 0.3
test_size = 0.2
n_jobs = -1


def sensitivity_at_specificity(y_true, y_score, target_specificity=0.98, pos_label=1):
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

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score_binary, pos_label=pos_label)

    # Find the threshold corresponding to the target specificity
    index = np.argmax(fpr >= (1 - target_specificity))
    threshold_at_specificity = thresholds[index]

    # Compute sensitivity at the chosen specificity
    # sensitivity = tpr[index]
    # return sensitivity

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
):
    n_dims_ = 4096
    n_samples_ = 4096
    overwrite = True

    fname = (
        root_dir / "data" / sim_name / f"{sim_name}_{n_samples_}_{n_dims_}_{idx}.npz"
    )
    if not fname.exists():
        raise RuntimeError(f"{fname} does not exist")

    # print(f"Reading {fname}")
    data = np.load(fname, allow_pickle=True)
    X, y = data["X"], data["y"]

    if n_samples < X.shape[0]:
        _cv = StratifiedShuffleSplit(n_splits=1, train_size=n_samples)
        for train_idx, _ in _cv.split(X, y):
            continue
        X = X[train_idx, :]
        y = y[train_idx, ...].squeeze()
    if n_dims < X.shape[1]:
        X = X[:, :n_dims]

    output_fname = (
        root_dir
        / "output"
        / model_name
        / sim_name
        / f"{sim_name}_{n_samples}_{n_dims}_{idx}.npz"
    )
    output_fname.parent.mkdir(exist_ok=True, parents=True)
    two_split = model_name == "two_split"

    print("Running analysis for: ", output_fname, two_split, overwrite)
    if not output_fname.exists() or overwrite:
        might_kwargs = MODEL_NAMES["might"]
        est = HonestForestClassifier(**might_kwargs)

        est, posterior_arr = build_hyppo_oob_forest(
            est,
            X,
            y,
            two_split=two_split,
            verbose=False,
        )

        # generate S@S98 from posterior array
        sas98 = sensitivity_at_specificity(y, posterior_arr, target_specificity=0.98)

        np.savez_compressed(
            output_fname,
            idx=idx,
            n_samples=n_samples,
            n_dims=n_dims,
            sas98=sas98,
            sim_type=sim_name,
        )


MODEL_NAMES = {
    "might": {
        "n_estimators": n_estimators * 5,
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

    SIMULATIONS = {
        "1": {},
        "2": {"m_factor": 1},
        # "3": {"band_type": "ar", "rho": 0.5},
        # "4": {"band_type": "ar", "m_factor": 1, "rho": 0.5},
        # "5": {"mix": 0.5},
    }
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

    n_repeats = 5
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

    results = Parallel(n_jobs=-1)(
        delayed(_run_simulation)(
            # est,
            n_samples,
            n_dims,
            idx,
            root_dir,
            sim_name,
            model_name,
        )
        for sim_name in SIMULATIONS_NAMES.values()
        for n_samples in n_samples_list
        for idx in range(n_repeats)
        for model_name in ["two_split", "three_split"]
    )
