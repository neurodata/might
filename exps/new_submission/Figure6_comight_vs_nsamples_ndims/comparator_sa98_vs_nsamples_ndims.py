"""Generating data for CoMIGHT simulations with S@S98."""

# A : Control ~ N(0, 1), Cancer ~ N(1, 1)
# B:  Control ~ N(0, 1), Cancer ~ 0.75*N(1, 1) + 0.25*N(5, 1)
# C:  Control~ 0.75*N(1, 1) + 0.25*N(5, 1), Cancer ~ 0.75*N(1, 1) + 0.25*N(5, 1)
from pathlib import Path
import numpy as np
from pathlib import Path
import numpy as np
from sklearn.model_selection import (
    StratifiedShuffleSplit,
    StratifiedKFold,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve
from joblib import delayed, Parallel


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


def Calculate_SA98(y_true, y_pred_proba, max_fpr=0.02) -> float:
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
    print(f"Output file: {output_fname} {output_fname.exists()}")
    if not overwrite and output_fname.exists():
        return

    if not fname.exists():
        raise RuntimeError(f"{fname} does not exist")
    print(f"Reading {fname}")
    try:
        data = np.load(fname, allow_pickle=True)
        X, y = data["X"], data["y"]
    except Exception as e:
        print(e, "Error with: ", fname)
        return

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

    print(
        "Running analysis for: ",
        output_fname,
        overwrite,
        X.shape,
        n_samples,
        n_dims_1,
        n_dims_2_,
        n_dims_1 + n_dims_2_,
    )
    if not output_fname.exists() or overwrite:
        feature_set_ends = [
            n_dims_1,
            n_dims_1 + n_dims_2_,
        ]  # [4090, 4096] for varying samples
        assert X.shape[1] == feature_set_ends[1]

        if model_name == "rf":
            model = RandomForestClassifier(random_state=seed, **MODEL_NAMES[model_name])
        elif model_name == "knn":
            model = KNeighborsClassifier(
                n_neighbors=int(np.sqrt(n_samples)+1),
            )
        elif model_name == "svm":
            model = SVC(random_state=seed, **MODEL_NAMES[model_name])
        elif model_name == "lr":
            model = LogisticRegression(random_state=seed, **MODEL_NAMES[model_name])

        # train model in...
        cv = StratifiedKFold(n_splits=5, shuffle=True)
        stats = []
        y_pred_probas = []
        y_test_list = []
        for train_ix, test_ix in cv.split(X, y):
            X_train, X_test = X[train_ix, :], X[test_ix, :]
            y_train, y_test = y[train_ix], y[test_ix]
            model.fit(X_train, y_train)
            observe_proba = model.predict_proba(X_test)
            # calculate S@98 or whatever the stat is
            y_pred_probas.append(observe_proba)
            y_test_list.append(y_test)
            stat = Calculate_SA98(y_test, observe_proba, max_fpr=0.02)
            stats.append(stat)
        
        # average the stats
        stat_avg = np.mean(stats)

        np.savez_compressed(
            output_fname,
            idx=idx,
            n_samples=n_samples,
            n_dims_1=n_dims_1,
            n_dims_2=n_dims_2_,
            sas98=stat_avg,
            sa98_list=stats,
            sim_type=sim_name,
            y_test_list=y_test_list,
            y_pred_probas=y_pred_probas,
        )


MODEL_NAMES = {
    "rf": {
        "n_estimators": 1200,
        "max_features": max_features,
    },
    # "knn": {
    #     "n_neighbors": 5,
    # },
    "svm": {
        "probability": True,
    },
    "lr": {
        "max_iter": 1000,
        "penalty": "l1",
        "solver": "liblinear",
    },
}

if __name__ == "__main__":
    root_dir = Path("/Volumes/Extreme Pro/cancer")
    root_dir = Path("/data/adam/")

    SIMULATIONS_NAMES = [
        "mean_shift_compounding",
        "multi_modal_compounding",
        "multi_equal",
    ]

    model_names = [
        # "rf",
        "knn",
        # "svm",
        # "lr",
    ]
    overwrite = False

    n_repeats = 100
    n_jobs = -1

    # Section: varying over sample-sizes
    n_samples_list = [2**x for x in range(8, 13)]
    n_dims_1 = 4090
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
        for idx in range(n_repeats)
        for model_name in model_names
    )

    # Section: varying over dimensions of the first view
    n_samples = 4096
    n_dims_list = [2**i - 6 for i in range(3, 13)]
    print(n_dims_list)
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
        for n_dims_1 in n_dims_list
        for idx in range(n_repeats)
        for model_name in model_names
    )
