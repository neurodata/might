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


def _run_simulation_oneview(
    n_samples,
    n_dims_1,
    idx,
    root_dir,
    sim_name,
    model_name,
    run_view,
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
    if run_view == "view_one":
        X = X[:, :n_dims_1]
        assert X.shape[1] == n_dims_1
    elif run_view == "view_two":
        X = X[:, -n_dims_2_:]
        assert X.shape[1] == n_dims_2_

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
        if model_name == "rf":
            model = RandomForestClassifier(random_state=seed, **MODEL_NAMES[model_name])
        elif model_name == "knn":
            model = KNeighborsClassifier(
                n_neighbors=int(np.sqrt(n_samples) + 1),
            )
        elif model_name == "svm":
            model = SVC(random_state=seed, **MODEL_NAMES[model_name])
        elif model_name == "lr":
            model = LogisticRegression(random_state=seed, **MODEL_NAMES[model_name])

        # train model in...
        cv = StratifiedKFold(n_splits=5, shuffle=True)
        stats = []
        y_pred_probas = np.full((5, n_samples), np.nan, dtype=np.float32)
        y_test_list = np.full((5, n_samples), np.nan, dtype=np.float32)

        for idx, (train_ix, test_ix) in enumerate(cv.split(X, y)):
            X_train, X_test = X[train_ix, :], X[test_ix, :]
            y_train, y_test = y[train_ix], y[test_ix]
            model.fit(X_train, y_train)
            observe_proba = model.predict_proba(X_test)
            # calculate S@98 or whatever the stat is
            y_pred_probas[idx, test_ix] = observe_proba
            y_test_list[idx, test_ix] = y_test
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
            sim_type=sim_name,
            # sa98_list=stats,
            y_test_list=y_test_list,
            y_pred_probas=y_pred_probas,
        )


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
        if model_name == "rf":
            model = RandomForestClassifier(random_state=seed, **MODEL_NAMES[model_name])
        elif model_name == "knn":
            model = KNeighborsClassifier(
                n_neighbors=int(np.sqrt(n_samples) + 1),
            )
        elif model_name == "svm":
            model = SVC(random_state=seed, **MODEL_NAMES[model_name])
        elif model_name == "lr":
            model = LogisticRegression(random_state=seed, **MODEL_NAMES[model_name])

        # train model in...
        cv = StratifiedKFold(n_splits=5, shuffle=True)
        stats = []
        y_pred_probas = np.full((5, n_samples), np.nan, dtype=np.float32)
        y_test_list = np.full((5, n_samples), np.nan, dtype=np.float32)

        for idx, (train_ix, test_ix) in enumerate(cv.split(X, y)):
            X_train, X_test = X[train_ix, :], X[test_ix, :]
            y_train, y_test = y[train_ix], y[test_ix]
            model.fit(X_train, y_train)
            observe_proba = model.predict_proba(X_test)
            # calculate S@98 or whatever the stat is
            y_pred_probas[idx, test_ix] = observe_proba
            y_test_list[idx, test_ix] = y_test
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
            sim_type=sim_name,
            # sa98_list=stats,
            y_test_list=y_test_list,
            y_pred_probas=y_pred_probas,
        )


MODEL_NAMES = {
    "rf": {
        "n_estimators": 1200,
        "max_features": 0.3,
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
        "mean_shiftv2",
        # "multi_modal_compounding",
        # "multi_equal",
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

    # Section: varying over sample-sizes
    model_name = "knn_viewone"
    n_samples_list = [2**x for x in range(8, 13)]
    n_dims_1 = 4090
    print(n_samples_list)
    results = Parallel(n_jobs=n_jobs)(
        delayed(_run_simulation_oneview)(
            n_samples,
            n_dims_1,
            idx,
            root_dir,
            sim_name,
            model_name,
            run_view="view_one",
            overwrite=False,
        )
        for sim_name in SIMULATIONS_NAMES
        for n_samples in n_samples_list
        for idx in range(n_repeats)
    )

    # Section: varying over sample-sizes
    model_name = "knn_viewtwo"
    n_samples_list = [2**x for x in range(8, 13)]
    n_dims_1 = 4090
    print(n_samples_list)
    results = Parallel(n_jobs=n_jobs)(
        delayed(_run_simulation_oneview)(
            n_samples,
            n_dims_1,
            idx,
            root_dir,
            sim_name,
            model_name,
            run_view="view_two",
            overwrite=False,
        )
        for sim_name in SIMULATIONS_NAMES
        for n_samples in n_samples_list
        for idx in range(n_repeats)
    )

    # Section: varying over dimensions of the first view
    # n_samples = 4096
    # n_dims_list = [2**i - 6 for i in range(3, 13)]
    # print(n_dims_list)
    # results = Parallel(n_jobs=n_jobs)(
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
    #     for model_name in model_names
    # )
