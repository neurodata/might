from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from treeple.stats.utils import _mutual_information

seed = 12345
rng = np.random.default_rng(seed)


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
    data = np.load(fname, allow_pickle=True)
    X, y = data["X"], data["y"]
    print(X.shape, y.shape)
    if n_samples < X.shape[0]:
        # _cv = StratifiedShuffleSplit(
        #     n_splits=1, train_size=n_samples, random_state=seed
        # )
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
        view_two = X[:, -n_dims_2_:]
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
        if model_name == "rf":
            est = RandomForestClassifier(random_state=seed, **MODEL_NAMES[model_name])
            perm_est = RandomForestClassifier(
                random_state=seed, **MODEL_NAMES[model_name]
            )
        elif "knn" in model_name:
            est = KNeighborsClassifier(
                n_neighbors=int(np.sqrt(n_samples) + 1),
            )
            perm_est = KNeighborsClassifier(
                n_neighbors=int(np.sqrt(n_samples) + 1),
            )
        elif model_name == "svm":
            est = SVC(random_state=seed, **MODEL_NAMES[model_name])
            perm_est = SVC(random_state=seed, **MODEL_NAMES[model_name])
        elif model_name == "lr":
            est = LogisticRegression(random_state=seed, **MODEL_NAMES[model_name])
            perm_est = LogisticRegression(random_state=seed, **MODEL_NAMES[model_name])
        # train model in...
        cv = StratifiedKFold(n_splits=5, shuffle=True)

        # permute the second view
        covariate_index = np.arange(n_dims_1)
        assert len(covariate_index) + n_dims_2_ == X.shape[1]

        # build the model on permuted and non-permuted data
        posterior_arr = np.full((n_samples, 2), np.nan, dtype=np.float32)
        perm_posterior_arr = np.full((n_samples, 2), np.nan, dtype=np.float32)

        for idx, (train_ix, test_ix) in enumerate(cv.split(X, y)):
            X_train, X_test = X[train_ix, :], X[test_ix, :]
            y_train, y_test = y[train_ix], y[test_ix]

            est.fit(X_train, y_train)
            observe_proba = est.predict_proba(X_test)
            posterior_arr[test_ix, :] = observe_proba

            # permute the second view
            X_train_perm = X_train.copy()
            X_train_perm[:, covariate_index] = rng.permutation(
                X_train[:, covariate_index]
            )
            perm_est.fit(X_train_perm, y_train)
            perm_proba = perm_est.predict_proba(X_test)

            perm_posterior_arr[test_ix, :] = perm_proba

        # mutual information for both
        y_pred_proba = np.nanmean(posterior_arr, axis=0)
        I_XZ_Y = _mutual_information(y, y_pred_proba)

        y_pred_proba = np.nanmean(perm_posterior_arr, axis=0)
        I_Z_Y = _mutual_information(y, y_pred_proba)

        if np.isnan(I_XZ_Y) or np.isnan(I_Z_Y):
            raise RuntimeError(f"NaN values for {output_fname}")

        print(
            I_XZ_Y - I_Z_Y,
            I_XZ_Y,
            I_Z_Y,
            sim_name,
            y.shape,
            posterior_arr.shape,
            perm_posterior_arr.shape,
        )
        np.savez_compressed(
            output_fname,
            idx=idx,
            n_samples=n_samples,
            n_dims_1=n_dims_1,
            n_dims_2=n_dims_2_,
            cmi=I_XZ_Y - I_Z_Y,
            I_XZ_Y=I_XZ_Y,
            I_Z_Y=I_Z_Y,
            sim_type=sim_name,
            y=y,
            posterior_arr=posterior_arr,
            perm_posterior_arr=perm_posterior_arr,
        )


MODEL_NAMES = {
    "rf": {
        "n_estimators": 1200,
        "max_features": 0.3,
    },
    "knn": {
        # XXX: above, we use sqrt of the total number of samples to allow
        # scaling wrt the number of samples
        # "n_neighbors": 5,
    },
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
    # root_dir = Path("/data/adam/")

    SIMULATIONS_NAMES = [
        "mean_shiftv4",
        "multi_modalv2",
        "multi_equal",
    ]

    overwrite = True
    n_repeats = 100
    n_jobs = 1

    # Section: varying kNN over sample-sizes
    n_dims_1 = 4090
    n_dims_1 = 512 - 6
    n_samples_list = [2**x for x in range(8, 11)]
    print(n_samples_list)
    for model_name in ["knn", "rf", "svm", "lr"]:
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
        )

    # Section: varying kNN over dimensions of the both views
    n_dims_list = [2**i - 6 for i in range(3, 11)]
    n_samples = 512
    print(n_dims_list)
    for model_name in ["knn", "rf", "svm", "lr"]:
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
        )
