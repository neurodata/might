"""Generating data for CoMIGHT simulations with S@S98."""

# A : Control ~ N(0, 1), Cancer ~ N(1, 1)
# B:  Control ~ N(0, 1), Cancer ~ 0.75*N(1, 1) + 0.25*N(5, 1)
# C:  Control~ 0.75*N(1, 1) + 0.25*N(5, 1), Cancer ~ 0.75*N(1, 1) + 0.25*N(5, 1)
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from npeet import entropy_estimators as ee
from sklearn.model_selection import StratifiedShuffleSplit
from sktree import HonestForestClassifier
from sktree.datasets import (make_trunk_classification,
                             make_trunk_mixture_classification)
from sktree.stats import (PermutationHonestForestClassifier,
                          build_hyppo_oob_forest)
from sktree.stats.utils import _mutual_information
from sktree.tree import MultiViewDecisionTreeClassifier

seed = 12345
rng = np.random.default_rng(seed)

### hard-coded parameters
n_estimators = 6000
max_features = 0.3


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
        _cv = StratifiedShuffleSplit(
            n_splits=1, train_size=n_samples, random_state=seed
        )
        for train_idx, _ in _cv.split(X, y):
            continue
        X = X[train_idx, :]
        y = y[train_idx, ...].squeeze()
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
        might_kwargs = MODEL_NAMES["might"]
        feature_set_ends = [
            n_dims_1,
            n_dims_1 + n_dims_2_,
        ]  # [4090, 4096] for varying samples
        assert X.shape[1] == feature_set_ends[1]
        est = HonestForestClassifier(
            seed=seed, feature_set_ends=feature_set_ends, **might_kwargs
        )
        perm_est = PermutationHonestForestClassifier(
            seed=seed, feature_set_ends=feature_set_ends, **might_kwargs
        )
        # permute the second view
        covariate_index = np.arange(n_dims_1, n_dims_1 + n_dims_2_)

        est, posterior_arr = build_hyppo_oob_forest(
            est,
            X,
            y,
            verbose=False,
        )
        perm_est, perm_posterior_arr = build_hyppo_oob_forest(
            perm_est, X, y, verbose=False, covariate_index=covariate_index
        )

        # mutual information for both
        y_pred_proba = np.nanmean(posterior_arr, axis=0)
        I_X1X2_Y = _mutual_information(y, y_pred_proba)

        y_pred_proba = np.nanmean(perm_posterior_arr, axis=0)
        I_X1_Y = _mutual_information(y, y_pred_proba)

        if np.isnan(I_X1_Y) or np.isnan(I_X1X2_Y):
            raise RuntimeError(f"NaN values for {output_fname}")
        np.savez_compressed(
            output_fname,
            idx=idx,
            n_samples=n_samples,
            n_dims_1=n_dims_1,
            n_dims_2=n_dims_2_,
            cmi=I_X1X2_Y - I_X1_Y,
            I_X1X2_Y=I_X1X2_Y,
            I_X1_Y=I_X1_Y,
            sim_type=sim_name,
            y=y,
            posterior_arr=posterior_arr,
            perm_posterior_arr=perm_posterior_arr,
        )


def _run_ksg_simulation(
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
        / "output-mi"
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
        _cv = StratifiedShuffleSplit(
            n_splits=1, train_size=n_samples, random_state=seed
        )
        for train_idx, _ in _cv.split(X, y):
            continue
        X = X[train_idx, :]
        y = y[train_idx, ...].squeeze()
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
        feature_set_ends = [
            n_dims_1,
            n_dims_1 + n_dims_2_,
        ]  # [4090, 4096] for varying samples
        assert X.shape[1] == feature_set_ends[1]

        # permute the second view
        covariate_index = np.arange(n_dims_1, n_dims_1 + n_dims_2_)
        x = X[:, covariate_index]
        z = X[:, :n_dims_1]
        cmi = ee.mi(x=x, y=y, z=z, k=3)

        print(x.shape, y.shape, z.shape, cmi)
        if np.isnan(cmi):
            raise RuntimeError(f"NaN values for {output_fname}")

        np.savez_compressed(
            output_fname,
            idx=idx,
            n_samples=n_samples,
            n_dims_1=n_dims_1,
            n_dims_2=n_dims_2_,
            cmi=cmi,
            sim_type=sim_name,
            y=y,
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
    root_dir = Path("/Volumes/Extreme Pro/cancer")
    root_dir = Path("/data/adam/")

    SIMULATIONS_NAMES = [
        "mean_shift_compounding",
        "multi_modal_compounding",
        "multi_equal",
    ]

    overwrite = False
    n_repeats = 100
    n_jobs = 20
    n_samples = 4096

    # Section: varying over dims
    # model_name = "comight-cmi"
    # n_dims_1_list = [2**i - 6 for i in range(3, 12)[::2]] + [2**12-6]
    # print('Analyzing for the following dims: ', n_dims_1_list)
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
    #     for n_dims_1 in n_dims_1_list
    #     for idx in range(n_repeats)
    # )

    # Section: varying over dims
    n_dims_1_list = [2**i - 6 for i in range(3, 12)[::2]] + [2**12 - 6]
    print(n_dims_1_list)

    model_name = "ksg"
    results = Parallel(n_jobs=n_jobs)(
        delayed(_run_ksg_simulation)(
            n_samples,
            n_dims_1,
            idx,
            root_dir,
            sim_name,
            model_name,
            overwrite=False,
        )
        for sim_name in SIMULATIONS_NAMES
        for n_dims_1 in n_dims_1_list
        for idx in range(n_repeats)
    )
