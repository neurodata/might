import sys
from pathlib import Path

import numpy as np
from hyppo.conditional import ConditionalDcorr
from joblib import Parallel, delayed
from numpy import log
from scipy.special import digamma
from sklearn.model_selection import StratifiedShuffleSplit


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

    # now compute the pvalue when shuffling X2
    covariate_index = np.arange(n_dims_1)
    assert len(covariate_index) == n_dims_2_

    cdcorr = ConditionalDcorr(bandwidth="silverman")
    Z = X[:, covariate_index]
    assert Z.shape[1] == n_dims_2_
    assert X.shape[1] == n_dims_1 + n_dims_2_
    mask_array = np.ones(X.shape[1])
    mask_array[covariate_index] = 0
    mask_array = mask_array.astype(bool)
    try:
        X_minus_Z = X[:, mask_array]
        if np.var(y) < 0.001:
            raise RuntimeError(
                f"{n_samples}_{n_dims_1}_{idx} errored out with no variance in y"
            )
        cdcorr_stat, cdcorr_pvalue = cdcorr.test(
            X_minus_Z.copy(), y.copy(), Z.copy(), random_state=idx
        )

        np.savez(
            output_fname,
            n_samples=n_samples,
            y=y,
            cdcorr_pvalue=cdcorr_pvalue,
            idx=idx,
            cdcorr_stat=cdcorr_stat,
            n_dims_2=n_dims_2_,
            n_dims_1=n_dims_1,
            sim_type=sim_name,
        )

    except Exception as e:
        print(e)


if __name__ == "__main__":
    # Extract arguments from terminal input
    idx = int(sys.argv[1])
    n_samples = int(sys.argv[2])
    n_dims_1 = int(sys.argv[3])
    sim_name = sys.argv[4]
    root_dir = sys.argv[5]

    model_name = "cdcorr"
    _run_simulation(
        n_samples,
        n_dims_1,
        idx,
        Path(root_dir),
        sim_name,
        model_name,
        overwrite=False,
    )

    # SIMULATIONS_NAMES = [
    #     "mean_shift_compounding",
    #     "multi_modal_compounding",
    #     "multi_equal",
    # ]
    # n_jobs = 1

    # overwrite = False
    # n_repeats = 100
    # model_name = "cdcorr"

    # # Section: varying over dimensions
    # n_samples = 4096
    # n_dims_list = [2**i - 6 for i in range(3, 13)]
    # print(n_dims_list)
    # results = Parallel(n_jobs=n_jobs, verbose=True)(
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

    # # Section: varying over sample-sizes
    # n_samples_list = [2**x for x in range(8, 13)]
    # n_dims_1 = 4090
    # print(n_samples_list)
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
    #     for n_samples in n_samples_list
    #     for idx in range(n_repeats)
    # )
