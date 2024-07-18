"""Generating data for CoMIGHT simulations with S@S98."""

# A : Control ~ N(0, 1), Cancer ~ N(1, 1)
# B:  Control ~ N(0, 1), Cancer ~ 0.75*N(1, 1) + 0.25*N(5, 1)
# C:  Control~ 0.75*N(1, 1) + 0.25*N(5, 1), Cancer ~ 0.75*N(1, 1) + 0.25*N(5, 1)
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from npeet import entropy_estimators as ee

seed = 12345
rng = np.random.default_rng(seed)


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
        feature_set_ends = [
            n_dims_1,
            n_dims_1 + n_dims_2_,
        ]  # [4090, 4096] for varying samples
        assert X.shape[1] == feature_set_ends[1]

        # permute the second view
        covariate_index = np.arange(n_dims_1)
        x = X[:, covariate_index]
        z = X[:, -n_dims_2_:]
        assert len(covariate_index) + n_dims_2_ == X.shape[1]
        cmi = ee.mi(x=x, y=y, z=z, k=int(np.sqrt(n_samples)))

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


if __name__ == "__main__":
    root_dir = Path("/Volumes/Extreme Pro/cancer")
    # root_dir = Path("/data/adam/")

    SIMULATIONS_NAMES = [
        "mean_shiftv4",
        "multi_modalv2",
        "multi_equal",
    ]

    overwrite = False
    n_repeats = 100
    n_jobs = -2

    # Section: varying over samples
    # n_dims_1 = 4096 - 6
    # n_dims_1 = 1024 - 6
    # n_dims_1 = 512 - 6
    # n_samples_list = [2**x for x in range(8, 11)]
    # print(n_samples_list)
    # model_name = "ksg"
    # results = Parallel(n_jobs=n_jobs)(
    #     delayed(_run_ksg_simulation)(
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

    # varying over dimensions
    n_dims_list = [2**i - 6 for i in range(3, 10)]
    n_samples = 512
    print(n_dims_list)
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
        for n_dims_1 in n_dims_list
        for idx in range(n_repeats)
    )
