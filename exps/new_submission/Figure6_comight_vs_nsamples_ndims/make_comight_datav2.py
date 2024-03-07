"""Generating data for CoMIGHT simulations with S@S98."""
import sys
from pathlib import Path
import numpy as np
from sktree.datasets import make_trunk_classification, make_trunk_mixture_classification
from pathlib import Path
from collections import defaultdict
import numpy as np
from joblib import delayed, Parallel


def make_mean_shift(
    root_dir,
    n_samples=4096,
    n_dim_1=4090,
    mu_viewone=-1,
    mu_viewtwo=1,
    rho=0.2,
    seed=None,
    n_dim_2=6,
    return_params=False,
    overwrite=False,
):
    """Make mean shifted binary classification data.

    X | Y = 0 ~ N(0, I)
    X | Y = 1 ~ N([mu_1, mu_2], [[1, 0.5], [0.5, 1]])

    We want to do a parameter sweep over mu_1 = mu_2 and get:

    - CMI
        - MI of each of the two individually
    - AUC of the combined views
        - AUC of each of the two indivdually
    - S@98 of the combined views
        - S@98 of each of the two individually
    """
    output_fname = (
        root_dir
        / "data"
        / "mean_shiftv2"
        / f"mean_shiftv2_{n_samples}_{n_dim_1}_{n_dim_2}_{seed}.npz"
    )
    output_fname.parent.mkdir(exist_ok=True, parents=True)
    if not overwrite and output_fname.exists():
        return
    
    rng = np.random.default_rng(seed)

    method = "svd"

    mu_1_vec = np.array([mu_viewone, mu_viewtwo])
    mu_0_vec = np.array([0, 0])
    cov = np.array([[1.0, rho], [rho, 1.0]])

    X = np.vstack(
        (
            rng.multivariate_normal(mu_1_vec, cov, n_samples // 2, method=method),
            rng.multivariate_normal(mu_0_vec, np.eye(2), n_samples // 2, method=method),
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
    # if return_params:
    #     return X, y, [mu_0_vec, mu_1_vec], [np.eye(2), cov]
    # return X, y

    np.savez_compressed(output_fname, X=X, y=y)


if __name__ == "__main__":
    # root_dir = sys.argv[1]

    overwrite = False
    n_repeats = 100

    # Section: Make data
    root_dir = Path("/Volumes/Extreme Pro/cancer")
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
            # make_multi_modal,
            # make_multi_equal,
        ]
    )