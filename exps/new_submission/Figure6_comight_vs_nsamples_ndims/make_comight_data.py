"""Generating data for CoMIGHT simulations with S@S98."""

# A : Control ~ N(0, 1), Cancer ~ N(1, 1)
# B:  Control ~ N(0, 1), Cancer ~ 0.75*N(1, 1) + 0.25*N(5, 1)
# C:  Control~ 0.75*N(1, 1) + 0.25*N(5, 1), Cancer ~ 0.75*N(1, 1) + 0.25*N(5, 1)
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from treeple.datasets import make_trunk_mixture_classification


def make_mean_shift(
    root_dir,
    n_samples=4096,
    n_dim_1=4090,
    mu_0=0,
    mu_1=1,
    seed=None,
    n_dim_2=6,
    return_params=False,
    overwrite=False,
):
    """Make mean shifted binary classification data.

    X comprises of [view_1, view_2] where view_1 is the first ``n_dim_1`` dimensions
    and view_2 is the last ``n_dim_2`` dimensions.

    view_1 is generated, such that [A, B] corresponding to class labels [0, 1]
    are generated as follows:

    A ~ N(1, I)
    B ~ N(m_factor, I)

    view_2 is generated, such that [A, B] corresponding to class labels [0, 1]
    are generated as follows:

    A ~ N(1 / np.sqrt(2), I)
    B ~ N(1 / np.sqrt(2) * m_factor, I)

    Parameters
    ----------
    n_samples : int, optional
        The number of samples to generate, by default 1024.
    n_dim_1 : int, optional
        The number of dimensions in first view, by default 4090.
    mu_0 : int, optional
        The mean of the first class, by default -1.
    mu_1 : int, optional
        The mean of the second class, by default 1.
    seed : int, optional
        Random seed, by default None.
    n_dim_2 : int, optional
        The number of dimensions in second view, by default 6.
    return_params : bool
        Whether to return parameters of the generating model or not. Default is False.

    Returns
    -------
    X : ArrayLike of shape (n_samples, n_dim_1 + n_dim_2)
        Data.
    y : ArrayLike of shape (n_samples,)
        Labels.
    """
    output_fname = (
        root_dir
        / "data"
        / "mean_shift_compounding"
        / f"mean_shift_compounding_{n_samples}_{n_dim_1}_{n_dim_2}_{seed}.npz"
    )
    output_fname.parent.mkdir(exist_ok=True, parents=True)
    if not overwrite and output_fname.exists():
        return
    rng = np.random.default_rng(seed)
    default_n_informative = 1

    # X, y, means, cov = make_trunk_classification(
    #     n_samples=n_samples // 2,
    #     n_dim=n_dim_1 + 1,
    #     n_informative=default_n_informative,
    #     mu_0=mu_0,
    #     mu_1=mu_1,
    #     return_params=True,
    #     rho=0.5,
    #     seed=seed,
    # )

    method = "svd"
    mu_1_vec = np.array([-0.5, 0.5])
    mu_0_vec = np.array([0 / np.sqrt(i) for i in range(1, 3)])
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])

    X = np.vstack(
        (
            rng.multivariate_normal(mu_1_vec, cov, n_samples // 2, method=method),
            rng.multivariate_normal(mu_0_vec, cov, n_samples // 2, method=method),
        )
    )
    assert X.shape[1] == 2
    view_1 = X[:, (0,)]
    view_1 = np.hstack(
        (view_1, rng.normal(loc=0, scale=1, size=(X.shape[0], n_dim_1 - 1)))
    )
    view_2 = X[:, 1:]

    # # get the second informative dimension
    # view_1 = X[:, 1:]

    # # only take one informative dimension
    # view_2 = X[:, (0,)]

    # add noise to the second view so that view_2 = (n_samples, n_dim_2)
    view_2 = np.concatenate(
        (view_2, rng.standard_normal(size=(n_samples, n_dim_2 - view_2.shape[1]))),
        axis=1,
    )

    X = np.concatenate((view_1, view_2), axis=1)
    y = np.concatenate((np.zeros(n_samples // 2), np.ones(n_samples // 2)))

    np.savez_compressed(output_fname, X=X, y=y)


def make_multi_modal(
    root_dir,
    n_samples=4096,
    n_dim_1=4090,
    mu_0=0,
    mu_1=5,
    mix=0.75,
    seed=None,
    n_dim_2=6,
    return_params=False,
    overwrite=False,
):
    """Make multi-modal binary classification data.

    X comprises of [view_1, view_2] where view_1 is the first ``n_dim_1`` dimensions
    and view_2 is the last ``n_dim_2`` dimensions.

    view_1 is generated, such that [A, B] corresponding to class labels [0, 1]
    are generated as follows:

    A ~ N(0, I)
    B ~ mix * N(1, I) + (1 - mix) * N(m_factor, I)

    view_2 is generated, such that [A, B] corresponding to class labels [0, 1]
    are generated as follows:

    A ~ N(0, I)
    B ~ mix * N(1 / np.sqrt(2), I) + (1 - mix) * N(1 / np.sqrt(2) * m_factor, I)

    Parameters
    ----------
    n_samples : int, optional
        The number of samples to generate, by default 1024.
    n_dim_1 : int, optional
        The number of dimensions in first view, by default 4090.
    mu_0 : int, optional
        The mean of the first class, by default 1.
    mu_1 : int, optional
        The mean of the second class, by default -1.
    seed : int, optional
        Random seed, by default None.
    n_dim_2 : int, optional
        The number of dimensions in second view, by default 6.
    return_params : bool
        Whether to return parameters of the generating model or not. Default is False.

    Returns
    -------
    X : ArrayLike of shape (n_samples, n_dim_1 + n_dim_2)
        Data.
    y : ArrayLike of shape (n_samples,)
        Labels.
    """
    output_fname = (
        root_dir
        / "data"
        / "multi_modal_compounding"
        / f"multi_modal_compounding_{n_samples}_{n_dim_1}_{n_dim_2}_{seed}.npz"
    )
    output_fname.parent.mkdir(exist_ok=True, parents=True)
    if not overwrite and output_fname.exists():
        return

    rng = np.random.default_rng(seed)
    default_n_informative = 2

    # X, y, means, covs, X_mixture = make_trunk_mixture_classification(
    #     n_samples=n_samples,
    #     n_dim=n_dim_1 + 1,
    #     n_informative=default_n_informative,
    #     mu_0=mu_0,
    #     mu_1=mu_1,
    #     mix=mix,
    #     scaling_factor=1,
    #     seed=seed,
    #     rho=0.5,
    #     return_params=True,
    # )
    # get the second informative dimension
    # view_1 = X[:, 1:]

    # # only take one informative dimension
    # view_2 = X[:, (0,)]

    method = "svd"
    mu_1_vec = np.array([-1, 2])
    mu_0_vec = np.array([0 / np.sqrt(i) for i in range(1, 3)])
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])

    mixture_idx = rng.choice(2, n_samples // 2, replace=True, shuffle=True, p=[mix, 1 - mix])  # type: ignore

    norm_params = [[mu_0_vec, cov], [mu_1_vec, cov]]
    X_mixture = np.fromiter(
        (
            rng.multivariate_normal(*(norm_params[i]), size=1, method=method)
            for i in mixture_idx
        ),
        dtype=np.dtype((float, 2)),
    )

    X = np.vstack(
        (
            rng.multivariate_normal(np.zeros(2), cov, n_samples // 2, method=method),
            X_mixture.reshape(n_samples // 2, 2),
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
    np.savez_compressed(output_fname, X=X, y=y)


def make_multi_equal(
    root_dir,
    n_samples=4096,
    n_dim_1=4090,
    mu_0=1,
    mu_1=5,
    mix=0.75,
    seed=None,
    n_dim_2=6,
    return_params=False,
):
    """Make multi-modal binary classification data.

    X comprises of [view_1, view_2] where view_1 is the first ``n_dim_1`` dimensions
    and view_2 is the last ``n_dim_2`` dimensions.

    view_1 is generated, such that [A, B] corresponding to class labels [0, 1]
    are generated as follows:

    A ~ mix * N(1, I) + (1 - mix) * N(m_factor, I)
    B ~ mix * N(1, I) + (1 - mix) * N(m_factor, I)

    view_2 is generated, such that [A, B] corresponding to class labels [0, 1]
    are generated as follows:

    A ~ mix * N(1 / np.sqrt(2), I) + (1 - mix) * N(1 / np.sqrt(2) * m_factor, I)
    B ~ mix * N(1 / np.sqrt(2), I) + (1 - mix) * N(1 / np.sqrt(2) * m_factor, I)

    Parameters
    ----------
    n_samples : int, optional
        The number of samples to generate, by default 1024.
    n_dim_1 : int, optional
        The number of dimensions in first view, by default 4090.
    mu_0 : int, optional
        The mean of the first class, by default 1.
    mu_1 : int, optional
        The mean of the second class, by default -1.
    seed : int, optional
        Random seed, by default None.
    n_dim_2 : int, optional
        The number of dimensions in second view, by default 6.
    return_params : bool
        Whether to return parameters of the generating model or not. Default is False.

    Returns
    -------
    X : ArrayLike of shape (n_samples, n_dim_1 + n_dim_2)
        Data.
    y : ArrayLike of shape (n_samples,)
        Labels.
    """
    output_fname = (
        root_dir
        / "data"
        / "multi_equal"
        / f"multi_equal_{n_samples}_{n_dim_1}_{n_dim_2}_{seed}.npz"
    )
    output_fname.parent.mkdir(exist_ok=True, parents=True)

    rng = np.random.default_rng(seed)
    default_n_informative = 2

    X1, _ = make_trunk_mixture_classification(
        n_samples=n_samples,
        n_dim=n_dim_1 + 1,
        n_informative=default_n_informative,
        mu_0=mu_0,
        mu_1=mu_1,
        mix=mix,
        rho=0.5,
        seed=rng.integers(0, np.iinfo(np.int32).max),
        return_params=False,
    )
    # only keep the second half of samples, corresponding to the mixture
    X1 = X1[n_samples // 2 :, :]

    X2, _ = make_trunk_mixture_classification(
        n_samples=n_samples,
        n_dim=n_dim_1 + 1,
        n_informative=default_n_informative,
        mu_0=mu_0,
        mu_1=mu_1,
        mix=mix,
        rho=0.5,
        seed=rng.integers(0, np.iinfo(np.int32).max),
        return_params=False,
    )
    # only keep the second half of samples, corresponding to the mixture
    X2 = X2[n_samples // 2 :, :]

    X = np.vstack((X1, X2))
    y = np.hstack((np.zeros(n_samples // 2), np.ones(n_samples // 2)))

    # get the second informative dimension
    view_1 = X[:, 1:]

    # only take one informative dimension
    view_2 = X[:, (0,)]

    # add noise to the second view so that view_2 = (n_samples, n_dim_2)
    view_2 = np.concatenate(
        (view_2, rng.standard_normal(size=(n_samples, n_dim_2 - view_2.shape[1]))),
        axis=1,
    )
    X = np.concatenate((view_1, view_2), axis=1)
    np.savez_compressed(output_fname, X=X, y=y)


if __name__ == "__main__":
    root_dir = sys.argv[1]

    overwrite = False

    # Section: Make data
    # root_dir = Path("/Volumes/Extreme Pro/cancer")
    # root_dir = Path("/data/adam/")

    n_repeats = 100
    Parallel(n_jobs=-1)(
        delayed(func)(
            Path(root_dir),
            seed=seed,
        )
        for seed in range(n_repeats)
        for func in [
            # make_mean_shift,
            # make_multi_modal,
            make_multi_equal,
        ]
    )
