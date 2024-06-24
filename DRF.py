#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from drf import drf
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
import math


import warnings

warnings.filterwarnings("ignore")
# In[2]:


def hellinger_dot(p, q):
    """Hellinger distance between two discrete distributions.
    Using numpy.
    For Python >= 3.5 only"""
    z = np.sqrt(p) - np.sqrt(q)
    #  print(z.shape)
    return np.linalg.norm(z) / math.sqrt(2 * len(z))


#  return np.sqrt(z @ z / 2)


# In[3]:


def make_trunk_classification(
    n_samples,
    n_dim=4096,
    n_informative=1,
    simulation: str = "trunk",
    mu_0: float = 0,
    mu_1: float = 1,
    rho: int = 0,
    band_type: str = "ma",
    return_params: bool = False,
    mix: float = 0.5,
    seed=None,
):
    if n_dim < n_informative:
        raise ValueError(
            f"Number of informative dimensions {n_informative} must be less than number "
            f"of dimensions, {n_dim}"
        )
    rng = np.random.default_rng(seed=seed)
    rng1 = np.random.default_rng(seed=seed)
    mu_0 = np.array([mu_0 / np.sqrt(i) for i in range(1, n_informative + 1)])
    mu_1 = np.array([mu_1 / np.sqrt(i) for i in range(1, n_informative + 1)])
    if rho != 0:
        if band_type == "ma":
            cov = _moving_avg_cov(n_informative, rho)
        elif band_type == "ar":
            cov = _autoregressive_cov(n_informative, rho)
        else:
            raise ValueError(f'Band type {band_type} must be one of "ma", or "ar".')
    else:
        cov = np.identity(n_informative)
    if mix < 0 or mix > 1:
        raise ValueError("Mix must be between 0 and 1.")
    # speed up computations for large multivariate normal matrix with SVD approximation
    if n_informative > 1000:
        method = "cholesky"
    else:
        method = "svd"
    if simulation == "trunk":
        X = np.vstack(
            (
                rng.multivariate_normal(mu_0, cov, n_samples // 2, method=method),
                rng1.multivariate_normal(mu_1, cov, n_samples // 2, method=method),
            )
        )
    elif simulation == "trunk_overlap":
        mixture_idx = rng.choice(
            2, n_samples // 2, replace=True, shuffle=True, p=[mix, 1 - mix]
        )
        norm_params = [[mu_0, cov], [mu_1, cov]]
        X_mixture = np.fromiter(
            (
                rng.multivariate_normal(*(norm_params[i]), size=1, method=method)
                for i in mixture_idx
            ),
            dtype=np.dtype((float, n_informative)),
        )
        X_mixture_2 = np.fromiter(
            (
                rng1.multivariate_normal(*(norm_params[i]), size=1, method=method)
                for i in mixture_idx
            ),
            dtype=np.dtype((float, n_informative)),
        )
        X = np.vstack(
            (
                X_mixture.reshape(n_samples // 2, n_informative),
                X_mixture_2.reshape(n_samples // 2, n_informative),
            )
        )
    elif simulation == "trunk_mix":
        mixture_idx = rng.choice(
            2, n_samples // 2, replace=True, shuffle=True, p=[mix, 1 - mix]
        )
        norm_params = [[mu_0, cov], [mu_1, cov]]
        X_mixture = np.fromiter(
            (
                rng1.multivariate_normal(*(norm_params[i]), size=1, method=method)
                for i in mixture_idx
            ),
            dtype=np.dtype((float, n_informative)),
        )
        X = np.vstack(
            (
                rng.multivariate_normal(
                    np.zeros(n_informative), cov, n_samples // 2, method=method
                ),
                X_mixture.reshape(n_samples // 2, n_informative),
            )
        )
    else:
        raise ValueError(f"Simulation must be: trunk, trunk_overlap, trunk_mix")
    if n_dim > n_informative:
        X = np.hstack(
            (X, rng.normal(loc=0, scale=1, size=(X.shape[0], n_dim - n_informative)))
        )
    y = np.concatenate((np.zeros(n_samples // 2), np.ones(n_samples // 2)))
    if return_params:
        returns = [X, y]
        if simulation == "trunk":
            returns += [[mu_0, mu_1], [cov, cov]]
        elif simulation == "trunk-overlap":
            returns += [[np.zeros(n_informative), np.zeros(n_informative)], [cov, cov]]
        elif simulation == "trunk-mix":
            returns += [*list(zip(*norm_params)), X_mixture]
        return returns
    return X, y


# In[4]:

N_ITR = 10

dim = 4096
SAMPLE_SIZES = [2**i for i in range(8, 13)]
for i in range(N_ITR):
    X, Y = make_trunk_classification(
        n_samples=4096,
        n_dim=4096,
        n_informative=1,
        mu_0=0,
        mu_1=1,
        # return_params=True,
        seed=i,
        # mix = 0.75,
        simulation="trunk",
        rho=0,
    )

    hell_dists = []
    for samp in SAMPLE_SIZES:
        X_0 = X[Y == 0]
        x_0 = X_0[: samp // 2, :dim]

        X_1 = X[Y == 1]
        x_1 = X_1[: samp // 2, :dim]

        x = np.vstack((x_0, x_1))
        y = np.array([0] * x_0.shape[0] + [1] * x_1.shape[0]).ravel()
        cv = StratifiedKFold(n_splits=4, shuffle=True)

        hell_dist = []
        for train_ix, test_ix in cv.split(x, y):
            X_train, X_test = x[train_ix, :], x[test_ix, :]
            y_train, y_test = y[train_ix], y[test_ix]
            X_train_df = pd.DataFrame(X_train)
            Y_train_df = pd.DataFrame(y_train)

            # fit model
            DRF = drf(num_trees=6000, num_threads = 4, mtry=math.floor(X_train.shape[1] * 0.3))
            DRF.fit(X_train_df, Y_train_df)
            out = DRF.predict(newdata=X_test, functional="mean")
            #    print(X_test.shape,out.mean.shape)
            #    print(len(out.mean[y_test == 0]))
            dist = roc_auc_score(y_test, out.mean)
            # hellinger_dot(out.mean[y_test == 0], out.mean[y_test == 1])
            hell_dist.append(dist)
        print(i, samp)
        hell_dists.append(np.mean(hell_dist))

    np.savetxt(
        "/home/hao/drf/trunk_auc_samp_rep{}.csv".format(i),
        hell_dists,
        delimiter=",",
    )


# In[5]:


dim = 4096
SAMPLE_SIZES = [2**i for i in range(8, 13)]
for i in range(N_ITR):
    X, Y = make_trunk_classification(
        n_samples=4096,
        n_dim=4096,
        n_informative=1,
        mu_0=0,
        mu_1=5,
        # return_params=True,
        seed=i,
        mix=0.75,
        simulation="trunk_mix",
        rho=0,
    )

    hell_dists = []
    for samp in SAMPLE_SIZES:
        X_0 = X[Y == 0]
        x_0 = X_0[: samp // 2, :dim]

        X_1 = X[Y == 1]
        x_1 = X_1[: samp // 2, :dim]

        x = np.vstack((x_0, x_1))
        y = np.array([0] * x_0.shape[0] + [1] * x_1.shape[0]).ravel()
        cv = StratifiedKFold(n_splits=4, shuffle=True)

        hell_dist = []
        for train_ix, test_ix in cv.split(x, y):
            X_train, X_test = x[train_ix, :], x[test_ix, :]
            y_train, y_test = y[train_ix], y[test_ix]
            X_train_df = pd.DataFrame(X_train)
            Y_train_df = pd.DataFrame(y_train)

            # fit model
            DRF = drf(num_trees=6000, num_threads = 4, mtry=math.floor(X_train.shape[1] * 0.3))
            DRF.fit(X_train_df, Y_train_df)
            out = DRF.predict(newdata=X_test, functional="mean")
            #    print(X_test.shape,out.mean.shape)
            #    print(len(out.mean[y_test == 0]))
            dist = roc_auc_score(y_test, out.mean)
            # hellinger_dot(out.mean[y_test == 0], out.mean[y_test == 1])
            hell_dist.append(dist)
        print(i, samp)
        hell_dists.append(np.mean(hell_dist))

    np.savetxt(
        "/home/hao/drf/trunk_mix_auc_sam_rep{}.csv".format(i),
        hell_dists,
        delimiter=",",
    )


# In[6]:


dim = 4096
SAMPLE_SIZES = [2**i for i in range(8, 13)]
for i in range(N_ITR):
    X, Y = make_trunk_classification(
        n_samples=4096,
        n_dim=4096,
        n_informative=1,
        mu_0=0,
        mu_1=5,
        # return_params=True,
        seed=i,
        mix=0.75,
        simulation="trunk_overlap",
        rho=0,
    )

    hell_dists = []
    for samp in SAMPLE_SIZES:
        X_0 = X[Y == 0]
        x_0 = X_0[: samp // 2, :dim]

        X_1 = X[Y == 1]
        x_1 = X_1[: samp // 2, :dim]

        x = np.vstack((x_0, x_1))
        y = np.array([0] * x_0.shape[0] + [1] * x_1.shape[0]).ravel()
        cv = StratifiedKFold(n_splits=4, shuffle=True)

        hell_dist = []
        for train_ix, test_ix in cv.split(x, y):
            X_train, X_test = x[train_ix, :], x[test_ix, :]
            y_train, y_test = y[train_ix], y[test_ix]
            X_train_df = pd.DataFrame(X_train)
            Y_train_df = pd.DataFrame(y_train)

            # fit model
            DRF = drf(num_trees=6000, num_threads = 4, mtry=math.floor(X_train.shape[1] * 0.3))
            DRF.fit(X_train_df, Y_train_df)
            out = DRF.predict(newdata=X_test, functional="mean")
            #    print(X_test.shape,out.mean.shape)
            #    print(len(out.mean[y_test == 0]))
            dist = roc_auc_score(y_test, out.mean)
            # hellinger_dot(out.mean[y_test == 0], out.mean[y_test == 1])
            hell_dist.append(dist)
        print(i, samp)
        hell_dists.append(np.mean(hell_dist))

    np.savetxt(
        "/home/hao/drf/trunk_overlap_auc_samp_rep{}.csv".format(i),
        hell_dists,
        delimiter=",",
    )
