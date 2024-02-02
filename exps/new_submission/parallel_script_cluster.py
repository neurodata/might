import os
import sys
from itertools import product

import numpy as np
from joblib import Parallel, delayed
from sktree import HonestForestClassifier
from sktree.datasets import make_trunk_classification
from sktree.stats import PermutationForestClassifier, build_coleman_forest

seed = 12345
rng = np.random.default_rng(seed)

# hard-coded parameters
n_estimators = 1500
max_features = 0.3
test_size = 0.2

# TODO: depends on how many CPUs are assigned per job
n_jobs = -1


def _run_parallel_might_permutations(
    idx,
    n_samples,
    n_features,
    sim_type,
    rootdir,
):
    """Run parallel job on pre-generated data.

    Parameters
    ----------
    idx : int
        The index of the pre-generated dataset, stored as npz file.
        Also used as the random seed where applicable.
    n_samples : int
        The number of samples to keep.
    n_features : int
        The number of dimensions to keep in feature set.
    sim_type : str
        The simulation type. Either 'independent', 'collider', 'confounder',
        or 'direct-indirect'.
    rootdir : str
        The root directory where 'data/' and 'output/' will be.
    """
    # set output directory to save npz files
    output_dir = os.path.join(rootdir, f"output/{model_name}/{sim_type}/")
    os.makedirs(output_dir, exist_ok=True)

    # now compute the pvalue when shuffling all
    covariate_index = None

    # load data
    if sim_type == "trunk-overlap":
        X, y, mu, cov = make_trunk_classification(
            n_samples=n_samples,
            n_dim=n_dims,
            m_factor=1,
            return_params=True,
            seed=idx,
        )
    elif sim_type == "trunk":
        X, y, mu, cov = make_trunk_classification(
            n_samples=n_samples,
            n_dim=n_dims,
            return_params=True,
            seed=idx,
        )
    X = np.float32(X)
    y = np.float32(y)

    forest_params = OOB_MODEL_NAMES[model_name]
    tree_est = HonestForestClassifier(**forest_params)

    # compute pvalue using permutation forest
    est = PermutationForestClassifier(estimator=tree_est, test_size=test_size, seed=idx)

    observe_test_stat, pvalue = est.test(
        X, y, covariate_index=covariate_index, n_repeats=1000, metric="s@s98"
    )

    np.savez(
        os.path.join(
            output_dir, f"might_{sim_type}_{n_samples}_{n_features}_{idx}.npz"
        ),
        n_samples=n_samples,
        n_features=n_features,
        y_true=y,
        might_pvalue=pvalue,
        might_stat=observe_test_stat,
    )


def _run_parallel_might(
    idx,
    n_samples,
    n_features,
    sim_type,
    rootdir,
):
    """Run parallel job on pre-generated data.

    Parameters
    ----------
    idx : int
        The index of the pre-generated dataset, stored as npz file.
        Also used as the random seed where applicable.
    n_samples : int
        The number of samples to keep.
    n_features : int
        The number of dimensions to keep in feature set.
    sim_type : str
        The simulation type. Either 'independent', 'collider', 'confounder',
        or 'direct-indirect'.
    rootdir : str
        The root directory where 'data/' and 'output/' will be.
    output_dir : str
        The directory under ``rootdir`` to store output.
    """
    for model_name in OOB_MODEL_NAMES.keys():
        # set output directory to save npz files
        output_dir = os.path.join(rootdir, f"output/{model_name}/{sim_type}/")
        os.makedirs(output_dir, exist_ok=True)

        # now compute the pvalue when shuffling all
        covariate_index = None

        # load data
        if sim_type == "trunk-overlap":
            X, y, mu, cov = make_trunk_classification(
                n_samples=n_samples,
                n_dim=n_dims,
                m_factor=1,
                return_params=True,
                seed=idx,
            )
        elif sim_type == "trunk":
            X, y, mu, cov = make_trunk_classification(
                n_samples=n_samples,
                n_dim=n_dims,
                return_params=True,
                seed=idx,
            )
        X = np.float32(X)
        y = np.float32(y)

        forest_params = OOB_MODEL_NAMES[model_name]
        est = HonestForestClassifier(**forest_params)

        # compute pvalue
        (
            observe_test_stat,
            pvalue,
            orig_forest_proba,
            perm_forest_proba,
        ) = build_coleman_forest(
            est,
            X,
            y,
            covariate_index=covariate_index,
            metric="s@s98",
            n_repeats=1000,
            seed=None,
            return_posteriors=True,
        )

        np.savez(
            os.path.join(
                output_dir, f"might_{sim_type}_{n_samples}_{n_features}_{idx}.npz"
            ),
            n_samples=n_samples,
            n_features=n_features,
            y_true=y,
            might_pvalue=pvalue,
            might_stat=observe_test_stat,
            might_posteriors=orig_forest_proba,
            might_null_posteriors=perm_forest_proba,
        )


NON_OOB_MODEL_NAMES = {
    "might-honestfraction05-og": {
        "n_estimators": 500,
        "random_state": seed,
        "honest_fraction": 0.5,
        "n_jobs": n_jobs,
        "bootstrap": False,
        "stratify": True,
        # "max_samples": ,
        "permute_per_tree": False,
    },
}

OOB_MODEL_NAMES = {
    "might-honestfraction05-bootstrap-permuteonce": {
        "n_estimators": n_estimators,
        "random_state": seed,
        "honest_fraction": 0.5,
        "n_jobs": n_jobs,
        "bootstrap": True,
        "stratify": True,
        "max_samples": 1.6,
        "permute_per_tree": False,
    },
    "might-honestfraction05-bootstrap": {
        "n_estimators": n_estimators,
        "random_state": seed,
        "honest_fraction": 0.5,
        "n_jobs": n_jobs,
        "bootstrap": True,
        "stratify": True,
        "max_samples": 1.6,
        "permute_per_tree": True,
    },
    "might-honestfraction025-bootstrap": {
        "n_estimators": n_estimators,
        "random_state": seed,
        "honest_fraction": 0.25,
        "n_jobs": n_jobs,
        "bootstrap": True,
        "stratify": True,
        "max_samples": 1.6,
        "permute_per_tree": True,
    },
    "might-honestfraction075-bootstrap": {
        "n_estimators": n_estimators,
        "random_state": seed,
        "honest_fraction": 0.75,
        "n_jobs": n_jobs,
        "bootstrap": True,
        "stratify": True,
        "max_samples": 1.6,
        "permute_per_tree": True,
    },
}

if __name__ == "__main__":
    # Extract arguments from terminal input
    # idx = int(sys.argv[1])
    # n_samples = int(sys.argv[2])
    # n_dims = int(sys.argv[3])
    # model_name = sys.argv[4]
    # sim_type = sys.argv[5]
    # rootdir = sys.argv[6]

    # TODO: add root dir here
    rootdir = ""

    SIM_TYPES = ["trunk", "trunk-overlap"]
    n_samples_list = [2**i for i in range(8, 12)]
    n_repeats = 100
    n_dims = 4096

    for sim_type in SIM_TYPES:
        for n_samples in n_samples_list:
            for idx in range(n_repeats):
                # Call your function with the extracted arguments
                _run_parallel_might(idx, n_samples, n_dims, sim_type, rootdir)
