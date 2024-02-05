import os
import sys
from itertools import product

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sktree import HonestForestClassifier
from sktree.datasets import make_trunk_classification
from sktree.stats import (
    PermutationForestClassifier,
    build_coleman_forest,
    PermutationHonestForestClassifier,
)

seed = 12345
rng = np.random.default_rng(seed)

# hard-coded parameters
n_estimators = 1500
max_features = 0.3
test_size = 0.2

# TODO: depends on how many CPUs are assigned per job
n_jobs = -1
n_jobs_trees = 1

n_repeats = 1000


def sensitivity_at_specificity(y_true, y_score, target_specificity=0.98, pos_label=1):
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

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score_binary, pos_label=pos_label)

    # Find the threshold corresponding to the target specificity
    index = np.argmax(fpr >= (1 - target_specificity))
    threshold_at_specificity = thresholds[index]

    # Compute sensitivity at the chosen specificity
    # sensitivity = tpr[index]
    # return sensitivity

    # Use the threshold to classify predictions
    y_pred_at_specificity = (y_score_binary >= threshold_at_specificity).astype(int)

    # Compute sensitivity at the chosen specificity
    sensitivity = np.sum((y_pred_at_specificity == 1) & (y_true == 1)) / np.sum(
        y_true == 1
    )

    return sensitivity


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
    # load data
    if sim_type == "trunk-overlap":
        X, y, mu, cov = make_trunk_classification(
            n_samples=n_samples,
            n_dim=n_dims,
            n_informative=min(256, n_dims),
            m_factor=1,
            return_params=True,
            seed=idx,
        )
    elif sim_type == "trunk":
        X, y, mu, cov = make_trunk_classification(
            n_samples=n_samples,
            n_dim=n_dims,
            n_informative=min(256, n_dims),
            return_params=True,
            seed=idx,
        )
    X = np.float32(X)
    y = np.float32(y).reshape(-1, 1)

    for model_name in NON_OOB_MODEL_NAMES.keys():
        print(
            f"Evaluating {model_name} on {sim_type} with {n_samples} samples and {n_features} features"
        )

        # set output directory to save npz files
        output_dir = os.path.join(rootdir, f"output/{model_name}/{sim_type}/")
        os.makedirs(output_dir, exist_ok=True)

        # now compute the pvalue when shuffling all
        covariate_index = np.arange(n_features, dtype=int)

        # Get drawn indices along both sample and feature axes
        indices = np.arange(n_samples, dtype=int)
        indices_train, indices_test = train_test_split(
            indices, test_size=test_size, shuffle=True, random_state=seed
        )
        X_train = X[indices_train, :]
        y_train = y[indices_train, :]
        X_test = X[indices_test, :]
        y_test = y[indices_test, :]

        forest_params = NON_OOB_MODEL_NAMES[model_name]
        permute_per_tree = forest_params.pop("permute_per_tree", False)
        est = HonestForestClassifier(**forest_params)

        # compute test statistic
        est.fit(X_train, y_train.ravel())
        y_score = est.predict_proba_per_tree(X_test)
        observe_test_stat = sensitivity_at_specificity(
            y_test, y_score, target_specificity=0.98, pos_label=1
        )

        # compute null distribution
        null_metrics = np.zeros((n_repeats,))
        indices_train = np.arange(X_train.shape[0], dtype=int).reshape(-1, 1)
        for idx in range(n_repeats):
            rng.shuffle(indices_train)
            perm_X_cov = X_train[indices_train, covariate_index]
            X_train[:, covariate_index] = perm_X_cov

            # train a new forest on the permuted data
            # XXX: should there be a train/test split here? even w/ honest forests?
            est.fit(X_train, y_train.ravel())
            y_score = est.predict_proba_per_tree(X_test)

            # compute two instances of the metric from the sampled trees
            metric_val = sensitivity_at_specificity(
                y_test, y_score, target_specificity=0.98, pos_label=1
            )

            null_metrics[idx] = metric_val

        pvalue = (1 + (null_metrics <= observe_test_stat).sum()) / (1 + n_repeats)
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
    # load data
    if sim_type == "trunk-overlap":
        X, y, mu, cov = make_trunk_classification(
            n_samples=n_samples,
            n_dim=n_dims,
            n_informative=min(256, n_dims),
            m_factor=1,
            return_params=True,
            seed=idx,
        )
    elif sim_type == "trunk":
        X, y, mu, cov = make_trunk_classification(
            n_samples=n_samples,
            n_dim=n_dims,
            n_informative=min(256, n_dims),
            return_params=True,
            seed=idx,
        )
    X = np.float32(X)
    y = np.float32(y).reshape(-1, 1)

    for model_name in OOB_MODEL_NAMES.keys():
        print(f"Evaluating {model_name} on {sim_type} with {n_samples} samples")

        # set output directory to save npz files
        output_dir = os.path.join(rootdir, f"output/{model_name}/{sim_type}/")
        os.makedirs(output_dir, exist_ok=True)

        # now compute the pvalue when shuffling all
        covariate_index = None

        forest_params = OOB_MODEL_NAMES[model_name]
        permute_per_tree = forest_params.pop("permute_per_tree", False)

        est = HonestForestClassifier(**forest_params)
        perm_est = PermutationHonestForestClassifier(
            permute_per_tree=permute_per_tree, **forest_params
        )

        # compute pvalue
        (
            observe_test_stat,
            pvalue,
            orig_forest_proba,
            perm_forest_proba,
        ) = build_coleman_forest(
            est,
            perm_est,
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
        "n_jobs": n_jobs_trees,
        "bootstrap": False,
        "stratify": True,
        # "max_samples": ,
        "permute_per_tree": False,
    },
}

OOB_MODEL_NAMES = {
    # "might-honestfraction05-bootstrap-permuteonce": {
    #     "n_estimators": n_estimators,
    #     "random_state": seed,
    #     "honest_fraction": 0.5,
    #     "n_jobs": n_jobs_trees,
    #     "bootstrap": True,
    #     "stratify": True,
    #     "max_samples": 1.6,
    #     "permute_per_tree": False,
    # },
    "might-honestfraction05-bootstrap": {
        "n_estimators": n_estimators,
        "random_state": seed,
        "honest_fraction": 0.5,
        "n_jobs": n_jobs_trees,
        "bootstrap": True,
        "stratify": True,
        "max_samples": 1.6,
        "permute_per_tree": True,
    },
    "might-honestfraction025-bootstrap": {
        "n_estimators": n_estimators,
        "random_state": seed,
        "honest_fraction": 0.25,
        "n_jobs": n_jobs_trees,
        "bootstrap": True,
        "stratify": True,
        "max_samples": 1.6,
        "permute_per_tree": True,
    },
    "might-honestfraction075-bootstrap": {
        "n_estimators": n_estimators,
        "random_state": seed,
        "honest_fraction": 0.75,
        "n_jobs": n_jobs_trees,
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
    # sim_type = sys.argv[4]
    # rootdir = sys.argv[5]

    # _run_parallel_might_permutations(idx, n_samples, n_dims, sim_type, rootdir)

    # TODO: add root dir here
    rootdir = "./test/"

    SIM_TYPES = ["trunk", "trunk-overlap"]
    [256, 512, 1024, 2048]
    n_samples_list = [2**i for i in range(8, 12)]
    n_repeats = 100
    n_dims = 4096

    # Run the parallel job
    Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_run_parallel_might)(idx, n_samples, n_dims, sim_type, rootdir)
        for idx, n_samples, sim_type in product(
            range(n_repeats), n_samples_list, SIM_TYPES
        )
    )

    # rootdir = "./test/"

    # SIM_TYPES = ["trunk", "trunk-overlap"]
    # n_samples_list = [2**i for i in range(8, 12)]
    # n_repeats = 100
    # n_dims = 4096

    # # Run the parallel job
    # Parallel(n_jobs=1, backend="loky")(
    #     delayed(_run_parallel_might_permutations)(
    #         idx, n_samples, n_dims, sim_type, rootdir
    #     )
    #     for idx, n_samples, sim_type in product(
    #         range(n_repeats), n_samples_list, SIM_TYPES
    #     )
    # )
