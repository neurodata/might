import contextlib
from collections import defaultdict
import joblib
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import roc_curve
from sktree.datasets import make_trunk_classification
from sktree.ensemble import HonestForestClassifier
from sktree.stats import build_hyppo_oob_forest

N_ESTIMATORS = list(range(100, 4001, 100))
REPS = 5


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


sas98s = []
n_samples = 2048
n_dim = 4096
n_informative = 256
n_repeats = 2
results = defaultdict(list)

for seed in range(n_repeats):
    X, y = make_trunk_classification(
        n_samples=2048,
        n_dim=n_dim,
        n_informative=n_informative,
        seed=seed,
    )
    for n_estimators in N_ESTIMATORS[0:2]:
        est = HonestForestClassifier(
            n_estimators=n_estimators,
            random_state=seed,
            honest_fraction=0.5,
            n_jobs=-1,
            bootstrap=True,
            stratify=True,
            max_samples=1.6,
            # permute_per_tree=True,
        )
        _, posterior_arr = build_hyppo_oob_forest(est, X, y, verbose=False)
        sas98 = sensitivity_at_specificity(y, posterior_arr, target_specificity=0.98)
        sas98s.append(sas98)
        results['sas98'].append(sas98)
        results['n_estimators'].append(n_estimators)
        results['seed'].append(seed)

# np.save("./n_trees_exp.npy", sas98)
np.savez("./n_trees_exp.npz", results=results)

# fig, ax = plt.subplots()
# ax.hist(sas98s)
# ax.set(xlabel="S@98", ylabel="Counts", title="Histogram of S@98 for 100000 Trees")
# plt.savefig("./figs/histogram.png", bbox_inches="tight")
