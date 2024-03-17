from time import time
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sktree import RandomForestClassifier as sktreeRandomForestClassifier


seed = 12345
rng = np.random.default_rng(seed)

n_repeats = 5
n_jobs = -1
n_estimators = 100
n_samples = 2048
n_dims = 4096
X = rng.standard_normal(size=(n_samples, n_dims))
y = rng.integers(0, 2, size=(n_samples,))


for idx in range(n_repeats):
    clf = RandomForestClassifier(
        n_estimators=n_estimators, random_state=seed, n_jobs=n_jobs
    )
    tstart = time()
    clf.fit(X, y)
    fit_time = time() - tstart
    print(f"Fit time for RandomForestClassifier: {fit_time}")


for idx in range(n_repeats):
    sktree_clf = sktreeRandomForestClassifier(
        n_estimators=n_estimators, random_state=seed, n_jobs=n_jobs
    )
    tstart = time()
    sktree_clf.fit(X, y)
    fit_time = time() - tstart
    print(f"Fit time for sktreeRandomForestClassifier: {fit_time}")
