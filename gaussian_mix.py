# scrips of generate mixture of gaussian data and calculate the statistics
import time
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.stats import entropy
from scipy.stats import gamma
from sklearn import metrics


def Mix_Gaussian(
    mu_class0, mu_class1_1, mu_class1_2, sig, n, dim_info, dim_total, mix, seed
):
    """
    Generate a mixture of Gaussian dataset
    Args:
        mu_class0: mean of class 0
        mu_class1_1: mean of class 1 for the first component
        mu_class1_2: mean of class 1 for the second component
        sig: covariance matrix
        n: number of samples
        dim_info: number of informative features
        dim_total: total number of features
        mix: mixing ratio
        seed: random seed
    Returns:
        X: informative features
        X_all: all features
        y: labels
    """

    np.random.seed(seed)
    pdf_class0 = multivariate_normal(mean=mu_class0, cov=sig, allow_singular=True)
    pdf_class1_1 = [
        multivariate_normal(mean=mu_class1_1_i, cov=sig, allow_singular=True)
        for mu_class1_1_i in mu_class1_1
    ]
    pdf_class1_2 = [
        multivariate_normal(mean=mu_class1_2_i, cov=sig, allow_singular=True)
        for mu_class1_2_i in mu_class1_2
    ]

    n2 = [int(n * mix_i) for mix_i in mix]
    n1 = [n - n2_i for n2_i in n2]
    x_0 = [pdf_class0.rvs(size=n) for _ in range(dim_info)]
    x_1_1 = [pdf_class1_1[i].rvs(size=n1[i]) for i in range(dim_info)]
    x_1_2 = [pdf_class1_2[i].rvs(size=n2[i]) for i in range(dim_info)]
    y = np.array([0] * n + [1] * n).reshape(-1, 1)

    y = np.array([0] * n + [1] * n).reshape(-1, 1)
    X = np.zeros((2 * n, dim_info))
    for i in range(dim_info):
        X[:n, i] = x_0[i]
        X[n : (n + n1[i]), i] = x_1_1[i]
        X[(n + n1[i]) :, i] = x_1_2[i]
    X_noise = np.random.normal(0, 1, (2 * n, dim_total - dim_info))
    X_all = np.hstack((X, X_noise))

    return X, X_all, y


def Mix_Gaussian_Truth_Calculation(
    x, y, mu_class0, mu_class1_1, mu_class1_2, sig, dim_info, mix
):
    """
    Calculate the statistics of the mixture of Gaussian dataset
    Args:
        x: informative features
        y: labels
        mu_class0: mean of class 0
        mu_class1_1: mean of class 1 for the first component
        mu_class1_2: mean of class 1 for the second component
        sig: covariance matrix
        dim_info: number of informative features
        mix: mixing ratio
    Returns:
        posterior: posterior probability
        tpr_s: true positive rate at 0.02 false positive rate
    """
    pdf_class0 = multivariate_normal(mean=mu_class0, cov=sig, allow_singular=True)
    pdf_class1_1 = [
        multivariate_normal(mean=mu_class1_1_i, cov=sig, allow_singular=True)
        for mu_class1_1_i in mu_class1_1
    ]
    pdf_class1_2 = [
        multivariate_normal(mean=mu_class1_2_i, cov=sig, allow_singular=True)
        for mu_class1_2_i in mu_class1_2
    ]
    prior = [0.5, 0.5]

    p_x_given_class0 = 1
    p_x_given_class1 = 1
    for d in range(dim_info):
        X = x[:, d].reshape(-1, 1)
        p_x_given_class0 *= np.nan_to_num(pdf_class0.pdf(X))
        p_x_given_class1_1 = np.nan_to_num(pdf_class1_1[d].pdf(X))
        p_x_given_class1_2 = np.nan_to_num(pdf_class1_2[d].pdf(X))
        p_x_given_class1 *= (1 - mix[d]) * p_x_given_class1_1 + mix[
            d
        ] * p_x_given_class1_2
    p_x = prior[0] * p_x_given_class0 + prior[1] * p_x_given_class1
    pos_class0 = p_x_given_class0 * prior[0] / p_x
    pos_class1 = p_x_given_class1 * prior[1] / p_x
    posterior = np.hstack((pos_class0.reshape(-1, 1), pos_class1.reshape(-1, 1)))

    fpr, tpr, thresholds = metrics.roc_curve(
        y, posterior[:, 1], pos_label=1, drop_intermediate=False
    )
    tpr_s = np.max(tpr[fpr <= 0.02])
    return posterior, tpr_s


def Gaussian_sim_truth(n, ratio, dim_info, dim_total, seed):
    np.random.seed(seed)

    n1 = int(n * ratio)
    n0 = n - n1

    mu_0 = [0 for d in range(dim_info)]
    mu_1 = [1 / np.sqrt(d) for d in range(1, dim_info + 1)]

    sig_0 = [1 for d in range(dim_info)]
    sig_1 = [2 for d in range(dim_info)]

    pdf_class0 = [
        multivariate_normal(mean=mu_0[i], cov=sig_0[i], allow_singular=True)
        for i in range(dim_info)
    ]
    pdf_class1 = [
        multivariate_normal(mean=mu_1[i], cov=sig_1[i], allow_singular=True)
        for i in range(dim_info)
    ]

    x_0 = [pdf_class0[i].rvs(n0) for i in range(dim_info)]
    x_1 = [pdf_class1[i].rvs(n1) for i in range(dim_info)]
    y = np.array([0] * n0 + [1] * n1).reshape(-1, 1)
    X = np.zeros((n, dim_info))

    for i in range(dim_info):
        X[:n0, i] = x_0[i]
        X[n0:, i] = x_1[i]
    X_noise = np.random.normal(0, 1, (n, dim_total - dim_info))
    X_all = np.hstack((X, X_noise))

    pdf_class0 = [
        multivariate_normal(mean=mu_0[i], cov=sig_0[i], allow_singular=True)
        for i in range(dim_info)
    ]
    pdf_class1 = [
        multivariate_normal(mean=mu_1[i], cov=sig_1[i], allow_singular=True)
        for i in range(dim_info)
    ]
    prior = [1 - ratio, ratio]

    p_x_given_class0 = 1
    p_x_given_class1 = 1
    for d in range(dim_info):
        x = X[:, d].reshape(-1, 1)
        p_x_given_class0 *= np.nan_to_num(pdf_class0[d].pdf(x))
        p_x_given_class1 *= np.nan_to_num(pdf_class1[d].pdf(x))
    p_x = prior[0] * p_x_given_class0 + prior[1] * p_x_given_class1
    pos_class0 = p_x_given_class0 * prior[0] / p_x
    pos_class1 = p_x_given_class1 * prior[1] / p_x
    posterior = np.hstack((pos_class0.reshape(-1, 1), pos_class1.reshape(-1, 1)))

    fpr, tpr, thresholds = metrics.roc_curve(
        y, posterior[:, 1], pos_label=1, drop_intermediate=False
    )
    tpr_s = np.max(tpr[fpr <= 0.02])

    return X_all, y, posterior, tpr_s


def Calculate_MI(y_true, y_pred_proba):
    H_YX = np.mean(entropy(y_pred_proba, base=2, axis=1))
    # empirical count of each class (n_classes)
    _, counts = np.unique(y_true, return_counts=True)
    H_Y = entropy(counts, base=np.exp(1))
    return H_Y - H_YX
