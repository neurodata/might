import numpy as np
import pandas as pd

from scipy.stats import entropy, multivariate_normal
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve


def generate_gauss_mix(
    N=50000,
    prior=(0.5, 0.5),
    p=1,
    mu_class0=0,
    mu_class0_0=0,
    mu_class0_1=5,
    mu_class1=1,
    mu_class1_0=0,
    mu_class1_1=5,
    sig_class0=1,
    sig_class0_0=1,
    sig_class0_1=1,
    sig_class1=1,
    sig_class1_0=1,
    sig_class1_1=1,
    split_class0=None,
    split_class1=None,
):
    class0_mix = True if split_class0 is not None else False
    class1_mix = True if split_class1 is not None else False
    
    prior_0, prior_1 = prior
    p_class0, p_class1 = prior
    n0 = int(N * prior_0)  # number of samples from class 0
    n1 = N - n0  # total number of samples from class 1

    if class1_mix:
        mixture_idx = np.random.choice(
            2, N // 2, replace=True, p=split_class1
        )
        norm_params = [[mu_class1_0 * np.ones(p), np.identity(p) * sig_class1_0], [mu_class1_1 * np.ones(p), np.identity(p) * sig_class1_1]]
        x_1 = np.fromiter(
            (np.random.multivariate_normal(*(norm_params[i]), size=1) for i in mixture_idx),
            dtype=np.dtype((float, p))
        ).reshape(N // 2, p)
    else:
        # p_class0, p_class1 = prior
        mu_class1 = np.array([mu_class1] * p)
        sig_class0 = np.identity(p) * sig_class0
        sig_class1 = np.identity(p) * sig_class1
        x_1 = np.random.multivariate_normal(mu_class1, sig_class1, size=n1)

    if class0_mix:
        mixture_idx = np.random.choice(
            2, N // 2, replace=True, p=split_class0
        )
        norm_params = [[mu_class0_0 * np.ones(p), np.identity(p) * sig_class0_0], [mu_class0_1 * np.ones(p), np.identity(p) * sig_class0_1]]
        x_0 = np.fromiter(
            (np.random.multivariate_normal(*(norm_params[i]), size=1) for i in mixture_idx),
            dtype=np.dtype((float, p))
        ).reshape(N // 2, p)
    else:
        mu_class0 = np.array([mu_class0] * p)
        x_0 = np.random.multivariate_normal(mu_class0, np.identity(p) * sig_class0, size=n0)

    x = np.vstack((x_0, x_1))
    y = np.array([0] * n0 + [1] * n1).reshape(-1, 1)

    # Create the probability density functions (PDFs) for the two Gaussian distributions
    if class0_mix:
        pdf_class0_0 = multivariate_normal(mu_class1_0, sig_class1_0)
        pdf_class0_1 = multivariate_normal(mu_class1_1, sig_class1_1)
        p_x_given_class0_0 = pdf_class0_0.pdf(x)
        p_x_given_class0_1 = pdf_class0_1.pdf(x)
        p_x_given_class0 = (
            split_class1[0] * p_x_given_class0_0 + split_class0[1] * p_x_given_class0_1
        )
    else:
        pdf_class0 = multivariate_normal(mu_class0, sig_class0)
        p_x_given_class0 = pdf_class0.pdf(x)

    if class1_mix:
        pdf_class1_0 = multivariate_normal(mu_class1_0, sig_class1_0)
        pdf_class1_1 = multivariate_normal(mu_class1_1, sig_class1_1)
        p_x_given_class1_0 = pdf_class1_0.pdf(x)
        p_x_given_class1_1 = pdf_class1_1.pdf(x)
        p_x_given_class1 = (
            split_class1[0] * p_x_given_class1_0 + split_class1[1] * p_x_given_class1_1
        )
    else:
        pdf_class1 = multivariate_normal(mu_class1, sig_class1)
        p_x_given_class1 = pdf_class1.pdf(x)

    p_x = p_x_given_class0 * p_class0 + p_x_given_class1 * p_class1

    pos_class0 = p_x_given_class0 * p_class0 / p_x
    pos_class1 = p_x_given_class1 * (1 - p_class0) / p_x

    posterior = np.hstack((pos_class0.reshape(-1, 1), pos_class1.reshape(-1, 1)))
    stats_conen = np.mean(entropy(posterior, base=np.exp(1), axis=1))

    # if class0_mix:
    #     prior_y_0 = np.array([p_class0_0, p_class0_1])
    # else:
    #     prior_y_0 = np.array(p_class0)

    # if class1_mix:
    #     prior_y_1 = np.array([p_class1_0, p_class1_1])
    # else:
    #     prior_y_1 = np.array(p_class1)
        
    entropy_y = entropy(prior, base=np.exp(1))
    correlation = np.corrcoef(x_0.T, x_1.T)
    MI = entropy_y - stats_conen
    auc = roc_auc_score(y, posterior[:, 1])
    pauc_90 = roc_auc_score(y, posterior[:, 1], max_fpr=0.1)
    pauc_98 = roc_auc_score(y, posterior[:, 1], max_fpr=0.02)
    if MI == 0.0:
        # replace the posterior with random numbers ~ U(0,1) to calculate the ROC curve
        posterior_ = np.random.uniform(0, 1, size=(N, 2))
        fpr, tpr, thresholds = roc_curve(
            y, posterior_[:, 1], pos_label=1, drop_intermediate=False
        )
    else: 
        fpr, tpr, thresholds = roc_curve(
        y, posterior[:, 1], pos_label=1, drop_intermediate=False
    )
    tpr_s = np.max(tpr[fpr <= 0.02])
    y_pred = np.argmax(posterior, axis=1)
    accuracy = accuracy_score(y, y_pred)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f1 = f1_score(y, y_pred)

    statistics = {
        # 'Correlation': correlation[0,1],
        "Accuracy": accuracy,
        # "F1": f1,
        "MI": MI,
        "AUC": auc,
        # "pAUC_90": pauc_90,
        # "pAUC_98": pauc_98,
        "S@98": tpr_s,
        # 'Sensitivity': sensitivity,
        # 'Specificity': specificity,
        # 'TN': tn,
        # 'FP': fp,
        # 'FN': fn,
        # 'TP': tp
        # "tpr": tpr,
        # "fpr": fpr,
    }
    x_min, x_max = np.min(x), np.max(x)
    # print(x_min, x_max)
    xs = np.linspace(x_min - 1, x_max + 1, 1000)
    pdf = pd.DataFrame()
    pdf["x"] = xs
    if class0_mix:
        pdf["pdf_class0"] =  split_class0[0] * pdf_class0_0.pdf(xs) + split_class0[1] * pdf_class0_1.pdf(xs)
    else:
        pdf["pdf_class0"] = pdf_class0.pdf(xs)
    if class1_mix:
        pdf["pdf_class1"] =  split_class1[0] * pdf_class1_0.pdf(xs) + split_class1[1] * pdf_class1_1.pdf(xs)
    else:
        pdf["pdf_class1"] = pdf_class1.pdf(xs)

    return x, y, posterior[:, 1], statistics, pdf