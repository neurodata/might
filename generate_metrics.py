import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sktree.ensemble import HonestForestClassifier
from sktree.stats import build_hyppo_oob_forest
from sklearn.metrics import (
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    roc_curve,
    accuracy_score,
)
from scipy.stats import multivariate_normal, entropy
from scipy.integrate import nquad


def Calculate_MI(y_true, y_pred_proba):
    H_YX = np.mean(entropy(y_pred_proba, base=np.exp(1), axis=1))
    # empirical count of each class (n_classes)
    _, counts = np.unique(y_true, return_counts=True)
    H_Y = entropy(counts, base=np.exp(1))
    return H_Y - H_YX


def Calculate_SA98(y_true, y_pred_proba, max_fpr=0.02) -> float:
    if y_true.squeeze().ndim != 1:
        raise ValueError(f"y_true must be 1d, not {y_true.shape}")
    if 0 in y_true or -1 in y_true:
        fpr, tpr, thresholds = roc_curve(
            y_true, y_pred_proba[:, 1], pos_label=1, drop_intermediate=False
        )
    else:
        fpr, tpr, thresholds = roc_curve(
            y_true, y_pred_proba[:, 1], pos_label=2, drop_intermediate=False
        )
    s98 = max([tpr for (fpr, tpr) in zip(fpr, tpr) if fpr <= max_fpr])
    return s98


def Calculate_Acc(y_true, y_pred_proba) -> float:
    return accuracy_score(y_true, y_pred_proba[:, 1] >= 0.5)


def Calculate_AUC(y_true, y_pred_proba) -> float:
    return roc_auc_score(y_true, y_pred_proba[:, 1])


DIM_SIZES = [2**i for i in range(2, 13)]
SAMPLE_SIZES = [256, 512, 1024, 2048, 4096]
N_ITR = 100

observe_dim_probas = []
null_dim_probas = []
for i in range(N_ITR):
    observe_dim_probas.append([])
    null_dim_probas.append([])
    for j in DIM_SIZES:
        with open("observe_pos_dim_a_" + str(i) + "_" + str(j) + ".pkl", "rb") as f:
            observe_dim_proba = pickle.load(f)
            observe_dim_probas[i].append(observe_dim_proba)

        with open("null_pos_dim_a_" + str(i) + "_" + str(j) + ".pkl", "rb") as f:
            null_dim_proba = pickle.load(f)
            null_dim_probas[i].append(null_dim_proba)

POS_DIM = []
NULL_POS_DIM = []
for i in range(N_ITR):
    POS_DIM.append([])
    NULL_POS_DIM.append([])
    for j in range(len(DIM_SIZES)):
        POS_DIM[i].append(np.nanmean(observe_dim_probas[i][j], axis=0))
        NULL_POS_DIM[i].append(np.nanmean(null_dim_probas[i][j], axis=0))

MI_DIM = []
NULL_MI_DIM = []
for j in range(len(DIM_SIZES)):
    y = np.concatenate((np.zeros(512 // 2), np.ones(512 // 2)))

    temp_MI_DIM = []
    temp_null_MI_DIM = []
    for i in range(N_ITR):
        temp_MI_DIM.append(Calculate_MI(y, POS_DIM[i][j]))
        temp_null_MI_DIM.append(Calculate_MI(y, NULL_POS_DIM[i][j]))

    MI_DIM.append(temp_MI_DIM)
    NULL_MI_DIM.append(temp_null_MI_DIM)

MI_DIM = np.array(MI_DIM)
NULL_MI_DIM = np.array(NULL_MI_DIM)

np.savetxt("linear-might-MI-vs-d-512.csv", MI_DIM, delimiter=",")

with open("observe_MI_dim_512_a.pkl", "wb") as f:
    pickle.dump(MI_DIM, f)
with open("null_MI_dim_512_a.pkl", "wb") as f:
    pickle.dump(NULL_MI_DIM, f)

SA98_DIM = []
NULL_SA98_DIM = []
for j in range(len(DIM_SIZES)):
    y = np.concatenate((np.zeros(512 // 2), np.ones(512 // 2)))

    temp_SA98_DIM = []
    temp_null_SA98_DIM = []
    for i in range(N_ITR):
        temp_SA98_DIM.append(Calculate_SA98(y, POS_DIM[i][j]))
        temp_null_SA98_DIM.append(Calculate_SA98(y, NULL_POS_DIM[i][j]))

    SA98_DIM.append(temp_SA98_DIM)
    NULL_SA98_DIM.append(temp_null_SA98_DIM)

SA98_DIM = np.array(SA98_DIM)
NULL_SA98_DIM = np.array(NULL_SA98_DIM)

np.savetxt("linear-might-SA98-vs-d-512.csv", SA98_DIM, delimiter=",")

with open("observe_SA98_dim_512_a.pkl", "wb") as f:
    pickle.dump(SA98_DIM, f)
with open("null_SA98_dim_512_a.pkl", "wb") as f:
    pickle.dump(NULL_SA98_DIM, f)

Acc_DIM = []
NULL_Acc_DIM = []
for j in range(len(DIM_SIZES)):
    y = np.concatenate((np.zeros(512 // 2), np.ones(512 // 2)))

    temp_Acc_DIM = []
    temp_null_Acc_DIM = []
    for i in range(N_ITR):
        temp_Acc_DIM.append(Calculate_Acc(y, POS_DIM[i][j]))
        temp_null_Acc_DIM.append(Calculate_Acc(y, NULL_POS_DIM[i][j]))

    Acc_DIM.append(temp_Acc_DIM)
    NULL_Acc_DIM.append(temp_null_Acc_DIM)

Acc_DIM = np.array(Acc_DIM)
NULL_Acc_DIM = np.array(NULL_Acc_DIM)

np.savetxt("linear-might-Acc-vs-d-512.csv", Acc_DIM, delimiter=",")

with open("observe_Acc_dim_512_a.pkl", "wb") as f:
    pickle.dump(Acc_DIM, f)
with open("null_Acc_dim_512_a.pkl", "wb") as f:
    pickle.dump(NULL_Acc_DIM, f)

AUC_DIM = []
NULL_AUC_DIM = []
for j in range(len(DIM_SIZES)):
    y = np.concatenate((np.zeros(512 // 2), np.ones(512 // 2)))

    temp_AUC_DIM = []
    temp_null_AUC_DIM = []
    for i in range(N_ITR):
        temp_AUC_DIM.append(Calculate_AUC(y, POS_DIM[i][j]))
        temp_null_AUC_DIM.append(Calculate_AUC(y, NULL_POS_DIM[i][j]))

    AUC_DIM.append(temp_AUC_DIM)
    NULL_AUC_DIM.append(temp_null_AUC_DIM)

AUC_DIM = np.array(AUC_DIM)
NULL_AUC_DIM = np.array(NULL_AUC_DIM)

np.savetxt("linear-might-AUC-vs-d-512.csv", AUC_DIM, delimiter=",")

with open("observe_AUC_dim_512_a.pkl", "wb") as f:
    pickle.dump(AUC_DIM, f)
with open("null_AUC_dim_512_a.pkl", "wb") as f:
    pickle.dump(NULL_AUC_DIM, f)
