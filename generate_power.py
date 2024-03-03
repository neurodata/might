import pickle
import numpy as np
import math

ALPHA = 0.05
N_ITR = 100
DIM_SIZES = [2**i for i in range(2, 13)]

with open("c_dim_512/observe_SA98_dim_512_c.pkl", "rb") as f:
    SA98_DIM = pickle.load(f)
with open("c_dim_512/null_SA98_dim_512_c.pkl", "rb") as f:
    NULL_SA98_DIM = pickle.load(f)

SA98_POWERS_DIM = []
for i in range(len(DIM_SIZES)):
    cutoff = np.sort(NULL_SA98_DIM[i])[math.ceil(N_ITR * (1 - ALPHA))]
    power = (1 + (SA98_DIM[i] >= cutoff).sum()) / (1 + N_ITR)
    SA98_POWERS_DIM.append(power)

np.savetxt("c_dim_512/overlap-might-SA98-POWER-vs-d.csv", SA98_POWERS_DIM, delimiter=",")

with open("c_dim_512/observe_MI_dim_512_c.pkl", "rb") as f:
    MI_DIM = pickle.load(f)
with open("c_dim_512/null_MI_dim_512_c.pkl", "rb") as f:
    NULL_MI_DIM = pickle.load(f)

MI_POWERS_DIM = []
for i in range(len(DIM_SIZES)):
    cutoff = np.sort(NULL_MI_DIM[i])[math.ceil(N_ITR * (1 - ALPHA))]
    power = (1 + (MI_DIM[i] >= cutoff).sum()) / (1 + N_ITR)
    MI_POWERS_DIM.append(power)

np.savetxt("c_dim_512/overlap-might-MI-POWER-vs-d.csv", MI_POWERS_DIM, delimiter=",")

with open("c_dim_512/observe_AUC_dim_512_c.pkl", "rb") as f:
    AUC_DIM = pickle.load(f)
with open("c_dim_512/null_AUC_dim_512_c.pkl", "rb") as f:
    NULL_AUC_DIM = pickle.load(f)

AUC_POWERS_DIM = []
for i in range(len(DIM_SIZES)):
    cutoff = np.sort(NULL_AUC_DIM[i])[math.ceil(N_ITR * (1 - ALPHA))]
    power = (1 + (AUC_DIM[i] >= cutoff).sum()) / (1 + N_ITR)
    AUC_POWERS_DIM.append(power)

np.savetxt("c_dim_512/overlap-might-AUC-POWER-vs-d.csv", AUC_POWERS_DIM, delimiter=",")


with open("c_dim_512/observe_Acc_dim_512_c.pkl", "rb") as f:
    Acc_DIM = pickle.load(f)
with open("c_dim_512/null_Acc_dim_512_c.pkl", "rb") as f:
    NULL_Acc_DIM = pickle.load(f)

Acc_POWERS_DIM = []
for i in range(len(DIM_SIZES)):
    cutoff = np.sort(NULL_Acc_DIM[i])[math.ceil(N_ITR * (1 - ALPHA))]
    power = (1 + (Acc_DIM[i] >= cutoff).sum()) / (1 + N_ITR)
    Acc_POWERS_DIM.append(power)

np.savetxt("c_dim_512/overlap-might-Acc-POWER-vs-d.csv", Acc_POWERS_DIM, delimiter=",")
