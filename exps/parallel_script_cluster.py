import sys
import os
from itertools import product
import numpy as np
from joblib import delayed, Parallel
from hyppo.conditional import ConditionalDcorr
from sklearn.model_selection import StratifiedShuffleSplit
from sktree.stats import (
    FeatureImportanceForestClassifier,
)
from sktree import HonestForestClassifier
from sktree.tree import MultiViewDecisionTreeClassifier
from numpy import log
from scipy.special import digamma
from sklearn.neighbors import BallTree, KDTree


FINISHED_LIST = [
    "direct-indirect/conddcorr_64_4096_0.npz",
    "direct-indirect/conddcorr_64_4096_1.npz",
    "direct-indirect/conddcorr_64_4096_2.npz",
    "direct-indirect/conddcorr_64_4096_3.npz",
    "direct-indirect/conddcorr_64_4096_4.npz",
    "direct-indirect/conddcorr_64_4096_5.npz",
    "direct-indirect/conddcorr_64_4096_6.npz",
    "direct-indirect/conddcorr_64_4096_7.npz",
    "direct-indirect/conddcorr_64_4096_8.npz",
    "direct-indirect/conddcorr_64_4096_9.npz",
    "direct-indirect/conddcorr_64_4096_10.npz",
    "direct-indirect/conddcorr_64_4096_11.npz",
    "direct-indirect/conddcorr_64_4096_12.npz",
    "direct-indirect/conddcorr_64_4096_13.npz",
    "direct-indirect/conddcorr_64_4096_14.npz",
    "direct-indirect/conddcorr_64_4096_15.npz",
    "direct-indirect/conddcorr_64_4096_16.npz",
    "direct-indirect/conddcorr_64_4096_17.npz",
    "direct-indirect/conddcorr_64_4096_18.npz",
    "direct-indirect/conddcorr_64_4096_19.npz",
    "direct-indirect/conddcorr_64_4096_20.npz",
    "direct-indirect/conddcorr_64_4096_21.npz",
    "direct-indirect/conddcorr_64_4096_22.npz",
    "direct-indirect/conddcorr_64_4096_23.npz",
    "direct-indirect/conddcorr_64_4096_24.npz",
    "direct-indirect/conddcorr_64_4096_25.npz",
    "direct-indirect/conddcorr_64_4096_26.npz",
    "direct-indirect/conddcorr_64_4096_27.npz",
    "direct-indirect/conddcorr_64_4096_28.npz",
    "direct-indirect/conddcorr_64_4096_29.npz",
    "direct-indirect/conddcorr_64_4096_30.npz",
    "direct-indirect/conddcorr_64_4096_31.npz",
    "direct-indirect/conddcorr_64_4096_32.npz",
    "direct-indirect/conddcorr_64_4096_33.npz",
    "direct-indirect/conddcorr_64_4096_34.npz",
    "direct-indirect/conddcorr_64_4096_35.npz",
    "direct-indirect/conddcorr_64_4096_36.npz",
    "direct-indirect/conddcorr_64_4096_37.npz",
    "direct-indirect/conddcorr_64_4096_38.npz",
    "direct-indirect/conddcorr_64_4096_39.npz",
    "direct-indirect/conddcorr_64_4096_40.npz",
    "direct-indirect/conddcorr_64_4096_41.npz",
    "direct-indirect/conddcorr_64_4096_42.npz",
    "direct-indirect/conddcorr_64_4096_43.npz",
    "direct-indirect/conddcorr_64_4096_44.npz",
    "direct-indirect/conddcorr_64_4096_45.npz",
    "direct-indirect/conddcorr_64_4096_46.npz",
    "direct-indirect/conddcorr_64_4096_47.npz",
    "direct-indirect/conddcorr_64_4096_48.npz",
    "direct-indirect/conddcorr_64_4096_49.npz",
    "direct-indirect/conddcorr_64_4096_50.npz",
    "direct-indirect/conddcorr_64_4096_51.npz",
    "direct-indirect/conddcorr_64_4096_52.npz",
    "direct-indirect/conddcorr_64_4096_53.npz",
    "direct-indirect/conddcorr_64_4096_54.npz",
    "direct-indirect/conddcorr_64_4096_55.npz",
    "direct-indirect/conddcorr_64_4096_56.npz",
    "direct-indirect/conddcorr_64_4096_57.npz",
    "direct-indirect/conddcorr_64_4096_58.npz",
    "direct-indirect/conddcorr_64_4096_59.npz",
    "direct-indirect/conddcorr_64_4096_60.npz",
    "direct-indirect/conddcorr_64_4096_61.npz",
    "direct-indirect/conddcorr_64_4096_62.npz",
    "direct-indirect/conddcorr_64_4096_63.npz",
    "direct-indirect/conddcorr_64_4096_64.npz",
    "direct-indirect/conddcorr_64_4096_65.npz",
    "direct-indirect/conddcorr_64_4096_66.npz",
    "direct-indirect/conddcorr_64_4096_67.npz",
    "direct-indirect/conddcorr_64_4096_68.npz",
    "direct-indirect/conddcorr_64_4096_69.npz",
    "direct-indirect/conddcorr_64_4096_70.npz",
    "direct-indirect/conddcorr_64_4096_71.npz",
    "direct-indirect/conddcorr_64_4096_72.npz",
    "direct-indirect/conddcorr_64_4096_73.npz",
    "direct-indirect/conddcorr_64_4096_74.npz",
    "direct-indirect/conddcorr_64_4096_75.npz",
    "direct-indirect/conddcorr_64_4096_76.npz",
    "direct-indirect/conddcorr_64_4096_77.npz",
    "direct-indirect/conddcorr_64_4096_78.npz",
    "direct-indirect/conddcorr_64_4096_79.npz",
    "direct-indirect/conddcorr_64_4096_80.npz",
    "direct-indirect/conddcorr_64_4096_81.npz",
    "direct-indirect/conddcorr_64_4096_82.npz",
    "direct-indirect/conddcorr_64_4096_83.npz",
    "direct-indirect/conddcorr_64_4096_84.npz",
    "direct-indirect/conddcorr_64_4096_85.npz",
    "direct-indirect/conddcorr_64_4096_86.npz",
    "direct-indirect/conddcorr_64_4096_87.npz",
    "direct-indirect/conddcorr_64_4096_88.npz",
    "direct-indirect/conddcorr_64_4096_89.npz",
    "direct-indirect/conddcorr_64_4096_90.npz",
    "direct-indirect/conddcorr_64_4096_91.npz",
    "direct-indirect/conddcorr_64_4096_92.npz",
    "direct-indirect/conddcorr_64_4096_93.npz",
    "direct-indirect/conddcorr_64_4096_94.npz",
    "direct-indirect/conddcorr_64_4096_95.npz",
    "direct-indirect/conddcorr_64_4096_96.npz",
    "direct-indirect/conddcorr_64_4096_97.npz",
    "direct-indirect/conddcorr_64_4096_98.npz",
    "direct-indirect/conddcorr_64_4096_99.npz",
    "direct-indirect/conddcorr_128_4096_0.npz",
    "direct-indirect/conddcorr_128_4096_1.npz",
    "direct-indirect/conddcorr_128_4096_2.npz",
    "direct-indirect/conddcorr_128_4096_3.npz",
    "direct-indirect/conddcorr_128_4096_4.npz",
    "direct-indirect/conddcorr_128_4096_5.npz",
    "direct-indirect/conddcorr_128_4096_6.npz",
    "direct-indirect/conddcorr_128_4096_7.npz",
    "direct-indirect/conddcorr_128_4096_8.npz",
    "direct-indirect/conddcorr_128_4096_9.npz",
    "direct-indirect/conddcorr_128_4096_10.npz",
    "direct-indirect/conddcorr_128_4096_11.npz",
    "direct-indirect/conddcorr_128_4096_12.npz",
    "direct-indirect/conddcorr_128_4096_13.npz",
    "direct-indirect/conddcorr_128_4096_14.npz",
    "direct-indirect/conddcorr_128_4096_15.npz",
    "direct-indirect/conddcorr_128_4096_16.npz",
    "direct-indirect/conddcorr_128_4096_17.npz",
    "direct-indirect/conddcorr_128_4096_18.npz",
    "direct-indirect/conddcorr_128_4096_19.npz",
    "direct-indirect/conddcorr_128_4096_20.npz",
    "direct-indirect/conddcorr_128_4096_21.npz",
    "direct-indirect/conddcorr_128_4096_22.npz",
    "direct-indirect/conddcorr_128_4096_23.npz",
    "direct-indirect/conddcorr_128_4096_24.npz",
    "direct-indirect/conddcorr_128_4096_25.npz",
    "direct-indirect/conddcorr_128_4096_26.npz",
    "direct-indirect/conddcorr_128_4096_27.npz",
    "direct-indirect/conddcorr_128_4096_28.npz",
    "direct-indirect/conddcorr_128_4096_29.npz",
    "direct-indirect/conddcorr_128_4096_30.npz",
    "direct-indirect/conddcorr_128_4096_31.npz",
    "direct-indirect/conddcorr_128_4096_32.npz",
    "direct-indirect/conddcorr_128_4096_33.npz",
    "direct-indirect/conddcorr_128_4096_34.npz",
    "direct-indirect/conddcorr_128_4096_35.npz",
    "direct-indirect/conddcorr_128_4096_36.npz",
    "direct-indirect/conddcorr_128_4096_37.npz",
    "direct-indirect/conddcorr_128_4096_38.npz",
    "direct-indirect/conddcorr_128_4096_39.npz",
    "direct-indirect/conddcorr_128_4096_40.npz",
    "direct-indirect/conddcorr_128_4096_41.npz",
    "direct-indirect/conddcorr_128_4096_42.npz",
    "direct-indirect/conddcorr_128_4096_43.npz",
    "direct-indirect/conddcorr_128_4096_44.npz",
    "direct-indirect/conddcorr_128_4096_45.npz",
    "direct-indirect/conddcorr_128_4096_46.npz",
    "direct-indirect/conddcorr_128_4096_47.npz",
    "direct-indirect/conddcorr_128_4096_48.npz",
    "direct-indirect/conddcorr_128_4096_49.npz",
    "direct-indirect/conddcorr_128_4096_50.npz",
    "direct-indirect/conddcorr_128_4096_51.npz",
    "direct-indirect/conddcorr_128_4096_52.npz",
    "direct-indirect/conddcorr_128_4096_53.npz",
    "direct-indirect/conddcorr_128_4096_54.npz",
    "direct-indirect/conddcorr_128_4096_55.npz",
    "direct-indirect/conddcorr_128_4096_56.npz",
    "direct-indirect/conddcorr_128_4096_57.npz",
    "direct-indirect/conddcorr_128_4096_58.npz",
    "direct-indirect/conddcorr_128_4096_59.npz",
    "direct-indirect/conddcorr_128_4096_60.npz",
    "direct-indirect/conddcorr_128_4096_61.npz",
    "direct-indirect/conddcorr_128_4096_62.npz",
    "direct-indirect/conddcorr_128_4096_63.npz",
    "direct-indirect/conddcorr_128_4096_64.npz",
    "direct-indirect/conddcorr_128_4096_65.npz",
    "direct-indirect/conddcorr_128_4096_66.npz",
    "direct-indirect/conddcorr_128_4096_67.npz",
    "direct-indirect/conddcorr_128_4096_68.npz",
    "direct-indirect/conddcorr_128_4096_69.npz",
    "direct-indirect/conddcorr_128_4096_70.npz",
    "direct-indirect/conddcorr_128_4096_71.npz",
    "direct-indirect/conddcorr_128_4096_72.npz",
    "direct-indirect/conddcorr_128_4096_73.npz",
    "direct-indirect/conddcorr_128_4096_74.npz",
    "direct-indirect/conddcorr_128_4096_75.npz",
    "direct-indirect/conddcorr_128_4096_76.npz",
    "direct-indirect/conddcorr_128_4096_77.npz",
    "direct-indirect/conddcorr_128_4096_78.npz",
    "direct-indirect/conddcorr_128_4096_80.npz",
    "direct-indirect/conddcorr_128_4096_82.npz",
    "direct-indirect/conddcorr_128_4096_83.npz",
    "direct-indirect/conddcorr_128_4096_84.npz",
    "direct-indirect/conddcorr_128_4096_85.npz",
    "direct-indirect/conddcorr_128_4096_86.npz",
    "direct-indirect/conddcorr_128_4096_87.npz",
    "direct-indirect/conddcorr_128_4096_88.npz",
    "direct-indirect/conddcorr_128_4096_89.npz",
    "direct-indirect/conddcorr_128_4096_90.npz",
    "direct-indirect/conddcorr_128_4096_91.npz",
    "direct-indirect/conddcorr_128_4096_92.npz",
    "direct-indirect/conddcorr_128_4096_93.npz",
    "direct-indirect/conddcorr_128_4096_94.npz",
    "direct-indirect/conddcorr_128_4096_95.npz",
    "direct-indirect/conddcorr_128_4096_96.npz",
    "direct-indirect/conddcorr_128_4096_97.npz",
    "direct-indirect/conddcorr_128_4096_98.npz",
    "direct-indirect/conddcorr_128_4096_99.npz",
    "direct-indirect/conddcorr_256_4096_0.npz",
    "direct-indirect/conddcorr_256_4096_1.npz",
    "direct-indirect/conddcorr_256_4096_2.npz",
    "direct-indirect/conddcorr_256_4096_3.npz",
    "direct-indirect/conddcorr_256_4096_4.npz",
    "direct-indirect/conddcorr_256_4096_5.npz",
    "direct-indirect/conddcorr_256_4096_6.npz",
    "direct-indirect/conddcorr_256_4096_7.npz",
    "direct-indirect/conddcorr_256_4096_8.npz",
    "direct-indirect/conddcorr_256_4096_9.npz",
    "direct-indirect/conddcorr_256_4096_10.npz",
    "direct-indirect/conddcorr_256_4096_11.npz",
    "direct-indirect/conddcorr_256_4096_12.npz",
    "direct-indirect/conddcorr_256_4096_13.npz",
    "direct-indirect/conddcorr_256_4096_14.npz",
    "direct-indirect/conddcorr_256_4096_15.npz",
    "direct-indirect/conddcorr_256_4096_16.npz",
    "direct-indirect/conddcorr_256_4096_17.npz",
    "direct-indirect/conddcorr_256_4096_18.npz",
    "direct-indirect/conddcorr_256_4096_19.npz",
    "direct-indirect/conddcorr_256_4096_20.npz",
    "direct-indirect/conddcorr_256_4096_21.npz",
    "direct-indirect/conddcorr_256_4096_22.npz",
    "direct-indirect/conddcorr_256_4096_23.npz",
    "direct-indirect/conddcorr_256_4096_24.npz",
    "direct-indirect/conddcorr_256_4096_25.npz",
    "direct-indirect/conddcorr_256_4096_26.npz",
    "direct-indirect/conddcorr_256_4096_27.npz",
    "direct-indirect/conddcorr_256_4096_28.npz",
    "direct-indirect/conddcorr_256_4096_29.npz",
    "direct-indirect/conddcorr_256_4096_30.npz",
    "direct-indirect/conddcorr_256_4096_31.npz",
    "direct-indirect/conddcorr_256_4096_32.npz",
    "direct-indirect/conddcorr_256_4096_33.npz",
    "direct-indirect/conddcorr_256_4096_34.npz",
    "direct-indirect/conddcorr_256_4096_35.npz",
    "direct-indirect/conddcorr_256_4096_36.npz",
    "direct-indirect/conddcorr_256_4096_37.npz",
    "direct-indirect/conddcorr_256_4096_38.npz",
    "direct-indirect/conddcorr_256_4096_39.npz",
    "direct-indirect/conddcorr_256_4096_40.npz",
    "direct-indirect/conddcorr_256_4096_41.npz",
    "direct-indirect/conddcorr_256_4096_42.npz",
    "direct-indirect/conddcorr_256_4096_43.npz",
    "direct-indirect/conddcorr_256_4096_44.npz",
    "direct-indirect/conddcorr_256_4096_45.npz",
    "direct-indirect/conddcorr_256_4096_46.npz",
    "direct-indirect/conddcorr_256_4096_47.npz",
    "direct-indirect/conddcorr_256_4096_48.npz",
    "direct-indirect/conddcorr_256_4096_49.npz",
    "direct-indirect/conddcorr_256_4096_50.npz",
    "direct-indirect/conddcorr_256_4096_51.npz",
    "direct-indirect/conddcorr_256_4096_52.npz",
    "direct-indirect/conddcorr_256_4096_53.npz",
    "direct-indirect/conddcorr_256_4096_54.npz",
    "direct-indirect/conddcorr_256_4096_55.npz",
    "direct-indirect/conddcorr_256_4096_56.npz",
    "direct-indirect/conddcorr_256_4096_57.npz",
    "direct-indirect/conddcorr_256_4096_58.npz",
    "direct-indirect/conddcorr_256_4096_59.npz",
    "direct-indirect/conddcorr_256_4096_60.npz",
    "direct-indirect/conddcorr_256_4096_61.npz",
    "direct-indirect/conddcorr_256_4096_62.npz",
    "direct-indirect/conddcorr_256_4096_63.npz",
    "direct-indirect/conddcorr_256_4096_64.npz",
    "direct-indirect/conddcorr_256_4096_65.npz",
    "direct-indirect/conddcorr_256_4096_66.npz",
    "direct-indirect/conddcorr_256_4096_67.npz",
    "direct-indirect/conddcorr_256_4096_68.npz",
    "direct-indirect/conddcorr_256_4096_69.npz",
    "direct-indirect/conddcorr_256_4096_70.npz",
    "direct-indirect/conddcorr_256_4096_71.npz",
    "direct-indirect/conddcorr_256_4096_72.npz",
    "direct-indirect/conddcorr_256_4096_73.npz",
    "direct-indirect/conddcorr_256_4096_74.npz",
    "direct-indirect/conddcorr_256_4096_75.npz",
    "direct-indirect/conddcorr_256_4096_76.npz",
    "direct-indirect/conddcorr_256_4096_77.npz",
    "direct-indirect/conddcorr_256_4096_78.npz",
    "direct-indirect/conddcorr_256_4096_79.npz",
    "direct-indirect/conddcorr_256_4096_80.npz",
    "direct-indirect/conddcorr_256_4096_81.npz",
    "direct-indirect/conddcorr_256_4096_82.npz",
    "direct-indirect/conddcorr_256_4096_83.npz",
    "direct-indirect/conddcorr_256_4096_84.npz",
    "direct-indirect/conddcorr_256_4096_85.npz",
    "direct-indirect/conddcorr_256_4096_86.npz",
    "direct-indirect/conddcorr_256_4096_87.npz",
    "direct-indirect/conddcorr_256_4096_88.npz",
    "direct-indirect/conddcorr_256_4096_89.npz",
    "direct-indirect/conddcorr_256_4096_90.npz",
    "direct-indirect/conddcorr_256_4096_91.npz",
    "direct-indirect/conddcorr_256_4096_92.npz",
    "direct-indirect/conddcorr_256_4096_93.npz",
    "direct-indirect/conddcorr_256_4096_94.npz",
    "direct-indirect/conddcorr_256_4096_95.npz",
    "direct-indirect/conddcorr_256_4096_96.npz",
    "direct-indirect/conddcorr_256_4096_97.npz",
    "direct-indirect/conddcorr_256_4096_98.npz",
    "direct-indirect/conddcorr_256_4096_99.npz",
    "direct-indirect/conddcorr_512_4096_0.npz",
    "direct-indirect/conddcorr_512_4096_1.npz",
    "direct-indirect/conddcorr_512_4096_2.npz",
    "direct-indirect/conddcorr_512_4096_3.npz",
    "direct-indirect/conddcorr_512_4096_4.npz",
    "direct-indirect/conddcorr_512_4096_5.npz",
    "direct-indirect/conddcorr_512_4096_6.npz",
    "direct-indirect/conddcorr_512_4096_7.npz",
    "direct-indirect/conddcorr_512_4096_8.npz",
    "direct-indirect/conddcorr_512_4096_9.npz",
    "direct-indirect/conddcorr_512_4096_10.npz",
    "direct-indirect/conddcorr_512_4096_11.npz",
    "direct-indirect/conddcorr_512_4096_12.npz",
    "direct-indirect/conddcorr_512_4096_13.npz",
    "direct-indirect/conddcorr_512_4096_14.npz",
    "direct-indirect/conddcorr_512_4096_15.npz",
    "direct-indirect/conddcorr_512_4096_16.npz",
    "direct-indirect/conddcorr_512_4096_17.npz",
    "direct-indirect/conddcorr_512_4096_18.npz",
    "direct-indirect/conddcorr_512_4096_19.npz",
    "direct-indirect/conddcorr_512_4096_20.npz",
    "direct-indirect/conddcorr_512_4096_21.npz",
    "direct-indirect/conddcorr_512_4096_22.npz",
    "direct-indirect/conddcorr_512_4096_23.npz",
    "direct-indirect/conddcorr_512_4096_24.npz",
    "direct-indirect/conddcorr_512_4096_25.npz",
    "direct-indirect/conddcorr_512_4096_26.npz",
    "direct-indirect/conddcorr_512_4096_27.npz",
    "direct-indirect/conddcorr_512_4096_28.npz",
    "direct-indirect/conddcorr_512_4096_29.npz",
    "direct-indirect/conddcorr_512_4096_30.npz",
    "direct-indirect/conddcorr_512_4096_31.npz",
    "direct-indirect/conddcorr_512_4096_32.npz",
    "direct-indirect/conddcorr_512_4096_33.npz",
    "direct-indirect/conddcorr_512_4096_34.npz",
    "direct-indirect/conddcorr_512_4096_35.npz",
    "direct-indirect/conddcorr_512_4096_36.npz",
    "direct-indirect/conddcorr_512_4096_37.npz",
    "direct-indirect/conddcorr_512_4096_38.npz",
    "direct-indirect/conddcorr_512_4096_39.npz",
    "direct-indirect/conddcorr_512_4096_40.npz",
    "direct-indirect/conddcorr_512_4096_41.npz",
    "direct-indirect/conddcorr_512_4096_42.npz",
    "direct-indirect/conddcorr_512_4096_43.npz",
    "direct-indirect/conddcorr_512_4096_44.npz",
    "direct-indirect/conddcorr_512_4096_45.npz",
    "direct-indirect/conddcorr_512_4096_46.npz",
    "direct-indirect/conddcorr_512_4096_47.npz",
    "direct-indirect/conddcorr_512_4096_48.npz",
    "direct-indirect/conddcorr_512_4096_49.npz",
    "direct-indirect/conddcorr_512_4096_50.npz",
    "direct-indirect/conddcorr_512_4096_51.npz",
    "direct-indirect/conddcorr_512_4096_52.npz",
    "direct-indirect/conddcorr_512_4096_53.npz",
    "direct-indirect/conddcorr_512_4096_54.npz",
    "direct-indirect/conddcorr_512_4096_55.npz",
    "direct-indirect/conddcorr_512_4096_56.npz",
    "direct-indirect/conddcorr_512_4096_57.npz",
    "direct-indirect/conddcorr_512_4096_58.npz",
    "direct-indirect/conddcorr_512_4096_59.npz",
    "direct-indirect/conddcorr_512_4096_60.npz",
    "direct-indirect/conddcorr_512_4096_61.npz",
    "direct-indirect/conddcorr_512_4096_62.npz",
    "direct-indirect/conddcorr_512_4096_63.npz",
    "direct-indirect/conddcorr_512_4096_64.npz",
    "direct-indirect/conddcorr_512_4096_65.npz",
    "direct-indirect/conddcorr_512_4096_66.npz",
    "direct-indirect/conddcorr_512_4096_67.npz",
    "direct-indirect/conddcorr_512_4096_68.npz",
    "direct-indirect/conddcorr_512_4096_69.npz",
    "direct-indirect/conddcorr_512_4096_70.npz",
    "direct-indirect/conddcorr_512_4096_71.npz",
    "direct-indirect/conddcorr_512_4096_72.npz",
    "direct-indirect/conddcorr_512_4096_73.npz",
    "direct-indirect/conddcorr_512_4096_74.npz",
    "direct-indirect/conddcorr_512_4096_75.npz",
    "direct-indirect/conddcorr_512_4096_76.npz",
    "direct-indirect/conddcorr_512_4096_77.npz",
    "direct-indirect/conddcorr_512_4096_78.npz",
    "direct-indirect/conddcorr_512_4096_79.npz",
    "direct-indirect/conddcorr_512_4096_80.npz",
    "direct-indirect/conddcorr_512_4096_81.npz",
    "direct-indirect/conddcorr_512_4096_82.npz",
    "direct-indirect/conddcorr_512_4096_83.npz",
    "direct-indirect/conddcorr_512_4096_84.npz",
    "direct-indirect/conddcorr_512_4096_85.npz",
    "direct-indirect/conddcorr_512_4096_86.npz",
    "direct-indirect/conddcorr_512_4096_87.npz",
    "direct-indirect/conddcorr_512_4096_88.npz",
    "direct-indirect/conddcorr_512_4096_89.npz",
    "direct-indirect/conddcorr_512_4096_90.npz",
    "direct-indirect/conddcorr_512_4096_91.npz",
    "direct-indirect/conddcorr_512_4096_92.npz",
    "direct-indirect/conddcorr_512_4096_93.npz",
    "direct-indirect/conddcorr_512_4096_94.npz",
    "direct-indirect/conddcorr_512_4096_95.npz",
    "direct-indirect/conddcorr_512_4096_96.npz",
    "direct-indirect/conddcorr_512_4096_97.npz",
    "direct-indirect/conddcorr_512_4096_98.npz",
    "direct-indirect/conddcorr_512_4096_99.npz",
    "direct-indirect/conddcorr_1024_4096_0.npz",
    "direct-indirect/conddcorr_1024_4096_1.npz",
    "direct-indirect/conddcorr_1024_4096_2.npz",
    "direct-indirect/conddcorr_1024_4096_3.npz",
    "direct-indirect/conddcorr_1024_4096_4.npz",
    "direct-indirect/conddcorr_1024_4096_5.npz",
    "direct-indirect/conddcorr_1024_4096_6.npz",
    "direct-indirect/conddcorr_1024_4096_7.npz",
    "direct-indirect/conddcorr_1024_4096_8.npz",
    "direct-indirect/conddcorr_1024_4096_9.npz",
    "direct-indirect/conddcorr_1024_4096_10.npz",
    "direct-indirect/conddcorr_1024_4096_11.npz",
    "direct-indirect/conddcorr_1024_4096_12.npz",
    "direct-indirect/conddcorr_1024_4096_13.npz",
    "direct-indirect/conddcorr_1024_4096_14.npz",
    "direct-indirect/conddcorr_1024_4096_15.npz",
    "direct-indirect/conddcorr_1024_4096_16.npz",
    "direct-indirect/conddcorr_1024_4096_17.npz",
    "direct-indirect/conddcorr_1024_4096_18.npz",
    "direct-indirect/conddcorr_1024_4096_19.npz",
    "direct-indirect/conddcorr_1024_4096_20.npz",
    "direct-indirect/conddcorr_1024_4096_21.npz",
    "direct-indirect/conddcorr_1024_4096_22.npz",
    "direct-indirect/conddcorr_1024_4096_23.npz",
    "direct-indirect/conddcorr_1024_4096_24.npz",
    "direct-indirect/conddcorr_1024_4096_25.npz",
    "direct-indirect/conddcorr_1024_4096_26.npz",
    "direct-indirect/conddcorr_1024_4096_27.npz",
    "direct-indirect/conddcorr_1024_4096_28.npz",
    "direct-indirect/conddcorr_1024_4096_29.npz",
    "direct-indirect/conddcorr_1024_4096_30.npz",
    "direct-indirect/conddcorr_1024_4096_31.npz",
    "direct-indirect/conddcorr_1024_4096_32.npz",
    "direct-indirect/conddcorr_1024_4096_33.npz",
    "direct-indirect/conddcorr_1024_4096_34.npz",
    "direct-indirect/conddcorr_1024_4096_35.npz",
    "direct-indirect/conddcorr_1024_4096_36.npz",
    "direct-indirect/conddcorr_1024_4096_37.npz",
    "direct-indirect/conddcorr_1024_4096_38.npz",
    "direct-indirect/conddcorr_1024_4096_39.npz",
    "direct-indirect/conddcorr_1024_4096_40.npz",
    "direct-indirect/conddcorr_1024_4096_41.npz",
    "direct-indirect/conddcorr_1024_4096_42.npz",
    "direct-indirect/conddcorr_1024_4096_43.npz",
    "direct-indirect/conddcorr_1024_4096_44.npz",
    "direct-indirect/conddcorr_1024_4096_45.npz",
    "direct-indirect/conddcorr_1024_4096_46.npz",
    "direct-indirect/conddcorr_1024_4096_47.npz",
    "direct-indirect/conddcorr_1024_4096_48.npz",
    "direct-indirect/conddcorr_1024_4096_49.npz",
    "direct-indirect/conddcorr_1024_4096_50.npz",
    "direct-indirect/conddcorr_1024_4096_51.npz",
    "direct-indirect/conddcorr_1024_4096_52.npz",
    "direct-indirect/conddcorr_1024_4096_53.npz",
    "direct-indirect/conddcorr_1024_4096_54.npz",
    "direct-indirect/conddcorr_1024_4096_55.npz",
    "direct-indirect/conddcorr_1024_4096_56.npz",
    "direct-indirect/conddcorr_1024_4096_57.npz",
    "direct-indirect/conddcorr_1024_4096_58.npz",
    "direct-indirect/conddcorr_1024_4096_59.npz",
    "direct-indirect/conddcorr_1024_4096_60.npz",
    "direct-indirect/conddcorr_1024_4096_61.npz",
    "direct-indirect/conddcorr_1024_4096_63.npz",
    "direct-indirect/conddcorr_1024_4096_64.npz",
    "direct-indirect/conddcorr_1024_4096_65.npz",
    "direct-indirect/conddcorr_1024_4096_66.npz",
    "direct-indirect/conddcorr_1024_4096_68.npz",
    "independent/conddcorr_64_4096_0.npz",
    "independent/conddcorr_64_4096_1.npz",
    "independent/conddcorr_64_4096_2.npz",
    "independent/conddcorr_64_4096_3.npz",
    "independent/conddcorr_64_4096_4.npz",
    "independent/conddcorr_64_4096_5.npz",
    "independent/conddcorr_64_4096_6.npz",
    "independent/conddcorr_64_4096_7.npz",
    "independent/conddcorr_64_4096_8.npz",
    "independent/conddcorr_64_4096_9.npz",
    "independent/conddcorr_64_4096_10.npz",
    "independent/conddcorr_64_4096_11.npz",
    "independent/conddcorr_64_4096_12.npz",
    "independent/conddcorr_64_4096_13.npz",
    "independent/conddcorr_64_4096_14.npz",
    "independent/conddcorr_64_4096_15.npz",
    "independent/conddcorr_64_4096_16.npz",
    "independent/conddcorr_64_4096_17.npz",
    "independent/conddcorr_64_4096_18.npz",
    "independent/conddcorr_64_4096_19.npz",
    "independent/conddcorr_64_4096_20.npz",
    "independent/conddcorr_64_4096_21.npz",
    "independent/conddcorr_64_4096_22.npz",
    "independent/conddcorr_64_4096_23.npz",
    "independent/conddcorr_64_4096_24.npz",
    "independent/conddcorr_64_4096_25.npz",
    "independent/conddcorr_64_4096_26.npz",
    "independent/conddcorr_64_4096_27.npz",
    "independent/conddcorr_64_4096_28.npz",
    "independent/conddcorr_64_4096_29.npz",
    "independent/conddcorr_64_4096_30.npz",
    "independent/conddcorr_64_4096_31.npz",
    "independent/conddcorr_64_4096_32.npz",
    "independent/conddcorr_64_4096_33.npz",
    "independent/conddcorr_64_4096_34.npz",
    "independent/conddcorr_64_4096_35.npz",
    "independent/conddcorr_64_4096_36.npz",
    "independent/conddcorr_64_4096_37.npz",
    "independent/conddcorr_64_4096_38.npz",
    "independent/conddcorr_64_4096_39.npz",
    "independent/conddcorr_64_4096_40.npz",
    "independent/conddcorr_64_4096_41.npz",
    "independent/conddcorr_64_4096_42.npz",
    "independent/conddcorr_64_4096_43.npz",
    "independent/conddcorr_64_4096_44.npz",
    "independent/conddcorr_64_4096_45.npz",
    "independent/conddcorr_64_4096_46.npz",
    "independent/conddcorr_64_4096_47.npz",
    "independent/conddcorr_64_4096_48.npz",
    "independent/conddcorr_64_4096_49.npz",
    "independent/conddcorr_64_4096_50.npz",
    "independent/conddcorr_64_4096_51.npz",
    "independent/conddcorr_64_4096_52.npz",
    "independent/conddcorr_64_4096_53.npz",
    "independent/conddcorr_64_4096_54.npz",
    "independent/conddcorr_64_4096_55.npz",
    "independent/conddcorr_64_4096_56.npz",
    "independent/conddcorr_64_4096_57.npz",
    "independent/conddcorr_64_4096_58.npz",
    "independent/conddcorr_64_4096_59.npz",
    "independent/conddcorr_64_4096_60.npz",
    "independent/conddcorr_64_4096_61.npz",
    "independent/conddcorr_64_4096_62.npz",
    "independent/conddcorr_64_4096_63.npz",
    "independent/conddcorr_64_4096_64.npz",
    "independent/conddcorr_64_4096_65.npz",
    "independent/conddcorr_64_4096_66.npz",
    "independent/conddcorr_64_4096_67.npz",
    "independent/conddcorr_64_4096_68.npz",
    "independent/conddcorr_64_4096_69.npz",
    "independent/conddcorr_64_4096_70.npz",
    "independent/conddcorr_64_4096_71.npz",
    "independent/conddcorr_64_4096_72.npz",
    "independent/conddcorr_64_4096_73.npz",
    "independent/conddcorr_64_4096_74.npz",
    "independent/conddcorr_64_4096_75.npz",
    "independent/conddcorr_64_4096_76.npz",
    "independent/conddcorr_64_4096_77.npz",
    "independent/conddcorr_64_4096_78.npz",
    "independent/conddcorr_64_4096_79.npz",
    "independent/conddcorr_64_4096_80.npz",
    "independent/conddcorr_64_4096_82.npz",
    "independent/conddcorr_64_4096_83.npz",
    "independent/conddcorr_64_4096_84.npz",
    "independent/conddcorr_64_4096_85.npz",
    "independent/conddcorr_64_4096_86.npz",
    "independent/conddcorr_64_4096_87.npz",
    "independent/conddcorr_64_4096_88.npz",
    "independent/conddcorr_64_4096_89.npz",
    "independent/conddcorr_64_4096_90.npz",
    "independent/conddcorr_64_4096_91.npz",
    "independent/conddcorr_64_4096_92.npz",
    "independent/conddcorr_64_4096_93.npz",
    "independent/conddcorr_64_4096_94.npz",
    "independent/conddcorr_64_4096_95.npz",
    "independent/conddcorr_64_4096_96.npz",
    "independent/conddcorr_64_4096_97.npz",
    "independent/conddcorr_64_4096_98.npz",
    "independent/conddcorr_64_4096_99.npz",
    "independent/conddcorr_128_4096_0.npz",
    "independent/conddcorr_128_4096_1.npz",
    "independent/conddcorr_128_4096_2.npz",
    "independent/conddcorr_128_4096_3.npz",
    "independent/conddcorr_128_4096_4.npz",
    "independent/conddcorr_128_4096_5.npz",
    "independent/conddcorr_128_4096_6.npz",
    "independent/conddcorr_128_4096_7.npz",
    "independent/conddcorr_128_4096_8.npz",
    "independent/conddcorr_128_4096_9.npz",
    "independent/conddcorr_128_4096_10.npz",
    "independent/conddcorr_128_4096_11.npz",
    "independent/conddcorr_128_4096_12.npz",
    "independent/conddcorr_128_4096_13.npz",
    "independent/conddcorr_128_4096_14.npz",
    "independent/conddcorr_128_4096_15.npz",
    "independent/conddcorr_128_4096_16.npz",
    "independent/conddcorr_128_4096_17.npz",
    "independent/conddcorr_128_4096_18.npz",
    "independent/conddcorr_128_4096_19.npz",
    "independent/conddcorr_128_4096_20.npz",
    "independent/conddcorr_128_4096_21.npz",
    "independent/conddcorr_128_4096_22.npz",
    "independent/conddcorr_128_4096_23.npz",
    "independent/conddcorr_128_4096_24.npz",
    "independent/conddcorr_128_4096_25.npz",
    "independent/conddcorr_128_4096_26.npz",
    "independent/conddcorr_128_4096_27.npz",
    "independent/conddcorr_128_4096_28.npz",
    "independent/conddcorr_128_4096_29.npz",
    "independent/conddcorr_128_4096_30.npz",
    "independent/conddcorr_128_4096_31.npz",
    "independent/conddcorr_128_4096_32.npz",
    "independent/conddcorr_128_4096_33.npz",
    "independent/conddcorr_128_4096_34.npz",
    "independent/conddcorr_128_4096_35.npz",
    "independent/conddcorr_128_4096_36.npz",
    "independent/conddcorr_128_4096_37.npz",
    "independent/conddcorr_128_4096_38.npz",
    "independent/conddcorr_128_4096_39.npz",
    "independent/conddcorr_128_4096_40.npz",
    "independent/conddcorr_128_4096_41.npz",
    "independent/conddcorr_128_4096_42.npz",
    "independent/conddcorr_128_4096_43.npz",
    "independent/conddcorr_128_4096_44.npz",
    "independent/conddcorr_128_4096_45.npz",
    "independent/conddcorr_128_4096_46.npz",
    "independent/conddcorr_128_4096_47.npz",
    "independent/conddcorr_128_4096_48.npz",
    "independent/conddcorr_128_4096_49.npz",
    "independent/conddcorr_128_4096_50.npz",
    "independent/conddcorr_128_4096_51.npz",
    "independent/conddcorr_128_4096_52.npz",
    "independent/conddcorr_128_4096_53.npz",
    "independent/conddcorr_128_4096_54.npz",
    "independent/conddcorr_128_4096_55.npz",
    "independent/conddcorr_128_4096_56.npz",
    "independent/conddcorr_128_4096_57.npz",
    "independent/conddcorr_128_4096_58.npz",
    "independent/conddcorr_128_4096_59.npz",
    "independent/conddcorr_128_4096_60.npz",
    "independent/conddcorr_128_4096_61.npz",
    "independent/conddcorr_128_4096_62.npz",
    "independent/conddcorr_128_4096_63.npz",
    "independent/conddcorr_128_4096_64.npz",
    "independent/conddcorr_128_4096_65.npz",
    "independent/conddcorr_128_4096_66.npz",
    "independent/conddcorr_128_4096_67.npz",
    "independent/conddcorr_128_4096_68.npz",
    "independent/conddcorr_128_4096_69.npz",
    "independent/conddcorr_128_4096_70.npz",
    "independent/conddcorr_128_4096_71.npz",
    "independent/conddcorr_128_4096_72.npz",
    "independent/conddcorr_128_4096_73.npz",
    "independent/conddcorr_128_4096_74.npz",
    "independent/conddcorr_128_4096_75.npz",
    "independent/conddcorr_128_4096_76.npz",
    "independent/conddcorr_128_4096_77.npz",
    "independent/conddcorr_128_4096_78.npz",
    "independent/conddcorr_128_4096_79.npz",
    "independent/conddcorr_128_4096_80.npz",
    "independent/conddcorr_128_4096_81.npz",
    "independent/conddcorr_128_4096_82.npz",
    "independent/conddcorr_128_4096_83.npz",
    "independent/conddcorr_128_4096_84.npz",
    "independent/conddcorr_128_4096_85.npz",
    "independent/conddcorr_128_4096_86.npz",
    "independent/conddcorr_128_4096_87.npz",
    "independent/conddcorr_128_4096_88.npz",
    "independent/conddcorr_128_4096_89.npz",
    "independent/conddcorr_128_4096_90.npz",
    "independent/conddcorr_128_4096_91.npz",
    "independent/conddcorr_128_4096_92.npz",
    "independent/conddcorr_128_4096_93.npz",
    "independent/conddcorr_128_4096_94.npz",
    "independent/conddcorr_128_4096_95.npz",
    "independent/conddcorr_128_4096_96.npz",
    "independent/conddcorr_128_4096_97.npz",
    "independent/conddcorr_128_4096_98.npz",
    "independent/conddcorr_128_4096_99.npz",
    "independent/conddcorr_256_4096_0.npz",
    "independent/conddcorr_256_4096_1.npz",
    "independent/conddcorr_256_4096_2.npz",
    "independent/conddcorr_256_4096_3.npz",
    "independent/conddcorr_256_4096_4.npz",
    "independent/conddcorr_256_4096_5.npz",
    "independent/conddcorr_256_4096_6.npz",
    "independent/conddcorr_256_4096_7.npz",
    "independent/conddcorr_256_4096_8.npz",
    "independent/conddcorr_256_4096_9.npz",
    "independent/conddcorr_256_4096_10.npz",
    "independent/conddcorr_256_4096_11.npz",
    "independent/conddcorr_256_4096_12.npz",
    "independent/conddcorr_256_4096_13.npz",
    "independent/conddcorr_256_4096_14.npz",
    "independent/conddcorr_256_4096_15.npz",
    "independent/conddcorr_256_4096_16.npz",
    "independent/conddcorr_256_4096_17.npz",
    "independent/conddcorr_256_4096_18.npz",
    "independent/conddcorr_256_4096_19.npz",
    "independent/conddcorr_256_4096_20.npz",
    "independent/conddcorr_256_4096_21.npz",
    "independent/conddcorr_256_4096_22.npz",
    "independent/conddcorr_256_4096_23.npz",
    "independent/conddcorr_256_4096_24.npz",
    "independent/conddcorr_256_4096_25.npz",
    "independent/conddcorr_256_4096_26.npz",
    "independent/conddcorr_256_4096_27.npz",
    "independent/conddcorr_256_4096_28.npz",
    "independent/conddcorr_256_4096_29.npz",
    "independent/conddcorr_256_4096_30.npz",
    "independent/conddcorr_256_4096_31.npz",
    "independent/conddcorr_256_4096_32.npz",
    "independent/conddcorr_256_4096_33.npz",
    "independent/conddcorr_256_4096_34.npz",
    "independent/conddcorr_256_4096_35.npz",
    "independent/conddcorr_256_4096_36.npz",
    "independent/conddcorr_256_4096_37.npz",
    "independent/conddcorr_256_4096_38.npz",
    "independent/conddcorr_256_4096_39.npz",
    "independent/conddcorr_256_4096_40.npz",
    "independent/conddcorr_256_4096_41.npz",
    "independent/conddcorr_256_4096_42.npz",
    "independent/conddcorr_256_4096_43.npz",
    "independent/conddcorr_256_4096_44.npz",
    "independent/conddcorr_256_4096_45.npz",
    "independent/conddcorr_256_4096_46.npz",
    "independent/conddcorr_256_4096_47.npz",
    "independent/conddcorr_256_4096_48.npz",
    "independent/conddcorr_256_4096_49.npz",
    "independent/conddcorr_256_4096_50.npz",
    "independent/conddcorr_256_4096_51.npz",
    "independent/conddcorr_256_4096_52.npz",
    "independent/conddcorr_256_4096_53.npz",
    "independent/conddcorr_256_4096_54.npz",
    "independent/conddcorr_256_4096_55.npz",
    "independent/conddcorr_256_4096_56.npz",
    "independent/conddcorr_256_4096_57.npz",
    "independent/conddcorr_256_4096_58.npz",
    "independent/conddcorr_256_4096_59.npz",
    "independent/conddcorr_256_4096_60.npz",
    "independent/conddcorr_256_4096_61.npz",
    "independent/conddcorr_256_4096_62.npz",
    "independent/conddcorr_256_4096_63.npz",
    "independent/conddcorr_256_4096_64.npz",
    "independent/conddcorr_256_4096_65.npz",
    "independent/conddcorr_256_4096_66.npz",
    "independent/conddcorr_256_4096_67.npz",
    "independent/conddcorr_256_4096_68.npz",
    "independent/conddcorr_256_4096_69.npz",
    "independent/conddcorr_256_4096_70.npz",
    "independent/conddcorr_256_4096_71.npz",
    "independent/conddcorr_256_4096_72.npz",
    "independent/conddcorr_256_4096_73.npz",
    "independent/conddcorr_256_4096_74.npz",
    "independent/conddcorr_256_4096_75.npz",
    "independent/conddcorr_256_4096_76.npz",
    "independent/conddcorr_256_4096_77.npz",
    "independent/conddcorr_256_4096_78.npz",
    "independent/conddcorr_256_4096_79.npz",
    "independent/conddcorr_256_4096_80.npz",
    "independent/conddcorr_256_4096_81.npz",
    "independent/conddcorr_256_4096_82.npz",
    "independent/conddcorr_256_4096_83.npz",
    "independent/conddcorr_256_4096_84.npz",
    "independent/conddcorr_256_4096_85.npz",
    "independent/conddcorr_256_4096_86.npz",
    "independent/conddcorr_256_4096_87.npz",
    "independent/conddcorr_256_4096_88.npz",
    "independent/conddcorr_256_4096_89.npz",
    "independent/conddcorr_256_4096_90.npz",
    "independent/conddcorr_256_4096_91.npz",
    "independent/conddcorr_256_4096_92.npz",
    "independent/conddcorr_256_4096_93.npz",
    "independent/conddcorr_256_4096_94.npz",
    "independent/conddcorr_256_4096_95.npz",
    "independent/conddcorr_256_4096_96.npz",
    "independent/conddcorr_256_4096_97.npz",
    "independent/conddcorr_256_4096_98.npz",
    "independent/conddcorr_256_4096_99.npz",
    "independent/conddcorr_512_4096_0.npz",
    "independent/conddcorr_512_4096_1.npz",
    "independent/conddcorr_512_4096_2.npz",
    "independent/conddcorr_512_4096_3.npz",
    "independent/conddcorr_512_4096_4.npz",
    "independent/conddcorr_512_4096_5.npz",
    "independent/conddcorr_512_4096_6.npz",
    "independent/conddcorr_512_4096_7.npz",
    "independent/conddcorr_512_4096_8.npz",
    "independent/conddcorr_512_4096_9.npz",
    "independent/conddcorr_512_4096_10.npz",
    "independent/conddcorr_512_4096_11.npz",
    "independent/conddcorr_512_4096_12.npz",
    "independent/conddcorr_512_4096_13.npz",
    "independent/conddcorr_512_4096_14.npz",
    "independent/conddcorr_512_4096_15.npz",
    "independent/conddcorr_512_4096_16.npz",
    "independent/conddcorr_512_4096_17.npz",
    "independent/conddcorr_512_4096_18.npz",
    "independent/conddcorr_512_4096_19.npz",
    "independent/conddcorr_512_4096_21.npz",
    "independent/conddcorr_512_4096_22.npz",
    "independent/conddcorr_512_4096_23.npz",
    "independent/conddcorr_512_4096_24.npz",
    "independent/conddcorr_512_4096_25.npz",
    "independent/conddcorr_512_4096_26.npz",
    "independent/conddcorr_512_4096_27.npz",
    "independent/conddcorr_512_4096_28.npz",
    "independent/conddcorr_512_4096_29.npz",
    "independent/conddcorr_512_4096_30.npz",
    "independent/conddcorr_512_4096_31.npz",
    "independent/conddcorr_512_4096_32.npz",
    "independent/conddcorr_512_4096_33.npz",
    "independent/conddcorr_512_4096_34.npz",
    "independent/conddcorr_512_4096_35.npz",
    "independent/conddcorr_512_4096_36.npz",
    "independent/conddcorr_512_4096_37.npz",
    "independent/conddcorr_512_4096_38.npz",
    "independent/conddcorr_512_4096_39.npz",
    "independent/conddcorr_512_4096_40.npz",
    "independent/conddcorr_512_4096_41.npz",
    "independent/conddcorr_512_4096_42.npz",
    "independent/conddcorr_512_4096_43.npz",
    "independent/conddcorr_512_4096_44.npz",
    "independent/conddcorr_512_4096_45.npz",
    "independent/conddcorr_512_4096_46.npz",
    "independent/conddcorr_512_4096_47.npz",
    "independent/conddcorr_512_4096_48.npz",
    "independent/conddcorr_512_4096_49.npz",
    "independent/conddcorr_512_4096_50.npz",
    "independent/conddcorr_512_4096_51.npz",
    "independent/conddcorr_512_4096_52.npz",
    "independent/conddcorr_512_4096_53.npz",
    "independent/conddcorr_512_4096_54.npz",
    "independent/conddcorr_512_4096_55.npz",
    "independent/conddcorr_512_4096_56.npz",
    "independent/conddcorr_512_4096_57.npz",
    "independent/conddcorr_512_4096_58.npz",
    "independent/conddcorr_512_4096_59.npz",
    "independent/conddcorr_512_4096_60.npz",
    "independent/conddcorr_512_4096_61.npz",
    "independent/conddcorr_512_4096_62.npz",
    "independent/conddcorr_512_4096_63.npz",
    "independent/conddcorr_512_4096_64.npz",
    "independent/conddcorr_512_4096_65.npz",
    "independent/conddcorr_512_4096_66.npz",
    "independent/conddcorr_512_4096_67.npz",
    "independent/conddcorr_512_4096_68.npz",
    "independent/conddcorr_512_4096_69.npz",
    "independent/conddcorr_512_4096_70.npz",
    "independent/conddcorr_512_4096_71.npz",
    "independent/conddcorr_512_4096_72.npz",
    "independent/conddcorr_512_4096_73.npz",
    "independent/conddcorr_512_4096_74.npz",
    "independent/conddcorr_512_4096_75.npz",
    "independent/conddcorr_512_4096_76.npz",
    "independent/conddcorr_512_4096_77.npz",
    "independent/conddcorr_512_4096_78.npz",
    "independent/conddcorr_512_4096_79.npz",
    "independent/conddcorr_512_4096_80.npz",
    "independent/conddcorr_512_4096_81.npz",
    "independent/conddcorr_512_4096_82.npz",
    "independent/conddcorr_512_4096_83.npz",
    "independent/conddcorr_512_4096_84.npz",
    "independent/conddcorr_512_4096_85.npz",
    "independent/conddcorr_512_4096_86.npz",
    "independent/conddcorr_512_4096_87.npz",
    "independent/conddcorr_512_4096_88.npz",
    "independent/conddcorr_512_4096_89.npz",
    "independent/conddcorr_512_4096_90.npz",
    "independent/conddcorr_512_4096_91.npz",
    "independent/conddcorr_512_4096_92.npz",
    "independent/conddcorr_512_4096_93.npz",
    "independent/conddcorr_512_4096_94.npz",
    "independent/conddcorr_512_4096_95.npz",
    "independent/conddcorr_512_4096_96.npz",
    "independent/conddcorr_512_4096_97.npz",
    "independent/conddcorr_512_4096_98.npz",
    "independent/conddcorr_512_4096_99.npz",
    "independent/conddcorr_1024_4096_0.npz",
    "independent/conddcorr_1024_4096_1.npz",
    "independent/conddcorr_1024_4096_2.npz",
    "independent/conddcorr_1024_4096_4.npz",
    "independent/conddcorr_1024_4096_6.npz",
    "independent/conddcorr_1024_4096_7.npz",
    "independent/conddcorr_1024_4096_8.npz",
    "independent/conddcorr_1024_4096_9.npz",
    "independent/conddcorr_1024_4096_10.npz",
    "independent/conddcorr_1024_4096_11.npz",
    "independent/conddcorr_1024_4096_12.npz",
    "independent/conddcorr_1024_4096_13.npz",
    "independent/conddcorr_1024_4096_14.npz",
    "independent/conddcorr_1024_4096_15.npz",
    "independent/conddcorr_1024_4096_16.npz",
    "independent/conddcorr_1024_4096_17.npz",
    "independent/conddcorr_1024_4096_18.npz",
    "independent/conddcorr_1024_4096_19.npz",
    "independent/conddcorr_1024_4096_20.npz",
    "independent/conddcorr_1024_4096_21.npz",
    "independent/conddcorr_1024_4096_23.npz",
    "independent/conddcorr_1024_4096_24.npz",
    "independent/conddcorr_1024_4096_25.npz",
    "independent/conddcorr_1024_4096_26.npz",
    "independent/conddcorr_1024_4096_27.npz",
    "independent/conddcorr_1024_4096_28.npz",
    "independent/conddcorr_1024_4096_29.npz",
    "independent/conddcorr_1024_4096_30.npz",
    "independent/conddcorr_1024_4096_31.npz",
    "independent/conddcorr_1024_4096_32.npz",
    "independent/conddcorr_1024_4096_33.npz",
    "independent/conddcorr_1024_4096_34.npz",
    "independent/conddcorr_1024_4096_35.npz",
    "independent/conddcorr_1024_4096_36.npz",
    "independent/conddcorr_1024_4096_37.npz",
    "independent/conddcorr_1024_4096_38.npz",
    "independent/conddcorr_1024_4096_39.npz",
    "independent/conddcorr_1024_4096_40.npz",
    "independent/conddcorr_1024_4096_42.npz",
    "independent/conddcorr_1024_4096_44.npz",
    "independent/conddcorr_1024_4096_45.npz",
    "independent/conddcorr_1024_4096_46.npz",
    "independent/conddcorr_1024_4096_47.npz",
    "independent/conddcorr_1024_4096_48.npz",
    "independent/conddcorr_1024_4096_50.npz",
    "independent/conddcorr_1024_4096_51.npz",
    "independent/conddcorr_1024_4096_52.npz",
    "independent/conddcorr_1024_4096_54.npz",
    "independent/conddcorr_1024_4096_57.npz",
    "independent/conddcorr_1024_4096_60.npz",
    "independent/conddcorr_1024_4096_62.npz",
    "independent/conddcorr_1024_4096_63.npz",
    "independent/conddcorr_1024_4096_64.npz",
    "independent/conddcorr_1024_4096_65.npz",
    "independent/conddcorr_1024_4096_67.npz",
    "independent/conddcorr_1024_4096_71.npz",
]


def add_noise(x, intens=1e-10):
    # small noise to break degeneracy, see doc.
    return x + intens * np.random.random_sample(x.shape)


def query_neighbors(tree, x, k):
    return tree.query(x, k=k + 1)[0][:, k]


def count_neighbors(tree, x, r):
    return tree.query_radius(x, r, count_only=True)


def avgdigamma(points, dvec):
    # This part finds number of neighbors in some radius in the marginal space
    # returns expectation value of <psi(nx)>
    tree = build_tree(points)
    dvec = dvec - 1e-15
    num_points = count_neighbors(tree, points, dvec)
    return np.mean(digamma(num_points))


def build_tree(points):
    if points.shape[1] >= 20:
        return BallTree(points, metric="chebyshev")
    return KDTree(points, metric="chebyshev")


def mi_ksg(x, y, z=None, k=3, base=2):
    """Mutual information of x and y (conditioned on z if z is not None)
    x, y should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
    if x is a one-dimensional scalar and we have four samples
    """
    assert len(x) == len(y), "Arrays should have same length"
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    x, y = np.asarray(x), np.asarray(y)
    x, y = x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1)
    x = add_noise(x)
    y = add_noise(y)
    points = [x, y]
    if z is not None:
        z = np.asarray(z)
        z = z.reshape(z.shape[0], -1)
        points.append(z)
    points = np.hstack(points)
    # Find nearest neighbors in joint space, p=inf means max-norm
    tree = build_tree(points)
    dvec = query_neighbors(tree, points, k)
    if z is None:
        a, b, c, d = (
            avgdigamma(x, dvec),
            avgdigamma(y, dvec),
            digamma(k),
            digamma(len(x)),
        )
    else:
        xz = np.c_[x, z]
        yz = np.c_[y, z]
        a, b, c, d = (
            avgdigamma(xz, dvec),
            avgdigamma(yz, dvec),
            avgdigamma(z, dvec),
            digamma(k),
        )
    return (-a - b + c + d) / log(base)


seed = 12345
rng = np.random.default_rng(seed)

# hard-coded parameters
n_estimators = 500
max_features = 0.3
test_size = 0.2


def _run_parallel_comight(
    idx,
    n_samples,
    seed,
    n_features_2,
    test_size,
    sim_type,
    rootdir,
    output_dir,
):
    """Run parallel job on pre-generated data.

    Parameters
    ----------
    idx : int
        The index of the pre-generated dataset, stored as npz file.
    n_samples : int
        The number of samples to keep.
    seed : int
        The random seed.
    n_features_2 : int
        The number of dimensions to keep in feature set 2.
    test_size : float
        The size of the test set to use for predictive-model based tests.
    sim_type : str
        The simulation type. Either 'independent', 'collider', 'confounder',
        or 'direct-indirect'.
    rootdir : str
        The root directory where 'data/' and 'output/' will be.
    run_cdcorr : bool, optional
        Whether or not to run conditional dcorr, by default True.
    """
    n_jobs = 1
    n_features_ends = [100, None]

    # set output directory to save npz files
    output_dir = os.path.join(rootdir, f"output/{output_dir}/{sim_type}/")
    os.makedirs(output_dir, exist_ok=True)

    # load data
    npy_data = np.load(os.path.join(rootdir, f"data/{sim_type}/{sim_type}_{idx}.npz"))

    X = npy_data["X"]
    y = npy_data["y"]

    X = X[:, : 100 + n_features_2]
    if n_samples < X.shape[0]:
        cv = StratifiedShuffleSplit(n_splits=1, train_size=n_samples)
        for train_idx, _ in cv.split(X, y):
            continue
        X = X[train_idx, :]
        y = y[train_idx, ...].squeeze()
    assert len(X) == len(y)
    assert len(y) == n_samples
    n_features_ends[1] = X.shape[1]

    est = FeatureImportanceForestClassifier(
        estimator=HonestForestClassifier(
            n_estimators=n_estimators,
            tree_estimator=MultiViewDecisionTreeClassifier(
                max_features=[max_features, min(n_features_2, max_features * 100)],
                feature_set_ends=n_features_ends,
                apply_max_features_per_feature_set=True,
            ),
            random_state=seed,
            honest_fraction=0.5,
            n_jobs=n_jobs,
        ),
        random_state=seed,
        test_size=test_size,
        sample_dataset_per_tree=False,
    )

    # now compute the pvalue when shuffling X2
    covariate_index = np.arange(n_features_ends[0], n_features_ends[1])

    # Estimate CMI with
    mi_rf, pvalue = est.test(
        X,
        y,
        covariate_index=covariate_index,
        return_posteriors=True,
        metric="mi",
    )
    comight_posteriors_x2 = est.observe_posteriors_
    comight_null_posteriors_x2 = est.permute_posteriors_

    samples = est.observe_samples_
    permute_samples = est.permute_samples_

    assert np.isnan(comight_posteriors_x2[:, samples, :]).sum() == 0

    np.savez(
        os.path.join(output_dir, f"comight_{n_samples}_{n_features_2}_{idx}.npz"),
        n_samples=n_samples,
        n_features_2=n_features_2,
        y_true=y,
        comight_pvalue=pvalue,
        comight_mi=mi_rf,
        comight_posteriors_x2=comight_posteriors_x2,
        comight_null_posteriors_x2=comight_null_posteriors_x2,
    )


def _run_parallel_cond_dcorr(
    idx, n_samples, seed, n_features_2, sim_type, rootdir, output_dir
):
    """Run parallel job on pre-generated data.

    Parameters
    ----------
    idx : int
        The index of the pre-generated dataset, stored as npz file.
    n_samples : int
        The number of samples to keep.
    seed : int
        The random seed.
    n_features_2 : int
        The number of dimensions to keep in feature set 2.
    test_size : float
        The size of the test set to use for predictive-model based tests.
    sim_type : str
        The simulation type. Either 'independent', 'collider', 'confounder',
        or 'direct-indirect'.
    rootdir : str
        The root directory where 'data/' and 'output/' will be.
    run_cdcorr : bool, optional
        Whether or not to run conditional dcorr, by default True.
    """
    n_jobs = 1
    n_features_ends = [100, None]

    # set output directory to save npz files
    output_dir = os.path.join(rootdir, f"output/{output_dir}/{sim_type}/")
    os.makedirs(output_dir, exist_ok=True)

    # load data
    npy_data = np.load(os.path.join(rootdir, f"data/{sim_type}/{sim_type}_{idx}.npz"))

    X = npy_data["X"]
    y = npy_data["y"]

    X = X[:, : 100 + n_features_2]
    if n_samples < X.shape[0]:
        cv = StratifiedShuffleSplit(n_splits=1, train_size=n_samples)
        for train_idx, _ in cv.split(X, y):
            continue
        X = X[train_idx, :]
        y = y[train_idx, ...].squeeze()
    assert len(X) == len(y)
    assert len(y) == n_samples
    n_features_ends[1] = X.shape[1]

    # now compute the pvalue when shuffling X2
    covariate_index = np.arange(n_features_ends[0], n_features_ends[1])

    cdcorr = ConditionalDcorr(bandwidth="silverman")
    Z = X[:, covariate_index]
    mask_array = np.ones(X.shape[1])
    mask_array[covariate_index] = 0
    mask_array = mask_array.astype(bool)
    try:
        X_minus_Z = X[:, mask_array]
        if np.var(y) < 0.001:
            raise RuntimeError(
                f"{n_samples}_{n_features_2}_{idx} errored out with no variance in y"
            )
        cdcorr_stat, cdcorr_pvalue = cdcorr.test(
            X_minus_Z.copy(), y.copy(), Z.copy(), random_state=seed
        )

        np.savez(
            os.path.join(output_dir, f"conddcorr_{n_samples}_{n_features_2}_{idx}.npz"),
            n_samples=n_samples,
            n_features_2=n_features_2,
            y_true=y,
            cdcorr_pvalue=cdcorr_pvalue,
            cdcorr_stat=cdcorr_stat,
        )
    except Exception as e:
        print(e)


if __name__ == "__main__":
    # Extract arguments from terminal input
    idx = int(sys.argv[1])
    n_samples = int(sys.argv[2])
    n_features_2 = int(sys.argv[3])
    sim_type = sys.argv[4]
    rootdir = sys.argv[5]

    output_dir = "overall"

    fname = f"{sim_type}/conddcorr_{n_samples}_{n_features_2}_{idx}.npz"
    if fname in FINISHED_LIST:
        print("Already finished job")
    else:
        # Call your function with the extracted arguments
        _run_parallel_cond_dcorr(
            idx, n_samples, seed, n_features_2, sim_type, rootdir, output_dir
        )
