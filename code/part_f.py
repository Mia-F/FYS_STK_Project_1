"""
Cross-validation as resampling techniques
"""

import numpy as np
from part_a import Franke_function, design_matrix, MSE, R2, beta_OLS
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from sklearn.model_selection import train_test_split
import time
from tqdm import tqdm

def k_fold(data):
    """
    Takes in data and splits it in to k different segments....
    """
    