import numpy as np
from part_a import Franke_function, design_matrix, MSE, R2
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from sklearn.model_selection import train_test_split

np.random.seed(2023)

def bootstrap():
    """Function for doing stuff."""
    return 