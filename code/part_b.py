"""
Implementing Ridge regression
"""

import numpy as np
from part_a import Franke_function, design_matrix, MSE, R2
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from sklearn.model_selection import train_test_split
import time
from tqdm import tqdm

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

np.random.seed(2023)

def beta_ridge(X, y, lamb):
    return (np.linalg.pinv(X.T @ X + lamb * np.identity(len(X[0]))) @ X.T @ y)

def model(lambdas, degree):
    MSE_test = np.zeros(len(lambdas))
    MSE_train = np.zeros(len(lambdas))
    X = design_matrix(x,y, degree)
    X_train, X_test, y_train, y_test = train_test_split(X, f, test_size=0.2) 

    for l in tqdm(range(len(lambdas))):

        beta = beta_ridge(X_train, y_train, lambdas[l])

        model_train = X_train @ beta
        model_test = X_test @ beta

        #calculating the MSE and R2 score
        MSE_train[l] = MSE(y_train, model_train)
        MSE_test[l] = MSE(y_test, model_test)

        time.sleep(0.01)
    return  MSE_train, MSE_test

    
n = 1000
degree = np.array([1])
lambdas = np.linspace(1e-5, 1e5,1000)

x = np.sort(np.random.uniform(0, 1, n))
y = np.sort(np.random.uniform(0, 1, n))

#x,y = np.meshgrid(x,y)
f = Franke_function(x,y,noise=True)


Error_train = np.zeros((len(degree), len(lambdas)))
Error_test = np.zeros((len(degree), len(lambdas)))

for d in range(len(degree)):
    MSE_train, MSE_test = model(lambdas, d)
    Error_train[d] = MSE_train
    Error_test[d] = MSE_test

    plt.plot(np.log10(lambdas), Error_train[d], label="Train")
    plt.plot(np.log10(lambdas), Error_test[d], label="Test")
    plt.legend()
    plt.show()
