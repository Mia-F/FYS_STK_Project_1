"""
Implementing Ridge regression
"""

import numpy as np
import seaborn as sns
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

    
n = 20
degree = np.linspace(0,5,6, dtype=int)
lambdas = np.logspace(-8, 2, 1000)


x = np.sort(np.random.uniform(0, 1, n))
y = np.sort(np.random.uniform(0, 1, n))

x,y = np.meshgrid(x,y)

x = np.ravel(x)
y = np.ravel(y)

f = Franke_function(x,y, noise=True)


Error_train = np.zeros((len(degree), len(lambdas)))
Error_test = np.zeros((len(degree), len(lambdas)))
R2_score_train = np.zeros((len(degree), len(lambdas)))
R2_score_test = np.zeros((len(degree), len(lambdas)))


for d in tqdm(range(len(degree))):
    print(d)
    X = design_matrix(x,y, degree[d])
    X_train, X_test, y_train, y_test = train_test_split(X, f, test_size=0.2) 
    for i in range(len(lambdas)):
        beta = beta_ridge(X_train, y_train, lambdas[i])

        model_train = X_train @ beta
        model_test = X_test @ beta
       
        Error_train[d][i] = MSE(y_train, model_train)
        Error_test[d][i] = MSE(y_test, model_test)
        R2_score_train[d][i] = R2(y_train, model_train)
        R2_score_test[d][i] = R2(y_test, model_test)


#Plotting heatmaps for lambda dependency
plt.title(r"Heatmap of the MSE for the training data as a fucntion of $\lambda$ values and complexity")
plt.imshow(Error_train, aspect='auto')
plt.grid()
plt.xlabel(r"$\lambda$ values from $10^{-8}$ to $10^{2}$")
plt.ylabel("Degree")
plt.colorbar(label="MSE")
plt.show()

plt.title(r"Heatmap of the MSE for the test data as a fucntion of $\lambda$ values and complexity")
plt.imshow(Error_test, aspect='auto')
plt.grid()
plt.xlabel(r"$\lambda$ values from $10^{-5}$ to $10^{5}$")
plt.ylabel("Degree")
plt.colorbar(label="MSE")
plt.show()

#Plotting MSE and R2 scores for polynomial 10 a
plt.title(r"MSE as a function of $\lambda$ for a model fitted with a polynomial of degree 10")  
plt.semilogx(lambdas, Error_train[-1][:], label="Error train")
plt.semilogx(lambdas, Error_test[-1][:], label="Error test")
plt.xlabel(r"$\lambda$")
plt.ylabel("MSE")
plt.legend()
plt.show()


plt.title(r"R2 score as a function of $\lambda$ for a model fitted with a polynomial of degree 10")  
plt.semilogx(lambdas, R2_score_train[-1][:], label="Error train")
plt.semilogx(lambdas, R2_score_test[-1][:], label="Error test")
plt.xlabel(r"$\lambda$")
plt.ylabel("R2 score")
plt.legend()
plt.show()

plt.plot(degree, Error_train)
plt.plot(degree, Error_test)
plt.show()

#Plottig 6 3D subplots of diffrent fitings of franke function
n=100
lambdas = 10e-5
x = np.sort(np.random.uniform(0, 1, n))
y = np.sort(np.random.uniform(0, 1, n))
x,y = np.meshgrid(x,y)

f = Franke_function(x,y, noise=True)
degree = np.array([0,1,2,3,4,5])

fig = plt.figure()

fig.suptitle(r"Models for the Franke function fitted with Ridge of different complexities and $\lambda = 10^{-5}$", fontsize = 20)

for i in range(len(degree)):
    if i == 0:
        ax = fig.add_subplot(2, len(degree)/2, i+1, projection='3d')
        ax.plot_surface(x, y, f, cmap=cm.twilight,linewidth=0, antialiased=False)
        ax.set_title(f"Real data with noise", fontsize = 20)
    else:
        #Create the design matrix
        X = design_matrix(x,y,degree[i])
        
        #Calculate the beta values
        beta = beta_ridge(X,f.flatten(), lambdas)
        model = X @ beta
        model = model.reshape(n,n)

        #For ploting 
        ax = fig.add_subplot(2, len(degree)/2, i+1, projection='3d')
        ax.plot_surface(x, y, model, cmap=cm.twilight,linewidth=0, antialiased=False)
        ax.set_title(f"Degree = {i}", fontsize = 20)

plt.show()