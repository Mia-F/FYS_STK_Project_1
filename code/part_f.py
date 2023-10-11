import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Seaborn style setting
sns.set_theme()
font = {'weight': 'bold', 'size': 16}
plt.rc('font', **font)
sns.set(font_scale=1.5)

def design_matrix(x, y, degree):
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)
    N = len(x)
    l = int((degree + 1) * (degree + 2) / 2)
    X = np.ones((N, l))
    for i in range(1, degree + 1):
        q = int((i) * (i + 1) / 2)
        for k in range(i + 1):
            X[:, q + k] = (x**(i - k)) * (y**k)
    return X

def Franke_function(x, y, noise=False):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    if noise:
        noise_val = np.random.normal(0, 0.1, len(x))
        return term1 + term2 + term3 + term4 + noise_val
    else:
        return term1 + term2 + term3 + term4

def k_fold(data, k):
    n_samples = len(data)
    fold_size = n_samples // k
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    k_fold_indices = []
    for i in range(k):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size
        test_indices = indices[test_start:test_end]
        train_indices = np.concatenate([indices[:test_start], indices[test_end:]])
        k_fold_indices.append((train_indices, test_indices))
    return k_fold_indices

np.random.seed(2024)
nsamples = 100
x = np.random.rand(nsamples)
y = np.random.rand(nsamples)
z = Franke_function(x, y, noise=True)
data = np.column_stack((x, y, z))

# Lambda values and colors
nlambdas = 6  
lambdas = np.logspace(-3, 3, nlambdas)
colors = sns.color_palette("husl", nlambdas)

# Degree setting
max_degree = 15
degrees = np.arange(1, max_degree + 1)
k = 5

# Ridge Regression
plt.figure(figsize=(10, 6))
for i, lmb in enumerate(lambdas):
    mse_per_degree_ridge = []
    for degree in degrees:
        scores_KFold = np.zeros(k)
        k_fold_indices = k_fold(data, k)
        for j, (train_indices, test_indices) in enumerate(k_fold_indices):
            train, test = data[train_indices], data[test_indices]
            Xtrain = design_matrix(train[:,0], train[:,1], degree=degree)
            ridge = Ridge(alpha=lmb)
            ridge.fit(Xtrain, train[:,2])
            Xtest = design_matrix(test[:,0], test[:,1], degree=degree)
            zpred = ridge.predict(Xtest)
            scores_KFold[j] = np.mean((zpred.ravel() - test[:,2]) ** 2)
        mse_per_degree_ridge.append(np.mean(scores_KFold))
    plt.plot(degrees, mse_per_degree_ridge, marker='o', color=colors[i], label=f'Ridge, λ={lmb:.1e}')

plt.xlabel('Degree')
plt.ylabel('MSE (log scale)')
plt.yscale('log')
plt.legend()
plt.savefig('Ridge_crossval_mse_deg')
plt.show()

# Lasso Regression
plt.figure(figsize=(10, 6))
for i, lmb in enumerate(lambdas):
    mse_per_degree_lasso = []
    for degree in degrees:
        scores_KFold = np.zeros(k)
        k_fold_indices = k_fold(data, k)
        for j, (train_indices, test_indices) in enumerate(k_fold_indices):
            train, test = data[train_indices], data[test_indices]
            Xtrain = design_matrix(train[:,0], train[:,1], degree=degree)
            lasso = Lasso(alpha=lmb, max_iter=10000)
            lasso.fit(Xtrain, train[:,2])
            Xtest = design_matrix(test[:,0], test[:,1], degree=degree)
            zpred = lasso.predict(Xtest)
            scores_KFold[j] = np.mean((zpred.ravel() - test[:,2]) ** 2)
        mse_per_degree_lasso.append(np.mean(scores_KFold))
    plt.plot(degrees, mse_per_degree_lasso, marker='o', color=colors[i], label=f'Lasso, λ={lmb:.1e}')

plt.xlabel('Degree')
plt.ylabel('MSE (log scale)')
plt.yscale('log')
plt.legend()
plt.savefig('Lasso_crossval_mse_deg')
plt.show()
