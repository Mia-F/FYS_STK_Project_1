import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
import seaborn as sns

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

# Seed and data generation
np.random.seed(2024)
nsamples = 100
x = np.random.rand(nsamples)
y = 3 * x ** 2 + np.random.randn(nsamples)
data = np.column_stack((x, y))

# Lambda values and colors
nlambdas = 6  # reduced for better visualization
lambdas = np.logspace(-3, 3, nlambdas)
colors = sns.color_palette("husl", nlambdas)  # using seaborn color palette

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
            xtrain, ytrain = data[train_indices, 0], data[train_indices, 1]
            xtest, ytest = data[test_indices, 0], data[test_indices, 1]
            Xtrain = design_matrix(xtrain, ytrain, degree=degree)
            ridge = Ridge(alpha=lmb)
            ridge.fit(Xtrain, ytrain)
            Xtest = design_matrix(xtest, ytest, degree=degree)
            ypred = ridge.predict(Xtest)
            scores_KFold[j] = np.mean((ypred.ravel() - ytest) ** 2)
        mse_per_degree_ridge.append(np.mean(scores_KFold))
    plt.plot(degrees, mse_per_degree_ridge, marker='o', color=colors[i], label=f'Ridge, λ={lmb:.1e}')

plt.xlabel('Degree')
plt.ylabel('MSE (log scale)')
plt.yscale('log')  # setting y-axis to log scale
plt.legend()
#plt.savefig('Ridge_crossval_mse_deg')
plt.show()

# Lasso Regression
plt.figure(figsize=(10, 6))
for i, lmb in enumerate(lambdas):
    mse_per_degree_lasso = []
    for degree in degrees:
        scores_KFold = np.zeros(k)
        k_fold_indices = k_fold(data, k)
        for j, (train_indices, test_indices) in enumerate(k_fold_indices):
            xtrain, ytrain = data[train_indices, 0], data[train_indices, 1]
            xtest, ytest = data[test_indices, 0], data[test_indices, 1]
            Xtrain = design_matrix(xtrain, ytrain, degree=degree)
            lasso = Lasso(alpha=lmb, max_iter=10000)
            lasso.fit(Xtrain, ytrain)
            Xtest = design_matrix(xtest, ytest, degree=degree)
            ypred = lasso.predict(Xtest)
            scores_KFold[j] = np.mean((ypred.ravel() - ytest) ** 2)
        mse_per_degree_lasso.append(np.mean(scores_KFold))
    plt.plot(degrees, mse_per_degree_lasso, marker='o', color=colors[i], label=f'Lasso, λ={lmb:.1e}')

plt.xlabel('Degree')
plt.ylabel('MSE (log scale)')
plt.yscale('log')  # setting y-axis to log scale
plt.legend()
#plt.savefig('Lasso_crossval_mse_deg')
plt.show()