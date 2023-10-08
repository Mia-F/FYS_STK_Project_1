import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.linear_model import Ridge


def design_matrix(x, y, degree):
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((degree + 1) * (degree + 2) / 2)  # Number of elements in beta
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
    np.random.shuffle(indices) #randomize the indices

    k_fold_indices = []

    for i in range(k):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size
        test_indices = indices[test_start:test_end]
        train_indices = np.concatenate([indices[:test_start], indices[test_end:]])
        k_fold_indices.append((train_indices, test_indices))

    return k_fold_indices

# A seed just to ensure that the random numbers are the same for every run.
np.random.seed(2024)

# Generate the data.
nsamples = 100
x = np.random.rand(nsamples)
y = 3 * x**2 + np.random.randn(nsamples)
data = np.column_stack((x, y))

# Decide which values of lambda to use
nlambdas = 500
lambdas = np.logspace(-3, 5, nlambdas)

colors = ['b', 'g', 'r', 'c', 'm', 'y']

n_bootstrap_iter = 100  # Number of bootstrap samples to create
scores_bootstrap = np.zeros((nlambdas, n_bootstrap_iter))

for i, lmb in enumerate(lambdas):
    ridge = Ridge(alpha=lmb)
    for j in range(n_bootstrap_iter):
        data_bootstrap = resample(data, replace=True, n_samples=nsamples, random_state=j)
        xtrain, ytrain = data_bootstrap[:, 0], data_bootstrap[:, 1]
        Xtrain = design_matrix(xtrain, ytrain, degree=6)
        ridge.fit(Xtrain, ytrain[:, np.newaxis])

        # We use the original data as test set in this case
        Xtest = design_matrix(x, y, degree=6)
        ypred = ridge.predict(Xtest)
        scores_bootstrap[i, j] = np.sum((ypred.ravel() - y)**2) / np.size(ypred)

estimated_mse_bootstrap = np.mean(scores_bootstrap, axis=1)

# Plotting
plt.figure(figsize=(10, 6))

plt.plot(np.log10(lambdas), estimated_mse_bootstrap, 'k--', label='Bootstrap')

for idx, k in enumerate(range(5, 11)):
    k_fold_indices = k_fold(data, k)
    scores_KFold = np.zeros((nlambdas, k))

    for i, lmb in enumerate(lambdas):
        ridge = Ridge(alpha=lmb)
        for j, (train_indices, test_indices) in enumerate(k_fold_indices):
            xtrain, ytrain = data[train_indices, 0], data[train_indices, 1]
            xtest, ytest = data[test_indices, 0], data[test_indices, 1]

            Xtrain = design_matrix(xtrain, ytrain, degree=6)
            ridge.fit(Xtrain, ytrain[:, np.newaxis])

            Xtest = design_matrix(xtest, ytest, degree=6)
            ypred = ridge.predict(Xtest)
            scores_KFold[i, j] = np.sum((ypred.ravel() - ytest)**2) / np.size(ypred)

    estimated_mse_KFold = np.mean(scores_KFold, axis=1)
    plt.plot(np.log10(lambdas), estimated_mse_KFold, color=colors[idx], label=f'k = {k}')

plt.xlabel(r'$\log_{10}(\lambda)$')
plt.ylabel('MSE')
plt.legend()
plt.title('Comparison of Bootstrap and k-Fold CV Methods')
plt.show()