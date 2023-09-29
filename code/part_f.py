import numpy as np
import matplotlib.pyplot as plt
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
np.random.seed(3155)

# Generate the data.
nsamples = 100
x = np.random.rand(nsamples)
y = 3 * x**2 + np.random.randn(nsamples)
data = np.column_stack((x, y))

# Decide which values of lambda to use
nlambdas = 500
lambdas = np.logspace(-3, 5, nlambdas)

colors = ['b', 'g', 'r', 'c', 'm', 'y']

plt.figure()

for idx, k in enumerate(range(5, 11)):
    k_fold_indices = k_fold(data, k)  # Generate fold indices using k_fold function

    # Perform the cross-validation to estimate MSE using KFold
    scores_KFold = np.zeros((nlambdas, k))

    i = 0
    for lmb in lambdas:
        ridge = Ridge(alpha=lmb)
        j = 0
        for train_indices, test_indices in k_fold_indices:
            xtrain, ytrain = data[train_indices, 0], data[train_indices, 1]
            xtest, ytest = data[test_indices, 0], data[test_indices, 1]

            Xtrain = design_matrix(xtrain, ytrain, degree=6)
            ridge.fit(Xtrain, ytrain[:, np.newaxis])

            Xtest = design_matrix(xtest, ytest, degree=6)
            ypred = ridge.predict(Xtest)

            scores_KFold[i, j] = np.sum((ypred - ytest[:, np.newaxis])**2) / np.size(ypred)

            j += 1
        i += 1

    estimated_mse_KFold = np.mean(scores_KFold, axis=1)

    ## Plot the results for custom KFold
    plt.plot(np.log10(lambdas), estimated_mse_KFold, color=colors[idx], label=f'k = {k}')

plt.xlabel('log10(lambda)')
plt.ylabel('mse')
plt.legend()
plt.show()
