import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures

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
    """
    Perform K-fold cross-validation by splitting the data into K sets.
    Returns:
    List of tuples: Each tuple contains (train_indices, test_indices) for a fold.
    """

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
    
np.random.seed(3155)

# Generate the data.
nsamples = 100
x = np.random.randn(nsamples)
y = 3 * x**2 + np.random.randn(nsamples)
data = np.column_stack((x, y))

k = 5
k_fold_indices = k_fold(data, k)  # Generate fold indices using k_fold function

## Cross-validation on Ridge regression using KFold only

# Decide degree on polynomial to fit
poly = PolynomialFeatures(degree=6)

# Decide which values of lambda to use
nlambdas = 500
lambdas = np.logspace(-3, 5, nlambdas)

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

## Cross-validation using cross_val_score from sklearn along with KFold

estimated_mse_sklearn = np.zeros(nlambdas)

i = 0
for lmb in lambdas:
    ridge = Ridge(alpha=lmb)

    X = design_matrix(x, y, degree=6)
    estimated_mse_folds = cross_val_score(ridge, X, y[:, np.newaxis], scoring='neg_mean_squared_error', cv=kfold)

    # cross_val_score returns an array containing the estimated negative mse for every fold.
    # get an estimate of the mse of the model
    estimated_mse_sklearn[i] = np.mean(-estimated_mse_folds)

    i += 1

## Plot
plt.figure()
plt.plot(np.log10(lambdas), estimated_mse_sklearn, label='cross_val_score (KFold)')
plt.plot(np.log10(lambdas), estimated_mse_KFold, 'r--', label='Custom KFold')
plt.xlabel('log10(lambda)')
plt.ylabel('mse')
plt.legend()
plt.show()