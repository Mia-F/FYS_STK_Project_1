import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.utils import resample

from utils import mse, design_matrix
from methods import beta_ols, beta_ridge, k_fold


def cv_ols(x, y, z, degree, k):
    X = design_matrix(x, y, degree)
    scores_KFold_train = np.zeros(k)
    scores_KFold_test = np.zeros(k)
    k_fold_indices = k_fold(X, k)

    for j, (train_indices, test_indices) in enumerate(k_fold_indices):
        X_train, X_test = X[train_indices], X[test_indices]
        z_train, z_test = z[train_indices], z[test_indices]

        beta = beta_ols(X_train, z_train)

        z_tilde = (X_train @ beta).ravel()
        z_predict = (X_test @ beta).ravel()
        scores_KFold_train[j] = mse(z_train, z_tilde)
        scores_KFold_test[j] = mse(z_test, z_predict)
    mse_train_ols = np.mean(scores_KFold_train)
    mse_test_ols = np.mean(scores_KFold_test)

    return (mse_train_ols, mse_test_ols)


def cv_ridge(x, y, z, degree, k, lamb):
    X = design_matrix(x, y, degree)
    scores_KFold_train = np.zeros(k)
    scores_KFold_test = np.zeros(k)
    k_fold_indices = k_fold(X, k)

    for j, (train_indices, test_indices) in enumerate(k_fold_indices):
        X_train, X_test = X[train_indices], X[test_indices]
        z_train, z_test = z[train_indices], z[test_indices]

        beta = beta_ridge(X_train, z_train, lamb)
        z_tilde = (X_train @ beta).ravel()
        z_predict = (X_test @ beta).ravel()

        scores_KFold_train[j] = mse(z_train.flatten(), z_tilde)
        scores_KFold_test[j] = mse(z_test.flatten(), z_predict)

    mse_train_ridge = np.mean(scores_KFold_train)
    mse_test_ridge = np.mean(scores_KFold_test)

    return (mse_train_ridge, mse_test_ridge)


def cv_lasso(x, y, z, degree, k, lamb):
    X = design_matrix(x, y, degree)
    scores_KFold_train = np.zeros(k)
    scores_KFold_test = np.zeros(k)
    k_fold_indices = k_fold(X, k)
    
    for j, (train_indices, test_indices) in enumerate(k_fold_indices):
        X_train, X_test = X[train_indices], X[test_indices]
        z_train, z_test = z[train_indices], z[test_indices]
        
        lasso_reg = Lasso(alpha=lamb, fit_intercept=False, max_iter=5000)
        lasso_reg.fit(X_train, z_train)

        z_tilde = lasso_reg.predict(X_train)
        z_predict = lasso_reg.predict(X_test)
        
        scores_KFold_train[j] = mse(z_train, z_tilde)
        scores_KFold_test[j] = mse(z_test, z_predict)

    mse_train_lasso = np.mean(scores_KFold_train)
    mse_test_lasso = np.mean(scores_KFold_test)

    return (mse_train_lasso, mse_test_lasso)


def bootstrap_ols(x, y, z, degree, n_bootstraps):
    """Resampling data using bootstrap algorithm.
    
    Args:
        x (np.ndarray): x-values
        y (np.ndarray): y-values
        z (np.ndarray): z-values
        degree (int): polynomial degree
        n_bootstraps (int): number of samples
        
    Returns:
        tuple: Error of train and test, bias and variance
    """
    X = design_matrix(x, y, degree)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

    z_tilde = np.empty((z_train.shape[0], n_bootstraps))
    z_predict = np.empty((z_test.shape[0], n_bootstraps))

    mse_train = np.empty(n_bootstraps)
    mse_test = np.empty(n_bootstraps)

    for i in range(n_bootstraps):
        X_, z_ = resample(X_train, z_train)
        beta = beta_ols(X_, z_)
        # beta = beta_ols(X_, z_)

        z_tilde[:, i] = (X_ @ beta).ravel() 
        z_predict[:, i] = (X_test @ beta).ravel() 

        mse_train[i] = mse(z_, z_tilde[:, i])
        mse_test[i] = mse(z_test, z_predict[:, i])

    error_train = np.mean(mse_train)
    error_test = np.mean(mse_test)
    bias = np.mean((z_test - np.mean(z_predict, axis=1))**2)
    variance = np.mean(np.var(z_predict, axis=1, keepdims=True))

    return (error_train, error_test, bias, variance)


def bootstrap_ridge():
    pass


def bootstrap_lasso():
    pass
