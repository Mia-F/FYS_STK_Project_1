import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.linear_model import Lasso
from tqdm import tqdm

from utils import design_matrix, mse, r2
from methods import beta_ols, beta_ridge


def ols_terrain_regression(x, y, z, degree):
    """Ordinary least squares regression analysis using terrain data, evaluated 
    by mean square error and R2 score.

    Args:
        x (np.ndarray): x-values
        y (np.ndarray): y-values
        z (np.ndarray): z-values
        degree (int): polynomial degree
    Returns:
        tuple: MSE and R2 score of predictions
    """
    X = design_matrix(x, y, degree)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

    beta = beta_ols(X_train, z_train)

    z_tilde = X_train @ beta
    z_predict = X_test @ beta

    mse_train = mse(z_train, z_tilde)
    mse_test = mse(z_test, z_predict)

    r2_train = r2(z_train, z_tilde)
    r2_test = r2(z_test, z_predict)

    return (mse_train, mse_test, r2_train, r2_test)


def ridge_terrain_regression(x, y, z, degree, lambdas):
    """Ridge regression analysis using terrain data, evaluated by mean square 
    error and R2 score.

    Args:
        x (np.ndarray): x-values
        y (np.ndarray): y-values
        z (np.ndarray): z-values
        degree (int): polynomial degree
        lambdas (np.ndarray): lambda values
    Returns:
        tuple: MSE and R2 score of predictions
    """
    n_lambdas = len(lambdas)
    ridge_score = np.zeros((n_lambdas, 4))

    X = design_matrix(x, y, degree)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)


    for j in range(n_lambdas):
        beta = beta_ridge(X_train, z_train, lambdas[j])

        z_tilde = X_train @ beta
        z_predict = X_test @ beta

        ridge_score[j, 0] = mse(z_train, z_tilde)
        ridge_score[j, 1] = mse(z_test, z_predict)
        ridge_score[j, 2] = r2(z_train, z_tilde)
        ridge_score[j, 3] = r2(z_test, z_predict)

    return ridge_score


def lasso_terrain_regression(x, y, z, degree, lambdas):
    """Lasso regression analysis using terrain data, evaluated by mean square 
    error and R2 score.

    Args:
        x (np.ndarray): x-values
        y (np.ndarray): y-values
        z (np.ndarray): z-values
        degree (int): polynomial degree
        lambdas (np.ndarray): lambda values
    Returns:
        tuple: MSE and R2 score of predictions
    """
    n_lambdas = len(lambdas)
    lasso_score = np.zeros((n_lambdas, 4))

    X = design_matrix(x, y, degree)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

    for j in tqdm(range(n_lambdas)):
        model = Lasso(fit_intercept=True, max_iter=5000, alpha=lambdas[j])
        model.fit(X_train, z_train)

        z_tilde = model.predict(X_train)
        z_predict = model.predict(X_test)

        lasso_score[j, 0] = mse(z_train, z_tilde)
        lasso_score[j, 1] = mse(z_test, z_predict)
        lasso_score[j, 2] = r2(z_train, z_tilde)
        lasso_score[j, 3] = r2(z_test, z_predict)

    return lasso_score