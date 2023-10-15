import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import cm
from matplotlib.ticker import LinearLocator

from regression import ols_regression, ridge_regression, lasso_regression
from resampling import cv_ols, cv_ridge, cv_lasso, bootstrap_ols


def plot_3d(x, y, z, name, save):
    """Plot terrain data as 3D surface
    
    Args:
        x (np.ndarray): x-values
        y (np.ndarray): y-values
        z (np.ndarray): z-values
        file_number (int): name of data
        name (str): 
        save (bool): save plot, default is False
        
    Returns:
        None
    """
    colors = sns.color_palette("twilight", as_cmap=True)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x, y, z, cmap=colors, linewidth=0, antialiased=False)

    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter()
    fig.colorbar(surf, shrink=0.5, aspect=5)

    if save is True:
        path = f'../latex/images/terrain_{name}_plot'
        fig.savefig(path)


def ols_plot(x, y, z, max_degree, name, save):
    """Perform ordinary least squares regression and plot MSE and R2 score.
    
    Args:
        x (np.ndarray): x-values
        y (np.ndarray): y-values
        z (np.ndarray): z-values
        max_degree (int): max polynomial degree
        name (str): name of data
        save (bool): save plot, default is False
        
    Returns:
        None
    """
    degrees = np.arange(max_degree)
    ols_score = np.zeros((max_degree, 4))
    for i in range(max_degree):
        ols_score[i, :] = ols_regression(x, y, z, i)

    color = sns.color_palette("tab10")
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

    ax1.plot(degrees, ols_score[:, 0], label='Train', color=color[0])
    ax1.plot(degrees, ols_score[:, 1], label='Test', color=color[1])
    ax1.set_ylabel('MSE')
    ax1.legend(loc='right')

    ax2.plot(degrees, ols_score[:, 2], label='Train', color=color[0])
    ax2.plot(degrees, ols_score[:, 3], label='Test', color=color[1])
    ax2.set_ylabel('R2')
    ax2.legend(loc='right')

    if save is True:
        path = f'../latex/images/ols_{name}_plot'
        fig.savefig(path)


def cv_ols_plot(x, y, z, max_degree, k, name, save):
    """Perform ordinary least squares regression, resampling using cross 
    validation, and plot MSE and R2 score.
    
    Args:
        x (np.ndarray): x-values
        y (np.ndarray): y-values
        z (np.ndarray): z-values
        max_degree (int): max polynomial degree
        k (int): number of folds
        name (str): name of data
        save (bool): save plot, default is False
        
    Returns:
        None
    """
    degrees = np.arange(max_degree)
    ols_score = np.zeros((max_degree, 2))
    for i in range(max_degree):
        ols_score[i, :] = cv_ols(x, y, z, i, k)

    color = sns.color_palette("tab10")
    fig, ax = plt.subplots()

    ax.plot(degrees, ols_score[:, 0], label='Train', color=color[0])
    ax.plot(degrees, ols_score[:, 1], label='Test', color=color[1])
    ax.set_ylabel('MSE')
    ax.legend(loc='right')

    if save is True:
        path = f'../latex/images/ols_{name}_plot'
        fig.savefig(path)


def bootstrap_ols_plot(x, y, z, max_degree, name, save, n_bootstraps=100):
    """Perform ordinary least squares regression, resampling using cross 
    validation, and plot MSE and R2 score.
    
    Args:
        x (np.ndarray): x-values
        y (np.ndarray): y-values
        z (np.ndarray): z-values
        max_degree (int): max polynomial degree
        name (str): name of data
        save (bool): save plot, default is False
        n_bootstraps (int): number of samples
        
    Returns:
        None
    """
    degrees = np.arange(max_degree)
    mse_bias_var = np.zeros((max_degree, 4))
    for i in range(max_degree):
        mse_bias_var[i, :] = bootstrap_ols(x, y, z, i, n_bootstraps)

    color = sns.color_palette("tab10")
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

    ax1.plot(degrees, mse_bias_var[:, 0], label='Train', color=color[0])
    ax1.plot(degrees, mse_bias_var[:, 1], label='Test', color=color[1])
    ax1.set_xticks(np.arange(0, max_degree+1, 2, dtype=np.int32))
    ax1.set_ylabel('MSE (log)')
    ax1.set_yscale('log')
    ax1.legend(loc='upper right')

    ax2.plot(degrees, mse_bias_var[:, 1], label='Error', color=color[1])
    ax2.plot(degrees, mse_bias_var[:, 2], label='Bias', color=color[2])
    ax2.plot(degrees, mse_bias_var[:, 3], label='Variance', color=color[2])
    ax2.set_xticks(np.arange(0, max_degree+1, 2, dtype=np.int32))
    ax2.set_ylabel('MSE (log)')
    ax2.set_yscale('log')
    ax2.legend(loc='upper right')

    if save is True:
        path = f'../latex/images/bias_variance_{name}_plot'
        fig.savefig(path)


def ridge_plot(x, y, z, max_degree, n_lambdas, name, save):
    """Perform Ridge regression and plot MSE and R2 score.
    
    Args:
        x (np.ndarray): x-values
        y (np.ndarray): y-values
        z (np.ndarray): z-values
        max_degree (int): max polynomial degree
        n_lambdas (int): number of lambda values
        name (str): name of data
        save (bool): save plot
        
    Returns:
        None
    """
    degrees = np.arange(max_degree)
    lambdas = np.zeros(n_lambdas)
    for d in range(n_lambdas):
        lambdas[d] = 10**(-d)

    ridge_score = np.zeros((max_degree, n_lambdas, 4))
    for i in range(max_degree):
        ridge_score[i, :, :] = ridge_regression(x, y, z, i, lambdas)

    color = sns.color_palette("tab10", n_lambdas)
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

    for j in range(n_lambdas):
        # Plot MSE
        # ax1.plot(degrees, ridge_score[:, j, 0], label='Train', color=color[j])
        ax1.plot(
            degrees, 
            ridge_score[:, j, 1], 
            label=rf'$10^{{{-j}}}$', 
            color=color[j])

        # Plot R2
        # ax2.plot(degrees, ridge_score[:, j, 2], label='Train', color=color[j])
        ax2.plot(
            degrees, 
            ridge_score[:, j, 3], 
            label=rf'$10^{{{-j}}}$', 
            color=color[j])
        
    ax1.set_ylabel('MSE')
    ax1.legend(loc='right', title=r'$\lambda_{Ridge}$')
    ax2.set_ylabel('R2')
    ax2.legend(loc='right', title=r'$\lambda_{Ridge}$')

    if save is True:
        filename = f'../latex/images/ridge_{name}_plot'
        fig.savefig(filename)


def cv_ridge_plot(x, y, z, max_degree, k, n_lambdas, name, save):
    """Perform Ridge regression, resampling using cross validation and plot MSE 
    and R2 score.
    
    Args:
        x (np.ndarray): x-values
        y (np.ndarray): y-values
        z (np.ndarray): z-values
        max_degree (int): max polynomial degree
        k (int): number of folds
        n_lambdas (int): number of lambda values
        name (str): name of data
        save (bool): save plot
        
    Returns:
        None
    """
    degrees = np.arange(max_degree)
    lambdas = np.zeros(n_lambdas)
    for d in range(n_lambdas):
        lambdas[d] = 10**(-d)

    ridge_score = np.zeros((max_degree, n_lambdas, 2))
    for i in range(max_degree):
        for j in range(n_lambdas):
            ridge_score[i, :, :] = cv_ridge(x, y, z, i, k, lambdas[j])

    color = sns.color_palette("tab10", n_lambdas)
    fig, ax = plt.subplots()

    for j in range(n_lambdas):
        # Plot MSE
        # ax1.plot(degrees, ridge_score[:, j, 0], label='Train', color=color[j])
        ax.plot(
            degrees, 
            ridge_score[:, j, 1], 
            label=rf'$10^{{{-j}}}$', 
            color=color[j])
        
    ax.set_ylabel('MSE')
    ax.legend(loc='right', title=r'$\lambda_{Ridge}$')

    if save is True:
        filename = f'../latex/images/ridge_{name}_plot'
        fig.savefig(filename)


def lasso_plot(x, y, z, max_degree, n_lambdas, name, save):
    """Perform Lasso regression and plot MSE and R2 score.
    
    Args:
        x (np.ndarray): x-values
        y (np.ndarray): y-values
        z (np.ndarray): z-values
        max_degree (int): max polynomial degree
        n_lambdas (int): number of lambda values
        name (str): name of data
        save (bool): save plot
        
    Returns:
        None
    """
    degrees = np.arange(max_degree)
    lambdas = np.zeros(n_lambdas)
    for d in range(n_lambdas):
        lambdas[d] = 10**(-d)

    lasso_score = np.zeros((max_degree, n_lambdas, 4))
    for i in range(max_degree):
        lasso_score[i, :, :] = lasso_regression(x, y, z, i, lambdas)

    color = sns.color_palette("tab10", n_lambdas)
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

    for j in range(n_lambdas):
        # Plot MSE
        # ax1.plot(degrees, lasso_score[:, j, 0], label='Train', color=color[j])
        ax1.plot(
            degrees, 
            lasso_score[:, j, 1], 
            label=rf'$10^{{{-j}}}$', 
            color=color[j])

        # Plot R2
        # ax2.plot(degrees, lasso_score[:, j, 2], label='Train', color=color[j])
        ax2.plot(
            degrees, 
            lasso_score[:, j, 3], 
            label=rf'$10^{{{-j}}}$', 
            color=color[j])
        
    ax1.set_ylabel('MSE')
    ax1.legend(loc='right', title=r'$\lambda_{Lasso}$')
    ax2.set_ylabel('R2')
    ax2.legend(loc='right', title=r'$\lambda_{Lasso}$')

    if save is True:
        filename = f'../latex/images/lasso_{name}_plot'
        fig.savefig(filename)


def cv_lasso_plot(x, y, z, max_degree, k, n_lambdas, name, save):
    """Perform Lasso regression and plot MSE and R2 score.
    
    Args:
        x (np.ndarray): x-values
        y (np.ndarray): y-values
        z (np.ndarray): z-values
        max_degree (int): max polynomial degree
        k (int): number of folds
        n_lambdas (int): number of lambda values
        name (str): name of data
        save (bool): save plot
        
    Returns:
        None
    """
    degrees = np.arange(max_degree)
    lambdas = np.zeros(n_lambdas)
    for d in range(n_lambdas):
        lambdas[d] = 10**(-d)

    lasso_score = np.zeros((max_degree, n_lambdas, 2))
    for i in range(max_degree):
        for j in range(n_lambdas):
            lasso_score[i, :, :] = cv_lasso(x, y, z, i, k, lambdas[j])

    color = sns.color_palette("tab10", n_lambdas)
    fig, ax = plt.subplots()

    for j in range(n_lambdas):
        # Plot MSE
        # ax.plot(degrees, lasso_score[:, j, 0], label='Train', color=color[j])
        ax.plot(
            degrees, 
            lasso_score[:, j, 1], 
            label=rf'$10^{{{-j}}}$', 
            color=color[j])
        
    ax.set_ylabel('MSE')
    ax.legend(loc='right', title=r'$\lambda_{Lasso}$')

    if save is True:
        filename = f'../latex/images/lasso_{name}_plot'
        fig.savefig(filename)