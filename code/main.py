import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils import create_function_data, create_terrain_data
from plot import plot_3d
from plot import ols_plot, cv_ols_plot, bootstrap_ols_plot
from plot import ridge_plot, cv_ridge_plot
from plot import lasso_plot, cv_lasso_plot


sns.set_theme()
params = {
    "font.family": "Serif",
    "font.serif": "Roman", 
    "text.usetex": True,
    "axes.titlesize": "large",
    "axes.labelsize": "large",
    "xtick.labelsize": "large",
    "ytick.labelsize": "large",
    "legend.fontsize": "medium"
}
plt.rcParams.update(params)


def plot_data_surface(
        n_points, 
        file_number=1,
        save=False):
    """Perform OLS, Ridge, and Lasso regression, then plot MSE and R2 score for
    the franke test data.
    
    Args:
        n_points (int): number of data points
        file_number (int): terrain data set number
        save (bool): save plot, default is False
        
    Returns:
        None
    """
    x, y, z = create_function_data(n_points)
    plot_3d(x, y, z, 'franke', save)

    x, y, z = create_function_data(n_points, noise_factor=1)
    plot_3d(x, y, z, 'franke_noise', save)

    x, y, z = create_terrain_data(n_points, file_number)
    plot_3d(x, y, z, 'terrain', save)

    if save is False:
        plt.show()


def plot_franke_regression(
        max_degree, 
        n_lambdas, 
        n_points, 
        noise=False, 
        save=False):
    """Perform OLS, Ridge, and Lasso regression, then plot MSE and R2 score for
    the franke test data.
    
    Args:
        max_degree (int): max polynomial degree
        n_lambdas (int): number of lambda values
        n_points (int): number of data points
        noise (bool): add noise to franke function, default is False
        save (bool): save plot, default is False
        
    Returns:
        None
    """
    if noise is True:
        x, y, z = create_function_data(n_points, noise_factor=1)
        noise = '_noise'
    else:
        x, y, z = create_function_data(n_points)
        noise = ''

    name = f'franke{noise}'
    ols_plot(x, y, z, max_degree, name, save)
    ridge_plot(x, y, z, max_degree, n_lambdas, name, save)
    lasso_plot(x, y, z, max_degree, n_lambdas, name, save)

    if save is False:
        plt.show()


def plot_franke_regression_cv(
        max_degree, 
        k, 
        n_lambdas, 
        n_points, 
        noise=False, 
        save=False):
    """Perform OLS, Ridge, and Lasso regression, with resampling of franke data 
    using cross validation, then plot MSE and R2 score.
    
    Args:
        max_degree (int): max polynomial degree
        k (int): number of folds
        n_lambdas (int): number of lambda values
        n_points (int): number of data points
        noise (bool): add noise to franke function, default is False
        save (bool): save plot, default is False
        
    Returns:
        None
    """
    if noise is True:
        x, y, z = create_function_data(n_points, noise_factor=1)
        noise = '_noise'
    else:
        x, y, z = create_function_data(n_points)
        noise = ''

    name = f'franke{noise}_cv'
    cv_ols_plot(x, y, z, max_degree, k, name, save)
    cv_ridge_plot(x, y, z, max_degree, k, n_lambdas, name, save)
    cv_lasso_plot(x, y, z, max_degree, k, n_lambdas, name, save)

    if save is False:
        plt.show()


def plot_franke_bootstrap(
        max_degree, 
        n_points, 
        noise=False, 
        save=False):
    """Perform OLS, Ridge, and Lasso regression, then plot MSE and R2 score for
    the franke test data.
    
    Args:
        max_degree (int): max polynomial degree
        n_points (int): number of data points
        noise (bool): add noise to franke function, default is False
        save (bool): save plot, default is False
        
    Returns:
        None
    """
    if noise is True:
        x, y, z = create_function_data(n_points, noise_factor=1)
        noise = '_noise'
    else:
        x, y, z = create_function_data(n_points)
        noise = ''

    name = f'franke{noise}'
    bootstrap_ols_plot(x, y, z, max_degree, name, save)

    if save is False:
        plt.show()


def plot_terrain_regression(
        max_degree, 
        n_lambdas, 
        n_points,
        file_number=1,
        save=False):
    """Perform OLS, Ridge, and Lasso regression, then plot MSE and R2 score for
    the terrain test data.
    
    Args:
        max_degree (int): max polynomial degree
        n_lambdas (int): number of lambda values
        n_points (int): number of data points
        file_number (int): terrain data set number
        save (bool): save plot, default is False
        
    Returns:
        None
    """
    x_, y_, z_ = create_terrain_data(n_points, file_number)

    # x = (x_ - np.min(x_)) / (np.max(x_) - np.min(x_))
    # y = (y_ - np.min(y_)) / (np.max(y_) - np.min(y_))
    z = (z_ - np.min(z_)) / (np.max(z_) - np.min(z_))

    name = f'terrain_{file_number}'

    ols_plot(x_, y_, z, max_degree, name, save)
    ridge_plot(x_, y_, z, max_degree, n_lambdas, name, save)
    lasso_plot(x_, y_, z, max_degree, n_lambdas, name, save)

    if save is False:
        plt.show()

    
def plot_terrain_regression_cv(
    max_degree, 
    k, 
    n_lambdas, 
    n_points, 
    file_number=1,
    save=False):
    """Perform OLS, Ridge, and Lasso regression, with resampling of terrain data 
    using cross validation, then plot MSE and R2 score.
    
    Args:
        max_degree (int): max polynomial degree
        k (int): number of folds
        n_lambdas (int): number of lambda values
        n_points (int): number of data points
        file_number (int): terrain data set number
        save (bool): save plot, default is False
        
    Returns:
        None
    """
    x_, y_, z_ = create_terrain_data(n_points, file_number)

    # x = (x_ - np.min(x_)) / (np.max(x_) - np.min(x_))
    # y = (y_ - np.min(y_)) / (np.max(y_) - np.min(y_))
    z = (z_ - np.min(z_)) / (np.max(z_) - np.min(z_))

    name = f'terrain_{file_number}_cv'

    cv_ols_plot(x_, y_, z, max_degree, k, name, save)
    cv_ridge_plot(x_, y_, z, max_degree, k, n_lambdas, name, save)
    cv_lasso_plot(x_, y_, z, max_degree, k, n_lambdas, name, save)

    if save is False:
        plt.show()


def plot_terrain_bootstrap(
        max_degree, 
        n_points, 
        file_number=1, 
        save=False):
    """Perform OLS, Ridge, and Lasso regression, then plot MSE and R2 score for
    the franke test data.
    
    Args:
        max_degree (int): max polynomial degree
        n_points (int): number of data points
        noise (bool): add noise to franke function, default is False
        save (bool): save plot, default is False
        
    Returns:
        None
    """
    x_, y_, z_ = create_terrain_data(n_points, file_number)

    # x = (x_ - np.min(x_)) / (np.max(x_) - np.min(x_))
    # y = (y_ - np.min(y_)) / (np.max(y_) - np.min(y_))
    z = (z_ - np.min(z_)) / (np.max(z_) - np.min(z_))

    name = f'terrain_{file_number}'

    bootstrap_ols_plot(x_, y_, z, max_degree, name, save)

    if save is False:
        plt.show()


def main():
    np.random.seed(2023)
    # plot_data_surface(20, save=True)
    plot_franke_regression(6, 6, 20, save=True)
    plot_terrain_regression(6, 6, 20, save=True)
    plot_franke_regression_cv(6, 5, 6, 20, save=True)
    plot_terrain_regression_cv(6, 5, 6, 20, save=True)
    plot_franke_bootstrap(16, 20, save=True)
    plot_terrain_bootstrap(16, 20, 1, save=True)


if __name__ == '__main__':
    main()
    