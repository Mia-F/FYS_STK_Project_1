import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from matplotlib.ticker import LinearLocator

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error


def franke_function(x, y, noise=False, noise_factor=0.1):
    t1 = 0.75 * np.exp(-(0.25 * (9*x - 2)**2) - (0.25 * (9*y - 2)**2))
    t2 = 0.75 * np.exp(-((9*x + 1)**2 / 49.0) - (0.1 * (9*y + 1)))
    t3 = 0.5 * np.exp(-(0.25 * (9*x - 7)**2) - (0.25 * (9*y - 3)**2))
    t4 = -0.2 * np.exp(-(9*x - 4)**2 - (9*y - 7)**2)
    result = t1 + t2 + t3 + t4

    if noise == True:
        m, n = x.shape
        result += noise_factor * np.random.normal(0.0, 1.0, m*n).reshape(m, n)

    return result.reshape(-1, 1)


def design_matrix(x, y, degree):
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    m = len(x)
    n = int((degree+1) * (degree+2) * 0.5)
    X = np.empty((m, n))
    # X[:, 0] = 1.0
    X = np.ones((m, n))

    for i in range(1, degree+1):
        q = int((i) * (i+1) * 0.5)
        for k in range(i+1):
            X[:, q+k] = x**(i-k) * (y**k)
    
    return X


def beta_ols(X, z):
    try:
        return np.linalg.inv(X.T @ X) @ X.T @ z
    except:
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        return Vt.T @ np.linalg.inv(np.diag(S)) @ U.T @ z.reshape(-1, 1)


def mse(z_data, z_model):
    return np.sum((z_data - z_model)**2) / len(z_data)


def plot_3d(x, y, z):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x, y, z, cmap=cm.winter, linewidth=0, antialiased=False)

    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter()
    fig.colorbar(surf, shrink=0.5, aspect=5)


def bootstrap(x, y, z, degree, num_bootstraps):
    """Resampling data using bootstrap algorithm.
    
    Args:
        x (np.ndarray): x-values
        y (np.ndarray): y-values
        z (np.ndarray): values of Franke function
        degree (int): polynomial degree
        num_bootstraps (int): number of resamples
        
    Returns:
        tuple: Error of train and test, bias and variance
    """
    X = design_matrix(x, y, degree)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

    z_tilde = np.empty((z_train.shape[0], num_bootstraps))
    z_predict = np.empty((z_test.shape[0], num_bootstraps))

    mse_train = np.empty(num_bootstraps)
    mse_test = np.empty(num_bootstraps)

    for i in range(num_bootstraps):
        X_, z_ = resample(X_train, z_train)
        beta = beta_ols(X_, z_)

        z_tilde[:, i] = (X_ @ beta).ravel() 
        z_predict[:, i] = (X_test @ beta).ravel() 

        mse_train[i] = mse(z_, z_tilde[:, i])
        mse_test[i] = mse(z_test, z_predict[:, i])

    error_train = np.mean(mse_train)
    error_test = np.mean(mse_test)
    bias = np.mean((z_test - np.mean(z_predict, axis=1, keepdims=True))**2)
    variance = np.mean(np.var(z_predict, axis=1, keepdims=True))

    return (error_train, error_test, bias, variance)


def bootstrap_sklearn(x, y, z, degree, num_bootstraps, intercept=True):
    """Resampling data using bootstrap algorithm and SKLearn methods.
    
    Args:
        x (np.ndarray): x-values
        y (np.ndarray): y-values
        z (np.ndarray): values of Franke function
        degree (int): polynomial degree
        num_bootstraps (int): number of resamples
        intercept (bool): include intercept 
        
    Returns:
        tuple: Error of train and test, bias and variance
    """
    X = np.column_stack((x.ravel(), y.ravel()))
    # poly = PolynomialFeatures(degree=degree)
    # X = poly.fit_transform(X_)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
    # scaler = StandardScaler()
    # scaler.fit(X_train)
    # X_train_scaled = scaler.transform(X_train)
    # X_test_scaled = scaler.transform(X_test)
    model = make_pipeline(StandardScaler(), PolynomialFeatures(degree=degree), LinearRegression(fit_intercept=intercept))
    
    z_tilde = np.empty((z_train.shape[0], num_bootstraps))
    z_predict = np.empty((z_test.shape[0], num_bootstraps))
    error = np.empty(num_bootstraps)

    for i in range(num_bootstraps):
        X_, z_ = resample(X_train, z_train)
        clf = model.fit(X_, z_)

        z_tilde[:, i] = clf.predict(X_train).ravel() 
        z_predict[:, i] = clf.predict(X_test).ravel() 
        error[i] = mean_squared_error(z_, z_tilde[:, i])


    error_train = np.mean(np.mean((z_train - z_tilde)**2, axis=1, keepdims=True))
    error_test = np.mean(np.mean((z_test - z_predict)**2, axis=1, keepdims=True))
    bias = np.mean((z_test - np.mean(z_predict, axis=1, keepdims=True))**2)
    variance = np.mean(np.var(z_predict, axis=1, keepdims=True))

    return (np.mean(error), error_train, error_test, bias, variance)


if __name__ == '__main__':
    np.random.seed(2023)
    n = 10
    max_degree = 15
    x_ = np.linspace(0, 1, n)
    y_ = np.linspace(0, 1, n)

    x, y = np.meshgrid(x_, y_)
    
    x = np.ravel(x)
    y = np.ravel(y)

    z = franke_function(x, y, noise=True, noise_factor=1)
    #z = z.reshape(-1, 1)
    # plot_3d(x, y, z)

    degrees = np.zeros(max_degree)
    error = np.zeros(max_degree)
    error_train = np.zeros(max_degree)
    error_test = np.zeros(max_degree)
    bias = np.zeros(max_degree)
    variance = np.zeros(max_degree)

    for i in range(max_degree):
        degrees[i] = i + 1
        error_train[i], error_test[i], bias[i], variance[i] = bootstrap(x, y, z, i+1, 100, intercept=False)
        # error[i], error_train[i], error_test[i], bias[i], variance[i] = bootstrap_sklearn(x, y, z, i+1, 1000, intercept=False)

    fig, ax = plt.subplots()
    # ax.plot(degrees, np.log10(error), label='Train sk')
    ax.plot(degrees, np.log10(error_train), label='Train')
    ax.plot(degrees, np.log10(error_test), label='Test')
    ax.plot(degrees, np.log10(bias), 'b--', label='Bias')
    ax.plot(degrees, np.log10(variance), 'g.', label='Variance')
    # ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    plt.show()

