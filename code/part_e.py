import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from imageio.v2 import imread

from matplotlib import cm
from matplotlib.ticker import LinearLocator

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error

from matplotlib import cm
from tqdm import tqdm

# from part_a import Franke_function, design_matrix

sns.set_theme()

def Franke_function(x,y, noise_factor=0):
    """
    Franke_function returns an array with dimension (len(x), len(y)), if noise is set to true
    the output will contain noise given by a gaussian distribution N(0,1)

    :x: is an array containing all the x values it can be a one dimensional array or a 2D array
    :y: is an array containing all the y values it can be a one dimensional array or a 2D array
    """
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    noise_val = np.random.normal(0, 0.1, len(x)*len(y)) 
    noise_val = noise_val.reshape(len(x),len(y))
    return term1 + term2 + term3 + term4 + noise_factor*noise_val


def design_matrix(x,y,degree):
	"""
	design_matrix create the design matrix for a polynomial of degree n with dimension (len(x)*len(y), degree)
	
	:x: is an array containing all the x values it can be a 1D array or a 2D array
  :y: is an array containing all the y values it can be a 1D array or a 2D array
  :degree: is the polynomial degree of the fit
  """
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((degree+1)*(degree+2)/2)		# Number of elements in beta
	X = np.ones((N,l))

	for i in range(1,degree+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)

	return X


def beta_OLS(X, y):
  """
  beta_OLS calculates the beta values either with matrix inversion or if not possible it uses 
  singular value decomposition, returns an array with dimension (degree, )

  :X: is the disign matrix with dimension (len(x)*len(y), degree)
  :y: is the data we whant to fit with dimension (len(x), len(y))
  """

  return np.linalg.pinv(X.T @ X) @ X.T @ y


def beta_ridge(X, y, lamb):
    return (np.linalg.pinv(X.T @ X + lamb * np.identity(len(X[0]))) @ X.T @ y)


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


def d_matrix(x, y, degree):
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


def mse(z_data, z_model):
    return np.sum((z_data - z_model)**2) / len(z_data)


def plot_3d(x, y, z):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x, y, z, cmap=cm.winter, linewidth=0, antialiased=False)

    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter()
    fig.colorbar(surf, shrink=0.5, aspect=5)


def bootstrap_reshape(x, y, z, degree, num_bootstraps):
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
        beta = beta_OLS(X_, z_)
        # beta = beta_ols(X_, z_)

        z_tilde[:, i] = (X_ @ beta).ravel() 
        z_predict[:, i] = (X_test @ beta).ravel() 

        mse_train[i] = mse(z_, z_tilde[:, i])
        mse_test[i] = mse(z_test, z_predict[:, i])

    error_train = np.mean(mse_train)
    error_test = np.mean(mse_test)
    bias = np.mean((z_test - np.mean(z_predict, axis=1, keepdims=True))**2)
    variance = np.mean(np.var(z_predict, axis=1, keepdims=True))

    return (error_train, error_test, bias, variance)


def bootstrap_ols(x, y, z, degree, num_bootstraps):
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
    X_train, X_test, z_train, z_test = train_test_split(X, z.flatten(), test_size=0.2)

    z_tilde = np.empty((z_train.shape[0], num_bootstraps))
    z_predict = np.empty((z_test.shape[0], num_bootstraps))

    mse_train = np.empty(num_bootstraps)
    mse_test = np.empty(num_bootstraps)

    for i in range(num_bootstraps):
        X_, z_ = resample(X_train, z_train)
        beta = beta_OLS(X_, z_)
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


def bootstrap_ridge(X_train, X_test, z_train, z_test, lamb, num_bootstraps):
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
    z_tilde = np.empty((z_train.shape[0], num_bootstraps))
    z_predict = np.empty((z_test.shape[0], num_bootstraps))

    mse_train = np.empty(num_bootstraps)
    mse_test = np.empty(num_bootstraps)

    for i in range(num_bootstraps):
        X_, z_ = resample(X_train, z_train)
        beta = beta_ridge(X_, z_, lamb)
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
    X_train, X_test, z_train, z_test = train_test_split(X, z.flatten(), test_size=0.2)
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


def creating_data(data_file, n):
    new_data = np.zeros((n,n))
    terrain1 = imread(data_file)
    for i in tqdm(range(n)):
        for j in range(n):
            new_data[i][j] = terrain1[i][j]
    return new_data

    
def load_real_data(n):
    # Load the terrain
    filename = '../DataFiles/SRTM_data_Norway_1.tif'
    data = creating_data(filename, n)

    #creating x and y
    n,m = data.shape

    x = np.sort(np.random.uniform(0, 1, n))
    y = np.sort(np.random.uniform(0, 1, m))

    x,y = np.meshgrid(x,y)


    x_1D = x.ravel()
    y_1D = y.ravel()
    z = data.ravel()

    return (x_1D, y_1D, z)


def ols_error_plot(x, y, z, max_degree):
    degrees = np.zeros(max_degree)
    error_train = np.zeros(max_degree)
    error_test = np.zeros(max_degree)
    bias = np.zeros(max_degree)
    variance = np.zeros(max_degree)

    for i in range(max_degree):
        degrees[i] = i + 1
        error_train[i], error_test[i], bias[i], variance[i] = bootstrap_ols(x, y, z, i+1, 100)

    fig, ax = plt.subplots()
    ax.plot(degrees, error_train, label='Train error')
    ax.plot(degrees, error_test, label='Test error')
    ax.plot(degrees, bias, 'b--', label='Bias')
    ax.plot(degrees, variance, 'r--', label='Variance')
    ax.set_yscale('log')
    ax.legend()

if __name__ == '__main__':
    np.random.seed(2023)
    n = 20
    max_degree = 15
    degrees = np.arange(1, max_degree+1, 1, dtype=np.int32)

    x_ = np.linspace(0, 1, n)
    y_ = np.linspace(0, 1, n)
    x, y = np.meshgrid(x_, y_)

    # Franke
    x.ravel()
    y.ravel()
    z = Franke_function(x, y, noise_factor=1)

    # Real data
    # x, y, z = load_real_data(n)

    # degrees = np.zeros(max_degree)
    error = np.zeros(max_degree)
    error_train = np.zeros(max_degree)
    error_test = np.zeros(max_degree)
    bias = np.zeros(max_degree)
    variance = np.zeros(max_degree)


    for i in range(max_degree):
        error_train[i], error_test[i], bias[i], variance[i] = bootstrap_ols(x, y, z, i+1, 100)
        # error_train[i], error_test[i], bias[i], variance[i] = bootstrap(x, y, z, i+1, 100)
        # error_train[i], error_test[i], bias[i], variance[i] = bootstrap_reshape(x, y, z, i+1, 100)
        # error[i], error_train[i], error_test[i], bias[i], variance[i] = bootstrap_sklearn(x, y, z, i+1, 100, intercept=False)

    # error_sum = bias + variance
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    ax1.plot(degrees, error_train, label='Train error')
    ax1.plot(degrees, error_test, label='Test error')

    ax2.plot(degrees, error_test, label='Error')
    ax2.plot(degrees, bias, 'b--', label='Bias')
    ax2.plot(degrees, variance, 'r--', label='Variance')
    # ax.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xticks(np.arange(0, max_degree+1, 2, dtype=np.int32))
    ax1.legend()
    fig1.savefig("../LaTeX/Images/bootstrap_error.png")
    ax2.set_yscale('log')
    ax2.set_xticks(np.arange(0, max_degree+1, 2, dtype=np.int32))
    ax2.legend()
    fig2.savefig("../LaTeX/Images/bias_variance.png")

    # lambdas = np.logspace(-8, 0, 9)
    # l = len(lambdas)

    # error_lamb = np.zeros((len(degrees), l))
    # error_train_lamb = np.zeros((len(degrees), l))
    # error_test_lamb = np.zeros((len(degrees), l))
    # bias_lamb = np.zeros((len(degrees), l))
    # variance_lamb = np.zeros((len(degrees), l))

    # # Ridge bootstrap
    # for i in range(max_degree):
    #     X = design_matrix(x, y, int(degrees[i]))
    #     X_train, X_test, z_train, z_test = train_test_split(X, z.flatten(), test_size=0.2)

    #     for j in tqdm(range(l)):
    #         error_train_lamb[i, j], error_test_lamb[i, j], bias_lamb[i, j], variance_lamb[i, j] = bootstrap_ridge(X_train, X_test, z_train, z_test, lambdas[j], 100)

    # e_train = np.zeros(max_degree)

    # fig, ax = plt.subplots()    
    # for j in range(l):
    #     # e_train = error_train_lamb[:, j]
    #     ax.plot(degrees, error_train_lamb[:, j], label=f'Train error {lambdas[j]}')
    #     ax.plot(degrees, error_test_lamb[:, j], label=f'Test error {lambdas[j]}')
    #     ax.plot(degrees, bias_lamb[:, j], label=f'Bias {lambdas[j]}')
    #     ax.plot(degrees, variance_lamb[:, j], label=f'Variance {lambdas[j]}')
    #     # ax.set_xscale('log')
    # ax.set_yscale('log')
    # ax.legend()
    plt.show()