import numpy as np
from imageio.v2 import imread


def franke_function(x, y, noise_factor=0):
    """The Franke function is a bivariate test function.

    Args:
        x (np.ndarray): x-values
        y (np.ndarray): y-values
        noise_factor (int): add noise to function, default is 0

    Returns:
        np.ndarray: array of function values
    """
    noise = 0
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

    if noise_factor != 0:
        noise = np.random.normal(0, 0.1, len(x)*len(y)) 
        noise = noise_factor.reshape(len(x),len(y))

    return term1 + term2 + term3 + term4 + noise_factor*noise


def design_matrix(x, y, degree):
	"""Create design matrix for polynomial degree n with dimension determined
	by the size of input arrays and degree.
	
	Args:
	    x (np.ndarray): x-values, 1D or 2D array
		y (np.ndarray): y-values, 1D or 2D array
        degree (int): order of polynomial degree
		
	Returns:
        np.ndarray: array with shape (len(x)*len(y), degree)
    """
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	# Number of elements in beta
	l = int((degree+1)*(degree+2) / 2)
	X = np.ones((N,l))

	for i in range(1,degree+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)

	return X


def mse(data, model):
	"""Calculate the mean square error of the fit.
	
	Args:
	    data (np.ndarray): data to fit
		model (np.ndarray): predicted model
		
	Returns:
	    np.ndarray: mean square error of fit
	"""
	return np.sum((data - model)**2) / len(data)


def r2(data, model):
	"""Calculate the R2 score of the fit

    Args:
        data (np.ndarray): original data to fit
        model (np.ndarray): predicted model
		
	Returns:
	    np.ndarray: R2 score of fit
    """
	return 1 - np.sum((data - model)**2) / np.sum((data - np.mean(data))**2)


def create_terrain_data(n, file_number):
	"""Load terrain data from file.
	
	Args:
		n (int): number of lines to load from file
        file_number (int): file number to create data from
		
	Returns:
        tuple: arrays of input and output data
	"""
	filename=f'../data/SRTM_data_Norway_{file_number}.tif'
	z = np.zeros((n, n))
	terrain_data = imread(filename)
	
	for i in range(n):
		for j in range(n):
			z[i, j] = terrain_data[i, j]

	# Create x and y
	x_ = np.sort(np.random.uniform(0, 1, n))
	y_ = np.sort(np.random.uniform(0, 1, n))
	x, y= np.meshgrid(x_, y_)
	
	return (x.ravel(), y.ravel(), z.ravel())


def create_function_data(n, noise_factor=0):
	"""Create input data, and output data using the Franke function
	
	Args:
        n (int): number of data points
	    noise_factor (int): add noise to function, default is 0
		
	Returns:
	    tuple: arrays of input and output data
	"""
	x_ = np.sort(np.random.uniform(0, 1, n))
	y_ = np.sort(np.random.uniform(0, 1, n))
	x, y = np.meshgrid(x_, y_)
	
	x, y = x.ravel(), y.ravel()
	z = franke_function(x, y, noise_factor)
	
	return (x, y, z.ravel())