"""
Implementing OLS regression
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sklearn.linear_model as skl
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

sns.set_theme()

def Franke_function(x,y, noise=False):
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
  if noise==True:
    noise_val = np.random.normal(0, 1, len(x)*len(y)) 
    noise_val = noise_val.reshape(len(x),len(y))
    return term1 + term2 + term3 + term4 + noise_val
  else:
    return term1 + term2 + term3 + term4


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
  try: 
    # check if the design matrix is invertible
    return np.linalg.inv(X.T @ X) @ X.T @ y 
  except: 
    # If the design matrix is not invertible we do a singular value decomposition
    U,S,Vt = np.linalg.svd(X, full_matrices=False)
    return Vt.T @ np.linalg.inv(np.diag(S)) @ U.T @ y.flatten()

def MSE(y_data, y_model):
  """
  MSE calculates the mean square error of the fit

  :y_data: the original data we want to fit, with dimension (len(x), len(y))
  :y_model: the model we have createt from y_data, with dimension (len(x), len(y))
  """
  return 1/len(y_data) * np.sum((y_data - y_model)**2)

def R2(y_data, y_model):
  """
  R2 calculates the R2 score for the fit

  :y_data: the original data we want to fit, with dimension (len(x), len(y))
  :y_model: the model we have createt from y_data, with dimension (len(x), len(y))
  """
  return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)

def plotting(x, y , y_model):
  fig = plt.figure(figsize=(20,8))
  ax = fig.gca(projection ='3d')
  surf = ax.plot_surface(x,y,y_model,cmap=cm.twilight, linewidth = 0, antialiased=False)
  ax.set_zlim(-0.10,1.40)
  ax.set_xlabel('X-axis', fontsize=30)
  ax.set_ylabel('Y-axis', fontsize=30)
  ax.set_zlabel('Z-axis', fontsize=30)
  ax.zaxis.set_major_locator(LinearLocator(10))
  ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
  ax.zaxis.labelpad = 12
  ax.xaxis.labelpad = 25
  ax.yaxis.labelpad = 20
  ax.tick_params(axis='x', labelsize=20)
  ax.tick_params(axis='y', labelsize=20)
  ax.tick_params(axis='z', labelsize=20)
  plt.show()
    


if __name__ == "__main__":
  np.random.seed(2023)
  n = 100 # amount of data points
  degree = np.linspace(1,15, 15, dtype=int)

  #Create x and y values
  x = np.linspace(0,1, n)
  y = x
  #Create a meshgrid of x and y so we later can plot in 3D
  x,y = np.meshgrid(x,y)

  #creating arrays to append 
  mse_error_OLS_train = np.zeros(len(degree))
  r2_score_train = np.zeros(len(degree))
  mse_error_OLS_test = np.zeros(len(degree))
  r2_score_test = np.zeros(len(degree))

  #Compute the Franke function with noise 
  f = Franke_function(x,y, noise=True)

  fig, ax = plt.subplots()
  for d in degree:
    #Create the design matrix
    X = design_matrix(x,y,d)
    X_train, X_test, y_train, y_test = train_test_split(X, f.flatten(), test_size=0.2)
    #Calculate the beta values
    beta = beta_OLS(X_train,y_train)

    #Crate the model 
    model_train = X_train @ beta
    model_test = X_test @ beta

    #calculating the MSE and R2 score
    mse_error_OLS_train[d-1] = MSE(y_train, model_train)
    r2_score_train[d-1] = R2(y_train, model_train)

    mse_error_OLS_test[d-1] = MSE(y_test, model_test)
    r2_score_test[d-1] = R2(y_test, model_test)

    if d-1 == 0:
      ax.scatter(degree[d-1], mse_error_OLS_train[d-1], color="tab:orange", label="MSE training")
      ax.scatter(degree[d-1], r2_score_train[d-1], color="tab:blue", label="R2 score training")

      ax.scatter(degree[d-1], mse_error_OLS_test[d-1], color="tab:green", label="MSE testing")
      ax.scatter(degree[d-1], r2_score_test[d-1], color="purple", label="R2 score testing")
    else:
      ax.scatter(degree[d-1], mse_error_OLS_train[d-1], color="tab:orange")
      ax.scatter(degree[d-1], r2_score_train[d-1], color="tab:blue")

      ax.scatter(degree[d-1], mse_error_OLS_test[d-1], color="tab:green")
      ax.scatter(degree[d-1], r2_score_test[d-1], color="purple")
        
  
  ax.set_title("MSE and R2 scores for fitts with different polynomial degrees")
  ax.set_xlabel("Degree")
  ax.set_ylabel("MSE and R2 values")
  ax.legend() 
  plt.show()

  fig, ax = plt.subplots()
  ax.scatter(x[0], (X @ beta)[0:100])
  ax.plot(x[0], f[0])
  plt.show()



  
