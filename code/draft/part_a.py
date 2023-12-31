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
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from tqdm import tqdm
from random import randint

fontsize = 30
sns.set_theme()
params = {
    "font.family": "Serif",
    "font.serif": "Roman", 
    "text.usetex": True,
    "axes.titlesize": fontsize,
    "axes.labelsize": fontsize,
    "xtick.labelsize": "large",
    "ytick.labelsize": "large",
    "legend.fontsize": fontsize
}
plt.rcParams.update(params)

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
    noise_val = np.random.normal(0, 0.1, len(x)*len(y)) 
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

  return np.linalg.pinv(X.T @ X) @ X.T @ y

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
  n = 20 # amount of data points
  degree = np.linspace(0,5, 6, dtype=int)

  #Create x and y values
  x = np.sort(np.random.uniform(0, 1, n))
  y = np.sort(np.random.uniform(0, 1, n))


  x,y = np.meshgrid(x,y)

  x = np.ravel(x)
  y = np.ravel(y)
  #Compute the Franke function with noise 
  f = Franke_function(x,y, noise=True)

  #creating arrays for MSE and R2 score for bothe training and test data  
  Error_train = np.zeros(len(degree))
  Score_train = np.zeros(len(degree))

  Error_test = np.zeros(len(degree))
  Score_test = np.zeros(len(degree))
 
  #creating arrays to chec if our implementation is correct
  Error_train_scitlearn = np.zeros(len(degree))
  Score_train_scitlearn = np.zeros(len(degree))

  Error_test_scitlearn = np.zeros(len(degree))
  Score_test_scitlearn = np.zeros(len(degree))

  beta_val = np.zeros(len(degree), dtype=object)
	
  beta_vals = []
  beta_vals_mean = []
  beta_vals_std = []

  for i in tqdm(range(len(degree))):
    #Create the design matrix
    X = design_matrix(x,y,degree[i])
    X_train, X_test, y_train, y_test = train_test_split(X, f, test_size=0.2)
    #Calculate the beta values
    beta = beta_OLS(X_train,y_train)

    #Calculating the beta mean and std
    beta_vals.append(np.mean(beta, axis= 1))
    beta_vals_degree = np.mean(beta, axis=1)
    beta_vals_mean.append(beta_vals_degree)
    beta_vals_std.append(np.std(beta, axis=1))

    #Crate the model 
    model_train = X_train @ beta
    model_test = X_test @ beta

    #calculating the MSE and R2 score
    Error_train[i] = MSE(y_train, model_train)
    Score_train[i] = R2(y_train, model_train)

    Error_test[i] = MSE(y_test, model_test)
    Score_test[i] = R2(y_test, model_test)

    #beta_val[i] = np.mean(beta, axis= 1)
    
    #calculating with scikitlearn
    model = make_pipeline(PolynomialFeatures(degree=degree[i]), LinearRegression(fit_intercept=False))
    clf = model.fit(X_train,y_train)
    y_fit = clf.predict(X_train)
    y_pred = clf.predict(X_test)

    #calculating the MSE and R2 score
    Error_train_scitlearn[i] = MSE(y_train, y_fit)
    Score_train_scitlearn[i] = R2(y_train, y_fit)

    Error_test_scitlearn[i] = MSE(y_test, y_pred)
    Score_test_scitlearn[i] = R2(y_test, y_pred)


  fig, ax = plt.subplots()
  plot_1 = ax.plot(degree, Error_train, label="MSE Train", color="tab:orange")
  plot_2 = ax.plot(degree, Error_test, label="MSE Test", color="tab:blue")
  ax.set_ylabel("MSE")
  ax.set_xlabel("Degrees")
 

  ax2 = ax.twinx()
  plot_3 = ax2.plot(degree, Score_train, label="R2 Train", color="tab:green")
  plot_4 = ax2.plot(degree, Score_test, label="R2 Test", color="tab:red")
  ax2.set_ylabel("R2 score")
  plt.grid()
  

  lns = plot_1 + plot_2 + plot_3 +plot_4
  labels = [l.get_label() for l in lns]
  plt.legend(lns, labels, loc="center right")
  plt.show()
 

  
  
  plt.plot(degree, Error_train, label="Train")
  plt.plot(degree, Error_test, label="Test")
  plt.ylabel("MSE")
  plt.xlabel("Degree")
  plt.legend()
  plt.show()

  plt.subplot(2,1,1)   
  plt.title("MSE and R2 score as a function of model complexity")     
  plt.plot(degree, Error_train, label="Train")
  plt.plot(degree, Error_test, label="Test")
  plt.ylabel("MSE")

  plt.subplot(2,1,2) 
  plt.plot(degree, Score_train, label="Train")
  plt.plot(degree, Score_test, label="Test")
  plt.ylabel("R2 score")
  plt.xlabel("Degree")
  plt.legend()
  plt.show()

  plt.plot(degree, Error_train_scitlearn, label="Train")
  plt.plot(degree, Error_test_scitlearn, label="Test")
  plt.legend()
  plt.show()

  plt.plot(degree, Score_train_scitlearn, label="Train")
  plt.plot(degree, Score_test_scitlearn, label="Test")
  plt.legend()
  plt.show()

  """
  variance = np.zeros(len(degree))
  #Analysing coefficent
  index = 0


  colors = []

  for i in range(len(degree)):
    colors.append('#%06X' % randint(0, 0xFFFFFF))
  
  for i in range(len(degree)):
      variance[i] = np.var(beta_val[i])
      print(beta_val[i])
      for j in range(len(beta_val[i]) ):
        plt.errorbar(index, np.log10(beta_val[i][j]), yerr=variance[i], fmt="b.", color=colors[i])
        index += 1

  plt.show()
  """
	

  color_palette = sns.color_palette("tab10", len(degree))
	
  # Plotting beta coefficients for different degrees
  plt.figure()
  for i in range(len(degree)):
      plt.plot(range(len(beta_vals[i])), beta_vals[i], marker = "o" ,label=f'Degree {degree[i]}', color=color_palette[i])

  locs, labels = plt.xticks()  # Get the current locations and labels.
  #plt.title(r"Plot of the $\beta$ values for the different polynomial degrees")
  plt.xticks(np.arange(0, 1, step=1))  # Set label locations.
  plt.xticks(np.arange(21), [r'$\beta_0$', r'$\beta_1$', r'$\beta_2$', \
            r'$\beta_3$', r'$\beta_4$', r'$\beta_5$', \
            r'$\beta_6$', r'$\beta_7$', r'$\beta_8$', \
            r'$\beta_9$', r'$\beta_{10}$', r'$\beta_{11}$', \
            r'$\beta_{12}$', r'$\beta_{13}$', r'$\beta_{14}$', \
            r'$\beta_{15}$', r'$\beta_{16}$', r'$\beta_{17}$', \
            r'$\beta_{18}$', r'$\beta_{19}$', r'$\beta_{20}$'\
            ], rotation=45)  # Set text labels.

  plt.xlabel("Coefficient Index")
  plt.ylabel("Beta Coefficient Value")
  plt.legend()
  plt.show()

  # Plotting beta coefficients with error bars
  plt.figure()
  for i in range(len(degree)):
      plt.errorbar(range(len(beta_vals_mean[i])), beta_vals_mean[i], yerr=beta_vals_std[i], fmt='o', label=f'Degree {degree[i]}')
    
  plt.xlabel("Coefficient Index")
  plt.ylabel("Beta Coefficient Value")
  plt.legend()
  plt.show()
	
