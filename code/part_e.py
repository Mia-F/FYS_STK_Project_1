import numpy as np
from part_a import Franke_function, design_matrix, MSE, R2, beta_OLS
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from sklearn.model_selection import train_test_split
import time
from tqdm import tqdm

np.random.seed(2023)

def bootstrap(data):
    """
    This function takes inn data and reshuffels it in to a "new" dataset 
    with the same lenght.

    :data: array containing the data points we whant to reshuffel
    """
    datapoints = len(data)
    resampled_data = np.zeros(datapoints) 

    for i in range(datapoints):
        resampled_data[i] = np.mean(data[np.random.randint(0,datapoints,datapoints)])
    #print("Bootstrap Statistics :")
    #print("original           bias      std. error")
    #print("%8g %8g %14g %15g" % (np.mean(data), np.std(data),np.mean(resampled_data),np.std(resampled_data)))
    return resampled_data

np.random.seed(2023)

n = 100 # amount of data points
degree = np.linspace(0,20,21, dtype=int)
#Set the amount of reshuffeling
n_bootsrap = 5


mean_mean_square_error_train = np.zeros(len(degree))
mean_bias_train = np.zeros(len(degree))
mean_variance_train = np.zeros(len(degree))

mean_mean_square_error_test = np.zeros(len(degree))
mean_bias_test = np.zeros(len(degree))
mean_variance_test = np.zeros(len(degree))

#Create x and y values
x = np.linspace(0,1, n)
y = x
#Create a meshgrid of x and y so we later can plot in 3D
x,y = np.meshgrid(x,y)

#Compute the Franke function with noise 
f = Franke_function(x,y, noise=True)


for d in tqdm(range(len(degree))):
    X = design_matrix(x,y,degree[d])

    #Splitting data in to test and train
    X_train, X_test, y_train, y_test = train_test_split(X, f.flatten(), test_size=0.2) 
    
    mean_square_error_train = np.zeros(n_bootsrap)
    bias_train = np.zeros(n_bootsrap)
    variance_train = np.zeros(n_bootsrap)

    mean_square_error_test = np.zeros(n_bootsrap)
    bias_test = np.zeros(n_bootsrap)
    variance_test = np.zeros(n_bootsrap)
    
    for i in range(n_bootsrap):
        y_ = bootstrap(y_train)
        y_test_ = bootstrap(y_test)
    
        #Calculate the beta values
        beta = beta_OLS(X_train, y_)
        model = X_train @ beta 
        model_test = X_test @ beta

        mean_square_error_train[i] = np.mean((y_ - model)**2)
        bias_train[i] = np.mean(((y_ - np.mean(model))**2))
        variance_train[i] = np.mean(np.var(model))

        mean_square_error_test[i] = np.mean((y_test_ - model_test)**2)
        bias_test[i] = np.mean((y_test_ - np.mean(model_test))**2)
        variance_test[i] = np.mean(np.var(model_test))

    mean_mean_square_error_train[d] = np.mean(mean_square_error_train)
    mean_bias_train[d] = np.mean(bias_train)
    mean_variance_train[d] = np.mean(variance_train)

    mean_mean_square_error_test[d] = np.mean(mean_square_error_test)
    mean_bias_test[d] = np.mean(bias_test)
    mean_variance_test[d] = np.mean(variance_test)


plt.plot(degree, mean_mean_square_error_train, label="Error")
plt.plot(degree, mean_bias_train , label= "Bias")
plt.plot(degree, mean_variance_train, label="Variance")
plt.xlabel("Degree")
plt.title("Bias-Variance Tradeoff for training data")
plt.legend()
plt.show()

plt.plot(degree, mean_mean_square_error_test, label="Error")
plt.plot(degree, mean_bias_test , label= "Bias")
plt.plot(degree, mean_variance_test, label="Variance")
plt.xlabel("Degree")
plt.title("Bias-Variance Tradeoff for testing data")
plt.legend()
plt.show()
    
