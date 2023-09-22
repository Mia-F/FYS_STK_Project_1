import numpy as np
from part_a import Franke_function, design_matrix, beta_OLS, MSE, R2
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample

np.random.seed(2023)

def bootstrap(original_data: np.ndarray, sample_size:int) -> np.ndarray:
    """Resampling data using bootstrap algorithm.
    
    Args:
        original_data (np.ndarray): Datapoints to resample from
        sample_size (int): Number of samples to generate
        
    Returns:
        np.ndarray: Resampled datapoints
    """
    resample_data = np.zeros(sample_size)
    n = original_data.size

    for i in range(sample_size):
        # draw random sample
        draw = original_data.flatten()
        resample_data[i] = np.mean(draw[np.random.randint(0, n, n)])
    
    print("Original data")
    print(f"Mean = {np.mean(original_data)}, Sigma = {np.std(original_data)}")

    print("Resampled data")
    print(f"Mean = {np.mean(resample_data)}, Sigma = {np.std(resample_data)}")

def problem_e():
    np.random.seed(12)

    n = 100
    n_boostraps = 100
    maxdegree = 16


    # Make data set.
    x_vec = np.linspace(0, 1, n)
    y_vec = np.linspace(0, 1, n)
    # y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2)+ np.random.normal(0, 0.1, x.shape)
    x, y = np.meshgrid(x_vec, y_vec)
    # franke_noise = Franke_function(x, y, noise=True)
    franke_smooth = Franke_function(x, y, noise=False)

    # X = np.column_stack((x.reshape(-1), y.reshape(-1)))
    F = franke_smooth.reshape(-1, 1)

    mse_train = np.zeros(maxdegree)
    mse_test = np.zeros(maxdegree)
    bias = np.zeros(maxdegree)
    variance = np.zeros(maxdegree)
    polydegree = np.zeros(maxdegree)

    for degree in range(maxdegree):
        X = design_matrix(x, y, degree)
        X_train, X_test, y_train, y_test = train_test_split(X, franke_smooth.flatten(), test_size=0.2)

        y_tild = np.empty((y_train.shape[0], n_boostraps))
        y_pred = np.empty((y_test.shape[0], n_boostraps))
        
        for i in range(n_boostraps):
            X_, y_ = resample(X_train, y_train)
            beta = beta_OLS(X_,y_)
            y_tilde = X_ @ beta
            y_predict = X_test @ beta
            y_tild[:, i] = y_tilde
            y_pred[:, i] = y_predict
            
            mse_train[degree] += mean_squared_error(y_, y_tilde)
            mse_test[degree] += mean_squared_error(y_test, y_predict)
        bias[degree] = np.mean((y_test - np.mean(y_pred, axis=1, keepdims=True))**2)
        variance[degree] = np.mean(np.var(y_pred, axis=1, keepdims=True))

        polydegree[degree] = degree
        # mse_train[degree] /= n_boostraps
        # mse_test[degree] /= n_boostraps
        # bias[degree] /= n_boostraps 
        # variance[degree] /= n_boostraps

    fig, ax = plt.subplots()
    ax.plot(polydegree, mse_train, label='Train')
    ax.plot(polydegree, mse_test, label='Test')
    plt.plot(polydegree, bias, label='bias')
    plt.plot(polydegree, variance, label='Variance')
    ax.set_xlabel('Model Complexity')
    ax.set_ylabel('Prediction Error')
    # ax.plot(polydegree, bias, label='Bias')
    # ax.plot(polydegree, variance, label='Variance')
    ax.set_xlim((0, 15))
    # ax.set_ylim(())
    ax.legend()




if __name__ == '__main__':
    problem_e()
    plt.show()
    