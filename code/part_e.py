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
    # Ensure reproducibility
    # np.random.seed(2023)
    # n = 100
    # d = 15
    # poly_d = np.linspace(1, 15, 15, dtype=int)
    # x0 = np.linspace(0, 1, n)
    # y0 = np.linspace(0, 1, n)
    
    # Create coordinate grid of x, y, datapoints and dict for storing computet error
    # x, y = np.meshgrid(x0, y0)
    # franke_data = Franke_function(x, y, noise=True)
    
    # error = {
    #     "mse_train": np.zeros_like(poly_d),
    #     "mse_test": np.zeros_like(poly_d)
    # }
    # num_resamples = 10
    # mse_test = np.zeros((d, num_resamples))
    # mse_train = np.zeros((d, num_resamples))
    # e_train = np.zeros(d)
    # e_test = np.zeros(d)

    # for i in range(d):
    #for d in poly_d:
        # X = design_matrix(x, y, poly_d[i])
        # X_train, X_test, y_train, y_test = train_test_split(X, franke_data.flatten(), test_size=0.2)
        # print(X)
        # Resample training data
        # n_x = y_test.shape[0]
        # n_y = y_train.shape[0]
        # y_pred = np.zeros((n_x, num_resamples))
        # y_til = np.zeros((n_y, num_resamples))
        # print(X_train[3, :])
        # y_predict = np.zeros((n, num_resamples))

        # for j in range(num_resamples):
            #X_new = np.zeros_like(X_train)
            #y_new = np.zeros_like(y_train)
            # idx = np.random.randint(0, n_y, n_y)
            # X_new = X_train[idx, :]
            # y_new = y_train[idx]
            # print(f"X_train {X_train.shape}")
            # print(f"X_test {X_test.shape}")
            # print(f"y_train {y_train.shape}")
            # print(f"y_test {y_test.shape}")
            # print(f"X_new {X_new.shape}")
            # print(f"y_new {y_new.shape}")
            # Prediction 
            # beta = beta_OLS(X_new, y_new)
            # y_tilde = X_new @ beta
            # y_til[:, j] = X_new @ beta
            # mse_train[i, j] = np.mean((y_new - y_tilde)**2)
            # mse_train[i, j] = MSE(y_new, y_tilde)
            # y_predict = X_test @ beta
            # y_pred[:, j] = X_test @ beta
            # mse_test[i, j] = np.mean((y_test - y_predict)**2)
            # mse_test[i, j] = MSE(y_test, y_predict)
        # print(np.mean((y_train - y_til)**2, axis=1, keepdims=True))
        # e_train[i] = np.mean( np.mean((y_train - y_til)**2, axis=1, keepdims=True))
        # e_test[i] = np.mean( np.mean((y_test - y_pred)**2, axis=1, keepdims=True))
    # print(np.mean(mse_train, axis=1))
    # fig, ax = plt.subplots()
    # ax.plot(poly_d, np.mean(mse_train, axis=1), scaley='log')
    # ax.plot(poly_d, np.mean(mse_test, axis=1), scaley='log')
    # plt.show()
    # print(np.mean(mse_train, axis=1))

        #print(X_train.shape, y_train)

    # bootstrap(franke_data, 100)
