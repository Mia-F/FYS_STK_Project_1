import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Seaborn style setting
sns.set_theme()
params = {
    "font.family": "Serif",
    "font.serif": "Roman", 
    "text.usetex": True,
    "axes.titlesize": "large",
    "axes.labelsize": "large",
    "xtick.labelsize": "large",
    "ytick.labelsize": "large",
    "legend.fontsize": "large"
}
plt.rcParams.update(params)

def design_matrix(x, y, degree):
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)
    N = len(x)
    l = int((degree + 1) * (degree + 2) / 2)
    X = np.ones((N, l))
    for i in range(1, degree + 1):
        q = int((i) * (i + 1) / 2)
        for k in range(i + 1):
            X[:, q + k] = (x**(i - k)) * (y**k)
    return X

def Franke_function(x, y, noise=False):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    if noise:
        noise_val = np.random.normal(0, 0.1, len(x))
        return term1 + term2 + term3 + term4 + noise_val
    else:
        return term1 + term2 + term3 + term4
    

def mse(z_data, z_model):
    return np.sum((z_data - z_model)**2) / len(z_data)


def k_fold(data, k):
    n_samples = len(data)
    fold_size = n_samples // k
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    k_fold_indices = []
    for i in range(k):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size
        test_indices = indices[test_start:test_end]
        train_indices = np.concatenate([indices[:test_start], indices[test_end:]])
        k_fold_indices.append((train_indices, test_indices))
    return k_fold_indices

# Beta OLS function
def beta_OLS(X, y):
    return np.linalg.pinv(X.T @ X) @ X.T @ y

# Cross-validation function
def crossval_ols(x, y, z, degree, k):
    x_data = design_matrix(x, y, degree)
    z_data = z.flatten()
    scores_KFold_train = np.zeros(k)
    scores_KFold_test = np.zeros(k)
    k_fold_indices = k_fold(x_data, k)
    for j, (train_indices, test_indices) in enumerate(k_fold_indices):
        X_train, X_test = x_data[train_indices], x_data[test_indices]
        y_train, y_test = z_data[train_indices], z_data[test_indices]
        beta_ols = beta_OLS(X_train, y_train)

        y_tilde = (X_train @ beta_ols).ravel()
        y_predict = (X_test @ beta_ols).ravel()
        scores_KFold_train[j] = mse(y_train.flatten(), y_tilde)
        scores_KFold_test[j] = mse(y_test.flatten(), y_predict)
    mse_train_ols = np.mean(scores_KFold_train)
    mse_test_ols = np.mean(scores_KFold_test)

    return (mse_train_ols, mse_test_ols)
    
if __name__ == '__main__':
    np.random.seed(2023)
    nsamples = 20
    nx, ny = nsamples, nsamples  
    x_ = np.sort(np.random.uniform(0, 1, nx))
    y_ = np.sort(np.random.uniform(0, 1, ny))
    x, y = np.meshgrid(x_, y_)
    x, y = x.ravel(), y.ravel()
    z = Franke_function(x, y, noise=False)
    z_noise = Franke_function(x, y, noise=True)  
    data = np.column_stack((x, y, z_noise))

    # Lambda values and colors
    nlambdas = 6  
    lambdas = np.logspace(-8, 2, nlambdas)
    colors = sns.color_palette("husl", nlambdas)

    # Degree setting
    max_degree = 15
    degrees = np.arange(1, max_degree + 1)
    k = 5

    # mse_per_degree_ols = []
    # for degree in degrees:
    #     mse_train_ols, mse_test_ols = crossval_ols(x, y, z_noise, degree, k=5)  
    #     mse_per_degree_ols.append(mse_test_ols)
    # plt.plot(degrees, mse_per_degree_ols, label='OLS', color='black', linewidth=2.0)
    # plt.xlabel('Degree')
    # plt.ylabel('MSE (log scale)')
    # plt.yscale('log')
    # plt.legend()
    # plt.savefig('OLS_crossval_mse_deg')
    # plt.show()

    # Ridge Regression
    # plt.figure(figsize=(10, 6))
    # for i, lmb in enumerate(lambdas):
    #     mse_per_degree_ridge = []
    #     for degree in degrees:
    #         scores_KFold = np.zeros(k)
    #         k_fold_indices = k_fold(data, k)
    #         for j, (train_indices, test_indices) in enumerate(k_fold_indices):
    #             train, test = data[train_indices], data[test_indices]
    #             Xtrain = design_matrix(train[:,0], train[:,1], degree=degree)
    #             ridge = Ridge(alpha=lmb)
    #             ridge.fit(Xtrain, train[:,2])
    #             Xtest = design_matrix(test[:,0], test[:,1], degree=degree)
    #             zpred = ridge.predict(Xtest)
    #             scores_KFold[j] = np.mean((zpred.ravel() - test[:,2]) ** 2)
    #         mse_per_degree_ridge.append(np.mean(scores_KFold))
    #     plt.plot(degrees, mse_per_degree_ridge, color=colors[i], label=f'Ridge, λ={lmb:.1e}')

    # plt.xlabel('Degree')
    # plt.ylabel('MSE (log scale)')
    # plt.yscale('log')
    # plt.legend()
    # plt.savefig('Ridge_crossval_mse_deg')
    # plt.show()

    # Lasso Regression
    # plt.figure(figsize=(10, 6))
    # for i, lmb in enumerate(lambdas):
    #     mse_per_degree_lasso = []
    #     for degree in degrees:
    #         scores_KFold = np.zeros(k)
    #         k_fold_indices = k_fold(data, k)
    #         for j, (train_indices, test_indices) in enumerate(k_fold_indices):
    #             train, test = data[train_indices], data[test_indices]
    #             Xtrain = design_matrix(train[:,0], train[:,1], degree=degree)
    #             lasso = Lasso(alpha=lmb, max_iter=10000)
    #             lasso.fit(Xtrain, train[:,2])
    #             Xtest = design_matrix(test[:,0], test[:,1], degree=degree)
    #             zpred = lasso.predict(Xtest)
    #             scores_KFold[j] = np.mean((zpred.ravel() - test[:,2]) ** 2)
    #         mse_per_degree_lasso.append(np.mean(scores_KFold))
    #     plt.plot(degrees, mse_per_degree_lasso, color=colors[i], label=f'Lasso, λ={lmb:.1e}')

    # plt.xlabel('Degree')
    # plt.ylabel('MSE (log scale)')
    # plt.yscale('log')
    # plt.legend()
    # plt.savefig('Lasso_crossval_mse_deg')
    # plt.show()

    # Plotting
    fig, ax =plt.subplots()

    k_values = np.arange(5, 11)  # k from 5 to 10
    colors = sns.color_palette("tab10", len(k_values))

    for idx, k in enumerate(k_values):
        mse_test_ols_vals = []
        mse_train_ols_vals = []
        for degree in degrees:
            mse_train_ols, mse_test_ols = crossval_ols(x, y, z_noise, degree, k)
            mse_train_ols_vals.append(mse_train_ols)
            mse_test_ols_vals.append(mse_test_ols)
        
        ax.plot(degrees, mse_test_ols_vals, label=f'k={k}', color=colors[idx])

    ax.set_xlabel('Polynomial degree')
    ax.set_ylabel('MSE (log)')
    ax.set_yscale('log')
    ax.legend(title='k-Fold', loc='upper left')
    # plt.grid(True, linestyle='--', alpha=0.001)
    # plt.tight_layout()
    plt
    fig.savefig("../LaTeX/Images/cv_kfolds.png")
    plt.show()