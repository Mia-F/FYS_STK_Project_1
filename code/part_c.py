"""
Implementing Lasso regression
"""

import numpy as np
import matplotlib.pyplot as plt
from part_a import Franke_function, design_matrix, MSE, R2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

np.random.seed(2023)
n = 100  
degree = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.linspace(0, 1, n)
y = x
x, y = np.meshgrid(x, y)

f = Franke_function(x, y, noise = False)

# Define the values of lambda (alpha) for Lasso regression
alphas = np.logspace(-4, 4, 100)

mse_error_Lasso_train = np.zeros(len(alphas))
r2_score_Lasso_train = np.zeros(len(alphas))
mse_error_Lasso_test = np.zeros(len(alphas))
r2_score_Lasso_test = np.zeros(len(alphas))

# Scaling the data
scaler = StandardScaler()
f_scaled = scaler.fit_transform(f)

for i, alpha in enumerate(alphas):
    for d in degree:
        X = design_matrix(x, y, d)
        X_train, X_test, y_train, y_test = train_test_split(X, f.flatten(), test_size=0.2)

        # Create and fit the Lasso regression model
        lasso = Lasso(alpha=alpha)
        lasso.fit(X_train, y_train)

        model_train = lasso.predict(X_train)
        model_test = lasso.predict(X_test)

        mse_error_Lasso_train[i] = MSE(y_train, model_train)
        r2_score_Lasso_train[i] = R2(y_train, model_train)
        mse_error_Lasso_test[i] = MSE(y_test, model_test)
        r2_score_Lasso_test[i] = R2(y_test, model_test)

#best_alpha = alphas[np.argmin(mse_error_Lasso_test)]
#print("Best alpha:", best_alpha)

# Plot the results
fig, ax = plt.subplots()
ax.semilogx(alphas, mse_error_Lasso_train, label="MSE Lasso (Train)", color="tab:blue")
ax.semilogx(alphas, r2_score_Lasso_train, label="R2 Lasso (Train)", color="tab:orange")
ax.semilogx(alphas, mse_error_Lasso_test, label="MSE Lasso (Test)", color="tab:green")
ax.semilogx(alphas, r2_score_Lasso_test, label="R2 Lasso (Test)", color="tab:red")
ax.set_title("MSE and R2 scores for Lasso with different polynomial degrees")
ax.set_xlabel("Alphas")
ax.set_ylabel("MSE and R2 values")
ax.legend()
plt.show()


