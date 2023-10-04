"""
Implementing Lasso regression
"""

import numpy as np
import matplotlib.pyplot as plt
from part_a import Franke_function, design_matrix, MSE, R2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from tqdm import tqdm

np.random.seed(2023)
n = 100
degree = np.array([1,2,3,4,5])
#degree = np.array([9])

x = np.sort(np.random.uniform(0, 1, n))
y = np.sort(np.random.uniform(0, 1, n))

x,y = np.meshgrid(x,y)

x = np.ravel(x)
y = np.ravel(y)

f = Franke_function(x, y)#, noise = True)

alphas = np.logspace(-8,2,1000)

mse_error_Lasso_train = np.zeros((len(degree),len(alphas)))
r2_score_Lasso_train = np.zeros((len(degree),len(alphas)))
mse_error_Lasso_test = np.zeros((len(degree),len(alphas)))
r2_score_Lasso_test = np.zeros((len(degree),len(alphas)))

# Scaling the data
#scaler = StandardScaler()
#f_scaled = scaler.fit_transform(f)

for d in range(len(degree)):
    X = design_matrix(x, y, d)
    X_train, X_test, y_train, y_test = train_test_split(X, f.flatten(), test_size=0.2)
  
    for i, alpha in tqdm(enumerate(alphas)):
        # Create and fit the Lasso regression model
        clf = linear_model.Lasso(fit_intercept=True, max_iter=1000000, alpha=alpha)
        clf.fit(X_train, y_train)

        model_train = clf.predict(X_train)
        model_test = clf.predict(X_test)

        mse_error_Lasso_train[d][i] = MSE(y_train, model_train)
        #print(mse_error_Lasso_train[i])
        r2_score_Lasso_train[d][i] = R2(y_train, model_train)
        mse_error_Lasso_test[d][i] = MSE(y_test, model_test)
        r2_score_Lasso_test[d][i] = R2(y_test, model_test)

plt.title(r"Heatmap of the MSE for the training data as a fucntion of $\lambda$ values and complexity")
plt.imshow(mse_error_Lasso_train, aspect='auto')
plt.grid()
plt.xlabel(r"$\lambda$ values from $10^{-8}$ to $10^{2}$")
plt.ylabel("Degree")
plt.colorbar(label="MSE")
plt.show()

plt.title(r"Heatmap of the MSE for the test data as a fucntion of $\lambda$ values and complexity")
plt.imshow(mse_error_Lasso_test, aspect='auto')
plt.grid()
plt.xlabel(r"$\lambda$ values from $10^{-8}$ to $10^{2}$")
plt.ylabel("Degree")
plt.colorbar(label="MSE")
plt.show()
#best_alpha = alphas[np.argmin(mse_error_Lasso_test)]
#print("Best alpha:", best_alpha)

# Plot the results
fig, ax = plt.subplots()
ax.semilogx(alphas, mse_error_Lasso_train[0][:], label="MSE Lasso (Train)", color="tab:blue")
ax.semilogx(alphas, r2_score_Lasso_train[0][:], label="R2 Lasso (Train)", color="tab:orange")
ax.semilogx(alphas, mse_error_Lasso_test[0][:], label="MSE Lasso (Test)", color="tab:green")
ax.semilogx(alphas, r2_score_Lasso_test[0][:], label="R2 Lasso (Test)", color="tab:red")
ax.set_title("MSE and R2 scores for Lasso with different polynomial degrees")
ax.set_xlabel("Alphas")
ax.set_ylabel("MSE and R2 values")
ax.legend()
plt.show()
fig, ax = plt.subplots()
ax.semilogx(alphas, mse_error_Lasso_train[1][:], label="MSE Lasso (Train)", color="tab:blue")
ax.semilogx(alphas, r2_score_Lasso_train[1][:], label="R2 Lasso (Train)", color="tab:orange")
ax.semilogx(alphas, mse_error_Lasso_test[1][:], label="MSE Lasso (Test)", color="tab:green")
ax.semilogx(alphas, r2_score_Lasso_test[1][:], label="R2 Lasso (Test)", color="tab:red")
ax.set_title("MSE and R2 scores for Lasso with different polynomial degrees")
ax.set_xlabel("Alphas")
ax.set_ylabel("MSE and R2 values")
ax.legend()
plt.show()
fig, ax = plt.subplots()
ax.semilogx(alphas, mse_error_Lasso_train[2][:], label="MSE Lasso (Train)", color="tab:blue")
ax.semilogx(alphas, r2_score_Lasso_train[2][:], label="R2 Lasso (Train)", color="tab:orange")
ax.semilogx(alphas, mse_error_Lasso_test[2][:], label="MSE Lasso (Test)", color="tab:green")
ax.semilogx(alphas, r2_score_Lasso_test[2][:], label="R2 Lasso (Test)", color="tab:red")
ax.set_title("MSE and R2 scores for Lasso with different polynomial degrees")
ax.set_xlabel("Alphas")
ax.set_ylabel("MSE and R2 values")
ax.legend()
plt.show()
fig, ax = plt.subplots()
ax.semilogx(alphas, mse_error_Lasso_train[3][:], label="MSE Lasso (Train)", color="tab:blue")
ax.semilogx(alphas, r2_score_Lasso_train[3][:], label="R2 Lasso (Train)", color="tab:orange")
ax.semilogx(alphas, mse_error_Lasso_test[3][:], label="MSE Lasso (Test)", color="tab:green")
ax.semilogx(alphas, r2_score_Lasso_test[3][:], label="R2 Lasso (Test)", color="tab:red")
ax.set_title("MSE and R2 scores for Lasso with different polynomial degrees")
ax.set_xlabel("Alphas")
ax.set_ylabel("MSE and R2 values")
ax.legend()
plt.show()
fig, ax = plt.subplots()
ax.semilogx(alphas, mse_error_Lasso_train[4][:], label="MSE Lasso (Train)", color="tab:blue")
ax.semilogx(alphas, r2_score_Lasso_train[4][:], label="R2 Lasso (Train)", color="tab:orange")
ax.semilogx(alphas, mse_error_Lasso_test[4][:], label="MSE Lasso (Test)", color="tab:green")
ax.semilogx(alphas, r2_score_Lasso_test[4][:], label="R2 Lasso (Test)", color="tab:red")
ax.set_title("MSE and R2 scores for Lasso with different polynomial degrees")
ax.set_xlabel("Alphas")
ax.set_ylabel("MSE and R2 values")
ax.legend()
plt.show()
fig, ax = plt.subplots()
ax.semilogx(alphas, mse_error_Lasso_train[5][:], label="MSE Lasso (Train)", color="tab:blue")
ax.semilogx(alphas, r2_score_Lasso_train[5][:], label="R2 Lasso (Train)", color="tab:orange")
ax.semilogx(alphas, mse_error_Lasso_test[5][:], label="MSE Lasso (Test)", color="tab:green")
ax.semilogx(alphas, r2_score_Lasso_test[5][:], label="R2 Lasso (Test)", color="tab:red")
ax.set_title("MSE and R2 scores for Lasso with different polynomial degrees")
ax.set_xlabel("Alphas")
ax.set_ylabel("MSE and R2 values")
ax.legend()
plt.show()
fig, ax = plt.subplots()
ax.semilogx(alphas, mse_error_Lasso_train[6][:], label="MSE Lasso (Train)", color="tab:blue")
ax.semilogx(alphas, r2_score_Lasso_train[6][:], label="R2 Lasso (Train)", color="tab:orange")
ax.semilogx(alphas, mse_error_Lasso_test[6][:], label="MSE Lasso (Test)", color="tab:green")
ax.semilogx(alphas, r2_score_Lasso_test[6][:], label="R2 Lasso (Test)", color="tab:red")
ax.set_title("MSE and R2 scores for Lasso with different polynomial degrees")
ax.set_xlabel("Alphas")
ax.set_ylabel("MSE and R2 values")
ax.legend()
plt.show()
fig, ax = plt.subplots()
ax.semilogx(alphas, mse_error_Lasso_train[7][:], label="MSE Lasso (Train)", color="tab:blue")
ax.semilogx(alphas, r2_score_Lasso_train[7][:], label="R2 Lasso (Train)", color="tab:orange")
ax.semilogx(alphas, mse_error_Lasso_test[7][:], label="MSE Lasso (Test)", color="tab:green")
ax.semilogx(alphas, r2_score_Lasso_test[7][:], label="R2 Lasso (Test)", color="tab:red")
ax.set_title("MSE and R2 scores for Lasso with different polynomial degrees")
ax.set_xlabel("Alphas")
ax.set_ylabel("MSE and R2 values")
ax.legend()
plt.show()
fig, ax = plt.subplots()
ax.semilogx(alphas, mse_error_Lasso_train[8][:], label="MSE Lasso (Train)", color="tab:blue")
ax.semilogx(alphas, r2_score_Lasso_train[8][:], label="R2 Lasso (Train)", color="tab:orange")
ax.semilogx(alphas, mse_error_Lasso_test[8][:], label="MSE Lasso (Test)", color="tab:green")
ax.semilogx(alphas, r2_score_Lasso_test[8][:], label="R2 Lasso (Test)", color="tab:red")
ax.set_title("MSE and R2 scores for Lasso with different polynomial degrees")
ax.set_xlabel("Alphas")
ax.set_ylabel("MSE and R2 values")
ax.legend()
plt.show()
fig, ax = plt.subplots()
ax.semilogx(alphas, mse_error_Lasso_train[9][:], label="MSE Lasso (Train)", color="tab:blue")
ax.semilogx(alphas, r2_score_Lasso_train[9][:], label="R2 Lasso (Train)", color="tab:orange")
ax.semilogx(alphas, mse_error_Lasso_test[9][:], label="MSE Lasso (Test)", color="tab:green")
ax.semilogx(alphas, r2_score_Lasso_test[9][:], label="R2 Lasso (Test)", color="tab:red")
ax.set_title("MSE and R2 scores for Lasso with different polynomial degrees")
ax.set_xlabel("Alphas")
ax.set_ylabel("MSE and R2 values")
ax.legend()
plt.show()

plt.plot(np.log10(alphas), mse_error_Lasso_train, label="MSE train")
plt.plot(np.log10(alphas), mse_error_Lasso_test, label="MSE test")
plt.show()


