"""
Implementing Lasso regression
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from part_a import Franke_function, design_matrix, MSE, R2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from tqdm import tqdm

sns.set_theme()
params = {
    "font.family": "Serif",
    "font.serif": "Roman", 
    "text.usetex": True,
    "axes.titlesize": "large",
    "axes.labelsize": "large",
    "xtick.labelsize": "large",
    "ytick.labelsize": "large",
    "legend.fontsize": "large"}
plt.rcParams.update(params)


np.random.seed(2023)
n = 50
degree = np.linspace(0,5,6, dtype=int)
#alphas  = np.logspace(-8, 2, 1000)
alphas = [0.0001, 0.001, 0.01, 0.1, 1.0]


x = np.sort(np.random.uniform(0, 1, n))
y = np.sort(np.random.uniform(0, 1, n))

x,y = np.meshgrid(x,y)

x = np.ravel(x)
y = np.ravel(y)

f = Franke_function(x, y)#, noise = True)

mse_error_Lasso_train = np.zeros((len(degree),len(alphas)))
r2_score_Lasso_train = np.zeros((len(degree),len(alphas)))
mse_error_Lasso_test = np.zeros((len(degree),len(alphas)))
r2_score_Lasso_test = np.zeros((len(degree),len(alphas)))

# Scaling the data
# f.reshape(-1, 1)
# scaler = StandardScaler()
# f_scaled = scaler.fit_transform(f)

for d in range(len(degree)):
    X = design_matrix(x, y, d)
    #X_train, X_test, y_train, y_test = train_test_split(X, f.flatten(), test_size=0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, f, test_size=0.2)
  
    for i, alpha in tqdm(enumerate(alphas)):
        # Create and fit the Lasso regression model
        clf = linear_model.Lasso(fit_intercept=True, alpha=alpha)
        clf.fit(X_train, y_train)

        model_train = clf.predict(X_train)
        model_test = clf.predict(X_test)

        mse_error_Lasso_train[d][i] = MSE(y_train, model_train)
        #print(mse_error_Lasso_train[i])
        r2_score_Lasso_train[d][i] = R2(y_train, model_train)
        mse_error_Lasso_test[d][i] = MSE(y_test, model_test)
        r2_score_Lasso_test[d][i] = R2(y_test, model_test)

#  plt.title(r"Heatmap of the MSE for the training data as a fucntion of $\lambda$ values and complexity")
# plt.imshow(mse_error_Lasso_train, aspect='auto')
# plt.grid()
# plt.xlabel(r"$\lambda$ values from $10^{-8}$ to $10^{2}$")
# plt.ylabel("Degree")
# plt.colorbar(label="MSE")
# plt.show()

# plt.title(r"Heatmap of the MSE for the test data as a fucntion of $\lambda$ values and complexity")
# plt.imshow(mse_error_Lasso_test, aspect='auto')
# plt.grid()
# plt.xlabel(r"$\lambda$ values from $10^{-8}$ to $10^{2}$")
# plt.ylabel("Degree")
# plt.colorbar(label="MSE")
# plt.show()
#best_alpha = alphas[np.argmin(mse_error_Lasso_test)]
#print("Best alpha:", best_alpha)
colors = sns.color_palette("twilight", n_colors=len(alphas))
# Plot the results
fig, ax = plt.subplots()
for i, alpha in enumerate(alphas):
    plt.plot(degree, mse_error_Lasso_train[:, i], linestyle='--', color=colors[i])
    plt.plot(degree, mse_error_Lasso_test[:, i], label=rf'$\lambda = {alpha:G}$', color=colors[i])
# ax.semilogx(alphas, mse_error_Lasso_train[0][:], label="MSE Lasso (Train)", color="tab:blue")
# ax.semilogx(alphas, r2_score_Lasso_train[0][:], label="R2 Lasso (Train)", color="tab:orange")
# ax.semilogx(alphas, mse_error_Lasso_test[0][:], label="MSE Lasso (Test)", color="tab:green")
# ax.semilogx(alphas, r2_score_Lasso_test[0][:], label="R2 Lasso (Test)", color="tab:red")
# ax.set_title("MSE and R2 scores for Lasso with different polynomial degrees")
# ax.set_xlabel("Alphas")
# ax.set_ylabel("MSE and R2 values")
ax.legend()
# plt.show()
# fig, ax = plt.subplots()
# ax.semilogx(alphas, mse_error_Lasso_train[1][:], label="MSE Lasso (Train)", color="tab:blue")
# ax.semilogx(alphas, r2_score_Lasso_train[1][:], label="R2 Lasso (Train)", color="tab:orange")
# ax.semilogx(alphas, mse_error_Lasso_test[1][:], label="MSE Lasso (Test)", color="tab:green")
# ax.semilogx(alphas, r2_score_Lasso_test[1][:], label="R2 Lasso (Test)", color="tab:red")
# ax.set_title("MSE and R2 scores for Lasso with different polynomial degrees")
# ax.set_xlabel("Alphas")
# ax.set_ylabel("MSE and R2 values")
# ax.legend()
# plt.show()
# fig, ax = plt.subplots()
# ax.semilogx(alphas, mse_error_Lasso_train[2][:], label="MSE Lasso (Train)", color="tab:blue")
# ax.semilogx(alphas, r2_score_Lasso_train[2][:], label="R2 Lasso (Train)", color="tab:orange")
# ax.semilogx(alphas, mse_error_Lasso_test[2][:], label="MSE Lasso (Test)", color="tab:green")
# ax.semilogx(alphas, r2_score_Lasso_test[2][:], label="R2 Lasso (Test)", color="tab:red")
# ax.set_title("MSE and R2 scores for Lasso with different polynomial degrees")
# ax.set_xlabel("Alphas")
# ax.set_ylabel("MSE and R2 values")
# ax.legend()
# plt.show()
# fig, ax = plt.subplots()
# ax.semilogx(alphas, mse_error_Lasso_train[3][:], label="MSE Lasso (Train)", color="tab:blue")
# ax.semilogx(alphas, r2_score_Lasso_train[3][:], label="R2 Lasso (Train)", color="tab:orange")
# ax.semilogx(alphas, mse_error_Lasso_test[3][:], label="MSE Lasso (Test)", color="tab:green")
# ax.semilogx(alphas, r2_score_Lasso_test[3][:], label="R2 Lasso (Test)", color="tab:red")
# ax.set_title("MSE and R2 scores for Lasso with different polynomial degrees")
# ax.set_xlabel("Alphas")
# ax.set_ylabel("MSE and R2 values")
# ax.legend()
# plt.show()
# fig, ax = plt.subplots()
# ax.semilogx(alphas, mse_error_Lasso_train[4][:], label="MSE Lasso (Train)", color="tab:blue")
# ax.semilogx(alphas, r2_score_Lasso_train[4][:], label="R2 Lasso (Train)", color="tab:orange")
# ax.semilogx(alphas, mse_error_Lasso_test[4][:], label="MSE Lasso (Test)", color="tab:green")
# ax.semilogx(alphas, r2_score_Lasso_test[4][:], label="R2 Lasso (Test)", color="tab:red")
# ax.set_title("MSE and R2 scores for Lasso with different polynomial degrees")
# ax.set_xlabel("Alphas")
# ax.set_ylabel("MSE and R2 values")
# ax.legend()
# plt.show()
# fig, ax = plt.subplots()
# ax.semilogx(alphas, mse_error_Lasso_train[5][:], label="MSE Lasso (Train)", color="tab:blue")
# ax.semilogx(alphas, r2_score_Lasso_train[5][:], label="R2 Lasso (Train)", color="tab:orange")
# ax.semilogx(alphas, mse_error_Lasso_test[5][:], label="MSE Lasso (Test)", color="tab:green")
# ax.semilogx(alphas, r2_score_Lasso_test[5][:], label="R2 Lasso (Test)", color="tab:red")
# ax.set_title("MSE and R2 scores for Lasso with different polynomial degrees")
# ax.set_xlabel("Alphas")
# ax.set_ylabel("MSE and R2 values")
# ax.legend()
# plt.show()
# fig, ax = plt.subplots()
# ax.semilogx(alphas, mse_error_Lasso_train[6][:], label="MSE Lasso (Train)", color="tab:blue")
# ax.semilogx(alphas, r2_score_Lasso_train[6][:], label="R2 Lasso (Train)", color="tab:orange")
# ax.semilogx(alphas, mse_error_Lasso_test[6][:], label="MSE Lasso (Test)", color="tab:green")
# ax.semilogx(alphas, r2_score_Lasso_test[6][:], label="R2 Lasso (Test)", color="tab:red")
# ax.set_title("MSE and R2 scores for Lasso with different polynomial degrees")
# ax.set_xlabel("Alphas")
# ax.set_ylabel("MSE and R2 values")
# ax.legend()
# plt.show()
# fig, ax = plt.subplots()
# ax.semilogx(alphas, mse_error_Lasso_train[7][:], label="MSE Lasso (Train)", color="tab:blue")
# ax.semilogx(alphas, r2_score_Lasso_train[7][:], label="R2 Lasso (Train)", color="tab:orange")
# ax.semilogx(alphas, mse_error_Lasso_test[7][:], label="MSE Lasso (Test)", color="tab:green")
# ax.semilogx(alphas, r2_score_Lasso_test[7][:], label="R2 Lasso (Test)", color="tab:red")
# ax.set_title("MSE and R2 scores for Lasso with different polynomial degrees")
# ax.set_xlabel("Alphas")
# ax.set_ylabel("MSE and R2 values")
# ax.legend()
# plt.show()
# fig, ax = plt.subplots()
# ax.semilogx(alphas, mse_error_Lasso_train[8][:], label="MSE Lasso (Train)", color="tab:blue")
# ax.semilogx(alphas, r2_score_Lasso_train[8][:], label="R2 Lasso (Train)", color="tab:orange")
# ax.semilogx(alphas, mse_error_Lasso_test[8][:], label="MSE Lasso (Test)", color="tab:green")
# ax.semilogx(alphas, r2_score_Lasso_test[8][:], label="R2 Lasso (Test)", color="tab:red")
# ax.set_title("MSE and R2 scores for Lasso with different polynomial degrees")
# ax.set_xlabel("Alphas")
# ax.set_ylabel("MSE and R2 values")
# ax.legend()
# plt.show()
# fig, ax = plt.subplots()
# ax.semilogx(alphas, mse_error_Lasso_train[9][:], label="MSE Lasso (Train)", color="tab:blue")
# ax.semilogx(alphas, r2_score_Lasso_train[9][:], label="R2 Lasso (Train)", color="tab:orange")
# ax.semilogx(alphas, mse_error_Lasso_test[9][:], label="MSE Lasso (Test)", color="tab:green")
# ax.semilogx(alphas, r2_score_Lasso_test[9][:], label="R2 Lasso (Test)", color="tab:red")
# ax.set_title("MSE and R2 scores for Lasso with different polynomial degrees")
# ax.set_xlabel("Alphas")
# ax.set_ylabel("MSE and R2 values")
# ax.legend()
# plt.show()

# plt.plot(np.log10(alphas), mse_error_Lasso_train, label="MSE train")
# plt.plot(np.log10(alphas), mse_error_Lasso_test, label="MSE test")
plt.show()

# Plotting MSE vs Degrees for different alpha values.
# plt.figure(figsize=(15, 8))
# for i, alpha in enumerate(alphas):
#     plt.plot(degree, mse_error_Lasso_train[:, i], label=rf'$\lambda$ = $10^{{{int(np.log10(alpha))}}}$', linestyle=':')
#     plt.plot(degree, mse_error_Lasso_test[:, i], label=rf'$\lambda$ = $10^{{{int(np.log10(alpha))}}}$', linestyle='-' )

# plt.xlabel('Degree')
# plt.ylabel('Mean Squared Error (MSE)')
# plt.title('MSE vs. Degrees for different Alpha values (Lasso)')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Plotting R2 vs Degrees for different alpha values.
# plt.figure(figsize=(12, 6))
# for i, alpha in enumerate(alphas):
#     plt.plot(degree, r2_score_Lasso_train[:, i], label=rf'$\lambda$ = $10^{{{int(np.log10(alpha))}}}$', linestyle=':' )
#     plt.plot(degree, r2_score_Lasso_test[:, i], label=rf'$\lambda$ = $10^{{{int(np.log10(alpha))}}}$', linestyle='-' )

# plt.xlabel('Degree')
# plt.ylabel('R-squared (R2)')
# plt.title('R2 vs. Degrees for different Alpha values (Lasso)')
# plt.legend()
# plt.grid(True)
# plt.show()


