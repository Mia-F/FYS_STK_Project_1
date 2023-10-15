"""
Implementing Lasso regression
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from part_a import Franke_function, design_matrix, MSE, R2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from tqdm import tqdm

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


np.random.seed(2023)
n = 20
degree = np.array([0, 1, 2, 3, 4, 5])
#alphas = np.logspace(-8,2,1000)
alphas = np.logspace(-5, -3, 3)

x = np.sort(np.random.uniform(0, 1, n))
y = np.sort(np.random.uniform(0, 1, n))

x,y = np.meshgrid(x,y)

x = np.ravel(x)
y = np.ravel(y)

f = Franke_function(x, y, noise = True)


mse_error_Lasso_train = np.zeros((len(degree),len(alphas)))
r2_score_Lasso_train = np.zeros((len(degree),len(alphas)))
mse_error_Lasso_test = np.zeros((len(degree),len(alphas)))
r2_score_Lasso_test = np.zeros((len(degree),len(alphas)))

# Scaling the data
#scaler = StandardScaler()
#f_scaled = scaler.fit_transform(f)

for d in range(len(degree)):
    X = design_matrix(x, y, d)
    X_train, X_test, y_train, y_test = train_test_split(X, f, test_size=0.2)
  
    for i, alpha in tqdm(enumerate(alphas)):
        # Create and fit the Lasso regression model
        #clf = linear_model.Lasso(fit_intercept=True, max_iter=1000000, alpha=alpha)
        clf = linear_model.Lasso(fit_intercept=True, max_iter=10000, alpha=alpha)
        clf.fit(X_train, y_train)

        model_train = clf.predict(X_train)
        model_test = clf.predict(X_test)

        mse_error_Lasso_train[d][i] = MSE(y_train, model_train)
        r2_score_Lasso_train[d][i] = R2(y_train, model_train)
        mse_error_Lasso_test[d][i] = MSE(y_test, model_test)
        r2_score_Lasso_test[d][i] = R2(y_test, model_test)

"""
#plt.title(r"Heatmap of the MSE for the training data as a fucntion of $\lambda$ values and complexity")
plt.imshow(mse_error_Lasso_train, aspect='auto')
plt.grid()
plt.xlabel(r"$\lambda$ values from $10^{-8}$ to $10^{2}$")
plt.ylabel("Degree")
plt.colorbar(label="MSE")
plt.show()

#plt.title(r"Heatmap of the MSE for the test data as a fucntion of $\lambda$ values and complexity")
plt.imshow(mse_error_Lasso_test, aspect='auto')
plt.grid()
plt.xlabel(r"$\lambda$ values from $10^{-8}$ to $10^{2}$")
plt.ylabel("Degree")
plt.colorbar(label="MSE")
plt.show()
"""

#best_alpha = alphas[np.argmin(mse_error_Lasso_test)]
#print("Best alpha:", best_alpha)

#Plotting MSE and R2 score
fig, ax = plt.subplots()
plot_1 = ax.plot(degree, mse_error_Lasso_train[:,0], label="MSE Train", color="tab:orange")
plot_2 = ax.plot(degree, mse_error_Lasso_test[:,0], label="MSE Test", color="tab:blue")
ax.set_ylabel("MSE")
ax.set_xlabel("Degrees")
 
ax2 = ax.twinx()
plot_3 = ax2.plot(degree, r2_score_Lasso_train[:,0], label="R2 Train", color="tab:green")
plot_4 = ax2.plot(degree, r2_score_Lasso_test[:,0], label="R2 Test", color="tab:red")
ax2.set_ylabel("R2 score")
plt.grid()
  
lns = plot_1 + plot_2 + plot_3 +plot_4
labels = [l.get_label() for l in lns]
plt.legend(lns, labels, loc="center right")
plt.show()
