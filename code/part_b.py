import numpy as np
from part_a import Franke_function, design_matrix, MSE, R2
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from sklearn.model_selection import train_test_split

np.random.seed(2023)

def beta_ridge(X, y, lamb):
    return np.linalg.pinv(X.T @ X + lamb * np.identity(len(X[0]))) @ X.T @ y.flatten()
    
n = 100   
degree = 20
lamb = np.linspace(1e-10, 1e-5, 100)
print(np.log10(lamb))

x = np.linspace(0, 1, n)
y = x

x,y = np.meshgrid(x,y)
f = Franke_function(x,y)#, noise=True)

mse_error_Ridge_train = np.zeros(len(lamb))
r2_score_train = np.zeros(len(lamb))
mse_error_Ridge_test = np.zeros(len(lamb))
r2_score_test = np.zeros(len(lamb))

index = 0
mse_error = 1000
R2_score = 0
fig, ax = plt.subplots()

for l in lamb:
    #print(l) 
    X = design_matrix(x,y, degree)
    X_train, X_test, y_train, y_test = train_test_split(X, f.flatten(), test_size=0.2) 
    
    beta = beta_ridge(X_train,y_train, l)

    model_train = X_train @ beta
    model_test = X_test @ beta

        #calculating the MSE and R2 score
    mse_error_Ridge_train[index] = MSE(y_train, model_train)
        #r2_score_train[index] = R2(y_train, model_train)

        #mse_error_Ridge_test[index] = MSE(y_test, model_test)
        #r2_score_test[index] = R2(y_test, model_test)
    index += 1
   
    
 

ax.plot(np.log10(lamb), mse_error_Ridge_train)
#ax.plot(lamb, mse_error_Ridge_test)
plt.legend()
plt.show()


#fig, ax = plt.subplots()
#ax.scatter(x[0], (X @ beta)[0:100])
#ax.plot(x[0], f[0])
#plt.show()

  
#fig = plt.figure(figsize=(20,8))
#ax = fig.gca(projection ='3d')
#surf = ax.plot_surface(x,y,model,cmap=cm.twilight, linewidth = 0, antialiased=False)
#ax.set_zlim(-0.10,1.40)
#ax.set_xlabel('X-axis', fontsize=30)
#ax.set_ylabel('Y-axis', fontsize=30)
#ax.set_zlabel('Z-axis', fontsize=30)
#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#ax.zaxis.labelpad = 12
#ax.xaxis.labelpad = 25
#ax.yaxis.labelpad = 20
#ax.tick_params(axis='x', labelsize=20)
#ax.tick_params(axis='y', labelsize=20)
#ax.tick_params(axis='z', labelsize=20)
#plt.show()
