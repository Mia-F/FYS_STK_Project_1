import numpy as np
from part_a import Franke_function, design_matrix
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm

def beta_ridge(X, y, lamb):
    return np.linalg.pinv(X.T @ X + lamb * np.identity(len(X[0]))) @ X.T @ y.flatten()
    
n = 100   
degree = 10
lamb = 0.1
x = np.linspace(0, 1, n)
y = x

x,y = np.meshgrid(x,y)
f = Franke_function(x,y, noise=True)

X = design_matrix(x,y, degree)

beta = beta_ridge(X,f, lamb)

model = X @ beta_ridge(X,f, lamb)
model = model.reshape(n,n)

fig = plt.figure(figsize=(20,8))
ax = fig.gca(projection ='3d')
surf = ax.plot_surface(x,y,model,cmap=cm.twilight, linewidth = 0, antialiased=False)
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
