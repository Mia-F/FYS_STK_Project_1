import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from part_a import Franke_function, design_matrix, MSE, R2, beta_OLS
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.model_selection import train_test_split
from matplotlib import cm
from tqdm import tqdm

sns.set_theme()
font = {'weight' : 'bold',
        'size'   : 32}
plt.rc('font', **font)
sns.set(font_scale=2)


def creating_data(data_file, n):
    new_data = np.zeros((n,n))
    terrain1 = imread(data_file)
    for i in tqdm(range(n)):
        for j in range(n):
            new_data[i][j] = terrain1[i][j]
    return new_data

# Load the terrain

n = 500
degree = np.array([0, 10, 20, 30, 40, 50])
data_file = 'SRTM_data_Norway_1.tif'

data = creating_data(data_file, n)

#creating x and y
n,m = data.shape

x = np.sort(np.random.uniform(0, 1, n))
y = np.sort(np.random.uniform(0, 1, m))

x,y = np.meshgrid(x,y)


x_1D = x.ravel()
y_1D = y.ravel()
z = data.ravel()

#scaler = StandardScaler(with_std=False)
#scaler.fit(z)
#zscaled = scaler.transform(z)
model = np.zeros((len(degree), n*n))
for d in tqdm(range(len(degree))):
    X = design_matrix(x_1D,y_1D, degree[d])

#X_train, X_test, y_train, y_test = train_test_split(X, z, test_size=0.2) 

    beta = beta_OLS(X,z)

#Crate the model 
    model[d][:] = X @ beta

fig = plt.figure()

fig.suptitle("Models for the terrain data fitted with OLS of different complexities", fontsize = 20)

for i in range(len(degree)):
    if i == 0:
        ax = fig.add_subplot(2, len(degree)/2, i+1, projection='3d')
        ax.plot_surface(x, y, data, cmap=cm.twilight,linewidth=0, antialiased=False)
        ax.set_title(f"Terrain data", fontsize = 20)
    else:
        mod = model[i][:].reshape(n,n)
        print(np.shape(mod))
        #For ploting 
        ax = fig.add_subplot(2, len(degree)/2, i+1, projection='3d')
        ax.plot_surface(x, y, mod, cmap=cm.twilight,linewidth=0, antialiased=False)
        ax.set_title(f"Degree = {degree[i]}", fontsize = 20)

plt.show()


plt.suptitle("2D plot of the terrain data fitted with OLS of different complexities")
for i in range(len(degree)):
    plt.subplot(2,3,i+1)
    if i==0:
        plt.title("Real data", fontsize = 20)
        plt.imshow(data, cmap=cm.twilight)
    else:
        mod = model[i][:].reshape(n,n)
        plt.title(f"Degree {degree[i]}", fontsize = 20)
        plt.imshow(mod, cmap=cm.twilight)
    
plt.show()