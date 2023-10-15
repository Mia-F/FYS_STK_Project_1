import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from part_a import Franke_function, design_matrix, MSE, R2, beta_OLS
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from matplotlib import cm
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

def creating_data(data_file, n):
    new_data = np.zeros((n,n))
    terrain1 = imread(data_file)
    for i in tqdm(range(n)):
        for j in range(n):
            new_data[i][j] = terrain1[i][j]
    return new_data

# Load the terrain

n = 500
degree = np.linspace(0,10,11, dtype=int)
data_file = '../DataFiles/SRTM_data_Norway_1.tif'

data = creating_data(data_file, n)

n,m = data.shape

x = np.sort(np.random.uniform(0, 1, n))
y = np.sort(np.random.uniform(0, 1, m))

x,y = np.meshgrid(x,y)



x_1D = x.ravel()
y_1D = y.ravel()
z = data.ravel()


#creating arrays for MSE and R2 score for bothe training and test data  
Error_train = np.zeros(len(degree))
Score_train = np.zeros(len(degree))

Error_test = np.zeros(len(degree))
Score_test = np.zeros(len(degree))

for i in tqdm(range(len(degree))):
    #Create the design matrix
    X = design_matrix(x,y,degree[i])
    X_train, X_test, y_train, y_test = train_test_split(X, z, test_size=0.2)
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_train_scaled = (y_train - np.mean(y_train))/np.std(y_train)
    y_test_scaled = (y_test - np.mean(y_train))/np.std(y_train)

 
    #Calculate the beta values
    beta = beta_OLS(X_train_scaled,y_train_scaled)

    #Crate the model 
    model_train = X_train_scaled @ beta
    model_test = X_test_scaled @ beta

    #calculating the MSE and R2 score
    Error_train[i] = MSE(y_train_scaled, model_train)
    Score_train[i] = R2(y_train_scaled, model_train)

    Error_test[i] = MSE(y_test_scaled, model_test)
    Score_test[i] = R2(y_test_scaled, model_test)

fig, ax = plt.subplots()
plot_1 = ax.plot(degree, Error_train, label="MSE Train", color="tab:orange")
plot_2 = ax.plot(degree, Error_test, label="MSE Test", color="tab:blue")
ax.set_ylabel("MSE")
ax.set_xlabel("Degrees")
 

ax2 = ax.twinx()
plot_3 = ax2.plot(degree, Score_train, label="R2 Train", color="tab:green")
plot_4 = ax2.plot(degree, Score_test, label="R2 Test", color="tab:red")
ax2.set_ylabel("R2 score")
plt.grid()
  

lns = plot_1 + plot_2 + plot_3 +plot_4
labels = [l.get_label() for l in lns]
plt.legend(lns, labels, loc="center right")
plt.show()
 
def beta_ridge(X, y, lamb):
    return (np.linalg.pinv(X.T @ X + lamb * np.identity(len(X[0]))) @ X.T @ y)

#Ridge
lambdas = np.logspace(-8, 2, 500)
#lambdas = np.array([10e-5])


Error_train_ridge = np.zeros((len(degree), len(lambdas)))
Error_test_ridge = np.zeros((len(degree), len(lambdas)))
R2_score_train_ridge = np.zeros((len(degree), len(lambdas)))
R2_score_test_ridge = np.zeros((len(degree), len(lambdas)))


for d in tqdm(range(len(degree))):
    X = design_matrix(x,y, degree[d])
    X_train, X_test, y_train, y_test = train_test_split(X, z, test_size=0.2) 

    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_train_scaled = (y_train - np.mean(y_train))/np.std(y_train)
    y_test_scaled = (y_test - np.mean(y_train))/np.std(y_train)

    for i in tqdm(range(len(lambdas))):
        beta = beta_ridge(X_train_scaled, y_train_scaled, lambdas[i])

        model_train = X_train_scaled @ beta
        model_test = X_test_scaled @ beta
       
        Error_train_ridge[d][i] = MSE(y_train_scaled, model_train)
        Error_test_ridge[d][i] = MSE(y_test_scaled, model_test)
        R2_score_train_ridge[d][i] = R2(y_train_scaled, model_train)
        R2_score_test_ridge[d][i] = R2(y_test_scaled, model_test)

plt.imshow(Error_train_ridge, aspect='auto')
plt.grid()
plt.xlabel(r"$\lambda$ values from $10^{-8}$ to $10^{2}$")
plt.ylabel("Degree")
plt.colorbar(label="MSE")
plt.show()

#plt.title(r"Heatmap of the MSE for the test data as a fucntion of $\lambda$ values and complexity")
plt.imshow(Error_test_ridge, aspect='auto')
plt.grid()
plt.xlabel(r"$\lambda$ values from $10^{-8}$ to $10^{2}$")
plt.ylabel("Degree")
plt.colorbar(label="MSE")
plt.show()


fig, ax = plt.subplots()
plot_1 = ax.plot(degree, Error_train_ridge[:,0], label="MSE Train", color="tab:orange")
plot_2 = ax.plot(degree, Error_test_ridge[:,0], label="MSE Test", color="tab:blue")
ax.set_ylabel("MSE")
ax.set_xlabel("Degrees")
 

ax2 = ax.twinx()
plot_3 = ax2.plot(degree, R2_score_train_ridge[:,0], label="R2 Train", color="tab:green")
plot_4 = ax2.plot(degree, R2_score_test_ridge[:,0], label="R2 Test", color="tab:red")
ax2.set_ylabel("R2 score")
plt.grid()
  

lns = plot_1 + plot_2 + plot_3 +plot_4
labels = [l.get_label() for l in lns]
plt.legend(lns, labels, loc="center right")
plt.show()

"""
#LASSo
lambdas = np.logspace(-8, 2, 10)


Error_train_LASSO = np.zeros((len(degree), len(lambdas)))
Error_test_LASSO = np.zeros((len(degree), len(lambdas)))
R2_score_train_LASSO = np.zeros((len(degree), len(lambdas)))
R2_score_test_LASSO = np.zeros((len(degree), len(lambdas)))


for d in tqdm(range(len(degree))):
    X = design_matrix(x,y, degree[d])
    X_train, X_test, y_train, y_test = train_test_split(X, z, test_size=0.2) 

    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_train_scaled = (y_train - np.mean(y_train))/np.std(y_train)
    y_test_scaled = (y_test - np.mean(y_train))/np.std(y_train)

    for i in tqdm(range(len(lambdas))):
        alpha = lambdas[i]
        clf = linear_model.Lasso(fit_intercept=True, max_iter=1000000, alpha=alpha)
        clf.fit(X_train_scaled, y_train_scaled)

        model_train = clf.predict(X_train_scaled)
        model_test = clf.predict(X_test_scaled)
        
        Error_train_LASSO[d][i] = MSE(y_train_scaled, model_train)
        Error_test_LASSO[d][i] = MSE(y_test_scaled, model_test)
        R2_score_train_LASSO[d][i] = R2(y_train_scaled, model_train)
        R2_score_test_LASSO[d][i] = R2(y_test_scaled, model_test)

plt.imshow(Error_train_LASSO, aspect='auto')
plt.grid()
plt.xlabel(r"$\lambda$ values from $10^{-8}$ to $10^{2}$")
plt.ylabel("Degree")
plt.colorbar(label="MSE")
plt.show()

#plt.title(r"Heatmap of the MSE for the test data as a fucntion of $\lambda$ values and complexity")
plt.imshow(Error_test_LASSO, aspect='auto')
plt.grid()
plt.xlabel(r"$\lambda$ values from $10^{-8}$ to $10^{2}$")
plt.ylabel("Degree")
plt.colorbar(label="MSE")
plt.show()

"""

"""
#plt.title("Data used in the analysis of the terrain data")
#plt.imshow(data, cmap=cm.twilight)
#plt.colorbar()
#plt.xlabel("X")
#plt.ylabel("Y")
#plt.show()

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

#fig = plt.figure()

#fig.suptitle("Models for the terrain data fitted with OLS of different complexities", fontsize = 20)

for i in range(len(degree)):
    if i == 0:
        print("noe")
       # ax = fig.add_subplot(2, len(degree)/2, i+1, projection='3d')
       # ax.plot_surface(x, y, data, cmap=cm.twilight,linewidth=0, antialiased=False)
       # ax.set_title(f"Terrain data", fontsize = 20)
    else:
        mod = model[i][:].reshape(n,n)
        print(np.shape(mod))
        #For ploting 
        #ax = fig.add_subplot(2, len(degree)/2, i+1, projection='3d')
        #ax.plot_surface(x, y, mod, cmap=cm.twilight,linewidth=0, antialiased=False)
       # ax.set_title(f"Degree = {degree[i]}", fontsize = 20)

#plt.show()


plt.suptitle("2D plot of the terrain data fitted with OLS of different complexities")
for i in range(len(degree)):
    plt.subplot(2,3,i+1)
    if i==0:
        print("noe")
        plt.title("Real data", fontsize = 20)
        plt.imshow(data, cmap=cm.twilight)
    else:
        mod = model[i][:].reshape(n,n)
        plt.title(f"Degree {degree[i]}", fontsize = 20)
        plt.imshow(mod, cmap=cm.twilight)
       
plt.colorbar()
plt.show()


lambdas = np.logspace(-10, 10, 100)
degree = 50

def beta_ridge(X, y, lamb):
    return (np.linalg.pinv(X.T @ X + lamb * np.identity(len(X[0]))) @ X.T @ y)

mse = np.zeros(len(lambdas))
X = design_matrix(x_1D,y_1D, degree)

X_mean = np.mean(X,axis=0)
z_mean = np.mean(z, axis=0)

X_scaled = X - X_mean
z_scaled = z - z_mean

for i in tqdm(range(len(lambdas))):
    beta = beta_ridge(X_scaled, z_scaled, lambdas[i])

    model= X_scaled @ beta
    mse[i] = MSE(z_scaled, model)

plt.semilogx(lambdas, mse)
plt.show()


lambdas = 10e-9

beta = beta_ridge(X_scaled, z_scaled, lambdas)

model= X_scaled @ beta
model = model.reshape(n,n)

plt.subplot(2,1,1)
plt.imshow(model)

plt.subplot(2,1,2)
plt.imshow(data)
plt.show()


plt.suptitle("2D plot of the terrain data fitted with OLS of different complexities")

for i in range(len(degree)):
    plt.subplot(2,3,i+1)
    if i==0:
        print("noe")
        plt.title("Real data", fontsize = 20)
        plt.imshow(data, cmap=cm.twilight)
    else:
        mod = model[i][:].reshape(n,n)
        plt.title(f"Degree {degree[i]}", fontsize = 20)
        plt.imshow(mod, cmap=cm.twilight)
    
plt.show()
"""

"""
# Scale data by subtracting mean value,own implementation
#For our own implementation, we will need to deal with the intercept by centering the design matrix and the target variable
X_train_mean = np.mean(X_train,axis=0)
#Center by removing mean from each feature
X_train_scaled = X_train - X_train_mean
X_test_scaled = X_test - X_train_mean
#The model intercept (called y_scaler) is given by the mean of the target variable (IF X is centered, note)
y_scaler = np.mean(y_train)
y_train_scaled = y_train - y_scaler
"""