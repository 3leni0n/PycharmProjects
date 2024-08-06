from scipy.io import loadmat
import numpy as np
from matplotlib import pyplot as plt

# (1) load data ‘romo_allpsth.mat’
data = loadmat(r'C:\Users\alexi\PycharmProjects\CAJAL Comp Neuro 2024\Day 5 - Christian Machens\romo_allpsth.mat',
               squeeze_me=True)  # loadmat is a function in scipy.io

# (2) Reshape data array
X = data['X']
X = X.reshape(370, 12*7501)

# (3) compute and plot data matrix X
plt.figure()
plt.imshow(X, aspect='auto')

# (4) center the data
X_mean = np.mean(X, axis=1)

# Substract the mean (X_mean) from each row of X_centered
X_centered = []
for i in range(370):
    X_centered.append(X[:, i] - X_mean)


# Plot example pair of neurons
timepoints = 1000
plt.figure()
plt.plot(X[0, 0:timepoints], X[1, 0:timepoints], 'o')

# (5) compute and plot covariance matrix
C = np.cov(X_centered)
plt.figure()
plt.imshow(C, aspect='auto')

# (6) determine eigenvalues and eigenvectors of this matrix
eigenvalues, eigenvectors = np.linalg.eig(C)

# (7) plot eigenvalues
plt.figure()
plt.plot(eigenvalues, 'o')

# (8) compute and plot the first principal components
PC1 = np.dot(eigenvectors[0, :].T, X_centered[:, 0:7501])
plt.plot(PC1)