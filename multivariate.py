import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy
from sklearn import preprocessing

def gradientDescent(X, y, theta, alpha, num_iters):
	m = len(y)
	J_history = np.zeros((num_iters, 1))

	for _ in range(num_iters):
		h = np.matmul(X,theta)
		y = np.reshape(y, h.shape)
		err = h-y
		dJ = np.matmul(X.T, err)/m

		theta = theta - alpha*dJ
	return theta

def computeCost(X, y, theta):
	m = len(y)
	J = 0
	
	# Vectorization
	h = np.matmul(X,theta)
	y = np.reshape(y, h.shape)
	err = h-y
	J = np.dot(err.T,err)/(2*m)
	return J

def normalize(X):
	X_norm = copy.deepcopy(X)
	mean = np.mean(X, axis=0)
	std = np.std(X, axis=0, ddof=1)

	for i in range(X.shape[1]):
		X_norm[:, i] = (X[:, i] - mean[i])/std[i]
	return X_norm, mean, std

def plot(X, y, theta=None):
	fig = plt.figure()
	ax = Axes3D(fig)
	
	ax.scatter(X[:,1], X[:,2], y)
	plt.show()

def loadData(filename):
	dataframe = pd.read_csv(filename, header=None) 
	X = dataframe.values[:, :2]
	y = dataframe.values[:, 2]
	return X, y

def main():
	filename = 'data2.csv'
	num_iters = 400;
	alpha = 0.01;

	X, y = loadData(filename)
	m = len(y) 

	if len(X.shape) == 1:
		X = np.reshape(X, (X.shape[0],1))

	X, mean, std = normalize(X)
	print mean, std

	X = np.c_[np.ones(m), X] # Add a column of ones to x
	theta = np.zeros((X.shape[1], 1))

	computeCost(X, y, theta)
	theta = gradientDescent(X, y, theta, alpha, num_iters)
	print theta

	plot(X, y,theta)

	element = np.array([1, (1650 - mean[0])/std[0], (3- mean[1])/std[1]])
	price = np.matmul(element, theta)
	print price
	

if __name__ == '__main__':
    main()