import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy

def gradientDescent(X, y, theta, alpha, num_iters):
	m = len(y)
	J_history = []

	for iter_ in range(num_iters):
		#if iter_ % 100 == 0:
		#	plotData(X, y, theta=theta, num=iter_)

		sum0, sum1 = 0, 0
		for i in range(m):
			sum0 += h(theta, X[i]) - y[i]
			sum1 += (h(theta, X[i]) - y[i])*X[i]

		temp0 = theta[0] - (alpha/m)*sum0
		temp1 = theta[1] - (alpha/m)*sum1
		theta[0] = temp0
		theta[1] = temp1

		J = computeCost(X, y, theta)
		print "theta0: ", theta[0], " theta1: ", theta[1], " error: ", J
		J_history.append(J)
	return theta

def h(theta, x):
	return theta[0] + theta[1]*x

def computeCost(X, y, theta):
	m = len(y); 
	J = 0
	for i in range(m):
		J += (h(theta, X[i]) - y[i])**2
	J = J/(2*m)

	return J

def plotData(X, y, theta=None, num=0):
	plt.scatter(X, y)
	plt.ylabel('Profit in $10,000s');
	plt.xlabel('Population of City in 10,000s');

	if theta != None:
		m = len(y) 
		h_ = []
		for i in range(len(y)):
			h_.append(h(theta,X[i]))
		plt.plot(X, h_, color = 'red')
	plt.show()

def loadData(filename):
	dataframe = pd.read_csv(filename, header=None) 
	X = dataframe.values[:, 0]
	y = dataframe.values[:, 1]
	return X, y

def main():
	filename = 'data.csv'
	num_iters = 1500;
	alpha = 0.01;

	X, y = loadData(filename)
	plotData(X, y)

	m = len(y) 
	theta = [0,0]

	#print(computeCost(X, y, theta))
	theta = gradientDescent(X, y, theta, alpha, num_iters)
	print(theta)
	plotData(X, y, theta=theta, num=num_iters)
	

if __name__ == '__main__':
    main()