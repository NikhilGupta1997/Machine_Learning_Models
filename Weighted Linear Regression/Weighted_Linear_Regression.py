'''
==================================
Locally Weighted Linear Regression
==================================

'''

import numpy as np
import matplotlib.pyplot as plt
import math
import itertools

# Global variables
INTERVAL_SIZE = 50
Tau = 0.8

''' This provides the analytical solution '''
def analytical_solution_W(W, X, Y):
    return np.linalg.inv(X.T * W * X) * X.T * W * Y

''' This provides the analytical solution '''
def analytical_solution(X, Y):
    return np.linalg.inv(X.T * X) * X.T * Y

''' To Read the X values '''
def Xread():
	return np.matrix([map(float, line.strip().split()) for line in open('x.dat')])

''' To Read the Y values '''
def Yread():
	return np.matrix([[float(line.strip())] for line in open('y.dat')])
 
''' To create a theta vector based on dimension of input '''
def initialize_theta(x):
	return np.matrix([[float(0)]] * (x))

''' Returns the value of J(theta) '''
def create_J(W, X, Y, Theta):
	return ((Y - X * Theta).T * W * (Y - X * Theta) / (2*X.shape[0])).item(0)

''' Returns the gradient of J(theta) '''
def create_gradJ(W, X, Y, Theta):
	return X.T * W * (X*Theta - Y) / X.shape[0]

''' Returns the Weighted Matrix for each x-value '''
def create_W(curr_x, X, Tau):
    m = X.shape[0]
    W = [[0] * m for _ in xrange(m)] # Creates an m x m matrix
    for i in range(m):
        xi = X[i].tolist()[0][1]
        W[i][i] = math.exp(-((curr_x[0] - xi) ** 2) / (2 * Tau * Tau))
    return np.matrix(W)

''' Equation of the hypothesis function '''
def linear(x, theta_0, theta_1):
	return theta_1*x + theta_0

''' The Gradient Descent Algorithm '''
def normal_solution_W(curr_x, X, Y):
	Theta = initialize_theta(X.shape[1])
	W = create_W(curr_x, X, Tau)
	return analytical_solution_W(W, X, Y)

''' The Gradient Descent Algorithm '''
def normal_solution(curr_x, X, Y):
	Theta = initialize_theta(X.shape[1])
	return analytical_solution(X, Y)

# Read input values
X = Xread()
Y = Yread()

# Normalize
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X-X_mean)/X_std
X = np.c_[np.ones((X.shape[0], 1)), X]

# Get min and max values
min_X = X.min(axis=0).tolist()[0]
max_X = X.max(axis=0).tolist()[0]

# Create a list of lists to hold x values over a range
Ranges = []
for i in xrange(1, X.shape[1]):
	temp = np.linspace(min_X[i], max_X[i], INTERVAL_SIZE).tolist()
	Ranges.append(temp)

# Calculate Theta for each value of x
Saved_Theta = []
for x in itertools.product(*Ranges):
	FinalTheta = normal_solution_W(x, X, Y)
	print 'Current X = ', x[0]
	print 'Theta obtained\n', FinalTheta
	Saved_Theta.append([x[0], FinalTheta.item(0), FinalTheta.item(1)])

### 2D plot of the Hypothesis Function ###
X_plot = [item[1] for item in X.tolist()]
Y_plot = [item[0] for item in Y.tolist()]

# Plot
plt.plot(X_plot, Y_plot, 'ro')
plt.axis([min(X_plot), max(X_plot), min(Y_plot), max(Y_plot)])
Points_X = [Saved_Theta[0][0]]; Points_Y = [linear(Saved_Theta[0][0], Saved_Theta[0][1], Saved_Theta[0][2])]
for i in range(len(Saved_Theta) - 1):
	Points_X.append(Saved_Theta[i+1][0])
	Points_Y.append(linear(Saved_Theta[i+1][0], Saved_Theta[i][1], Saved_Theta[i][2]))
plt.plot(Points_X, Points_Y)

# Label
plt.ylabel('Prices')
plt.xlabel('Area')
plt.title('House Prices')
plt.show()


