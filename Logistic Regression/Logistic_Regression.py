'''
===================
Logistic Regression
===================

'''

import numpy as np
import matplotlib.pyplot as plt
import math
import itertools

# Global variables
iteration = 0
Epsilon = 1e-10

''' To Read the X values '''
def Xread():
	return np.matrix([map(float, line.strip().split()) for line in open('x.dat')])

''' To Read the Y values '''
def Yread():
	return np.matrix([[float(line.strip())] for line in open('y.dat')])
 
''' To create a theta vector based on dimension of input '''
def initialize_theta(x):
	return np.matrix([[float(0)]] * (x))

''' Obtains the Signum function of our hypothesis function '''
def signum(i, theta, x):
	return 1 / (1 + math.exp((-x[i] * theta).item()))

''' Generate the Hessian '''
def Hessian(Theta, X, Y):
	D = np.diag([signum(i, Theta, X) * (1 - signum(i, Theta, X)) for i in range(X.shape[0])])
	return -(X.T * D * X) / X.shape[0]

''' Generate the Gradient of Log Likelihood '''
def grad_LL(Theta, X, Y):
	Hypo = np.matrix([[signum(i, Theta, X)] for i in range(X.shape[0])])
	return X.T * (Y - Hypo) / X.shape[0]

''' Calculates the Norm difference between two matrices '''
def norm(newTheta, Theta):
	return np.linalg.norm(newTheta - Theta)

''' Defines the boundary of the Hypothesis Function '''
def boundary(x, Theta):
	return (-Theta.item(1)*x-Theta.item(0))/Theta.item(2)

# Read input values
X = Xread()
Y = Yread()

# Normalize
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X-X_mean)/X_std
X = np.c_[np.ones((X.shape[0], 1)), X]

# Calculate Boundary using Newton's Method
Theta = initialize_theta(X.shape[1])
while(True):
	iteration += 1
	newTheta = Theta - np.linalg.inv(Hessian(Theta, X, Y)) * grad_LL(Theta, X, Y)
	if norm(newTheta, Theta) < Epsilon:
		break
	Theta = newTheta
print 'Theta \n', Theta
print 'Iterations = ', iteration

### Create 2D Plot of points and classification boundary ###
# Create two lists based on classification
X_one = []
X_zero = []
for i in range(len(Y.tolist())):
	if Y.item(i) > 0.5:
		X_one.append([X.tolist()[i][1], X.tolist()[i][2]])
	else:
		X_zero.append([X.tolist()[i][1], X.tolist()[i][2]])

# Plot 
X_plot1 = [item[0] for item in X_one]
Y_plot1 = [item[1] for item in X_one]
plt.plot(X_plot1, Y_plot1, 'ro', label='Class 1')

X_plot0 = [item[0] for item in X_zero]
Y_plot0 = [item[1] for item in X_zero]
plt.plot(X_plot0, Y_plot0, 'rx', label='Class 2')

plt.axis([min(min(X_plot0),min(X_plot1))-0.2, max(max(X_plot0),max(X_plot1))+0.2, 
	min(min(Y_plot0),min(Y_plot1))-0.2, max(max(Y_plot0),max(Y_plot1))+0.2])

# Boundary
x = np.arange(-2, 2.2, 0.1)
plt.plot(x, boundary(x, Theta))

# Label
plt.ylabel('x2')
plt.xlabel('x1')
plt.legend(loc='upper left')
plt.title('Classification')
plt.show()
