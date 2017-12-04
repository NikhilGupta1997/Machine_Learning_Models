'''
=================
Linear Regression
=================

'''

from mpl_toolkits.mplot3d import axes3d
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.animation as animation
import time

# Global variables
iteration = 0
LearningRate = 2.5
Epsilon = 1e-25
Saved_Theta = []

''' This provides the analytical solution '''
def analytical_solution(X, Y):
    return np.linalg.inv(X.T* X) * X.T * Y

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
def create_J(X, Y, Theta):
	return ((Y - X * Theta).T * (Y - X * Theta) / (2*X.shape[0])).item(0)

''' Returns the gradient of J(theta) '''
def create_gradJ(X, Y, Theta):
	return X.T * (X*Theta - Y) / X.shape[0]

''' The Gradient Descent Algorithm '''
def gradient_descent(X, Y):
	global iteration
	global LearningRate
	global Saved_Theta
	Theta = initialize_theta(X.shape[1])
	J = create_J(X, Y, Theta)
	Saved_Theta.append([item for sublist in Theta.tolist() for item in sublist] + [J])
	while(True):
		print LearningRate
		iteration += 1;
		gradJ = create_gradJ(X, Y, Theta)
		newTheta = Theta - LearningRate * gradJ
		newJ = create_J(X, Y, newTheta)
		if math.fabs(J - newJ) < Epsilon: 	# Value has converged
			break
		elif newJ > J:						# Overshoot condition
			if iteration > 8:
				break		
		else:								# Normal Gradient Descent
			LearningRate *= 1.0001			
		Theta = newTheta
		J = newJ
		Saved_Theta.append([item for sublist in Theta.tolist() for item in sublist] + [J])
	return Theta

''' Equation of the hypothesis function '''
def linear(Theta, x):
	return Theta.item(1)*x + Theta.item(0)

# Read input values
X = Xread()
Y = Yread()

# Normalize
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X-X_mean)/X_std
X = np.c_[np.ones((X.shape[0], 1)), X]

# Perform Gradient Descent
FinalTheta = gradient_descent(X, Y)

# Print Output
print 'Analytical Solution\n', analytical_solution(X,Y)
print 'Gradient Decent Solution\n', FinalTheta
print 'Iterations used = ', iteration

### 2D plot of the hypothesis function ###
X_plot = [item[1] for item in X.tolist()]
Y_plot = [item[0] for item in Y.tolist()]
x = np.arange(min(X_plot)-1, max(X_plot)+1, 0.1)

# Plot
plt.plot(X_plot, Y_plot, 'ro')
plt.plot(x, linear(FinalTheta, x))
plt.axis([min(X_plot)-0.2, max(X_plot)+0.2, min(Y_plot)-0.2, max(Y_plot)+0.2])

# Label
plt.ylabel('Prices')
plt.xlabel('Area')
plt.title('House Prices')
plt.show()

### 3D plot of the J(theta) function ###
# Returns the value of J(theta)
def create_J_plot(Theta_0, Theta_1):
	Theta = np.matrix([[Theta_0],[Theta_1]])
	return ((Y - X * Theta).T * (Y - X * Theta) / (2*X.shape[0])).item(0)

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_zlim(-100, 200)

# Plot the 3D curve
A = []; B = []; C = []
theta_0_plot = np.arange(-15, 20, 0.5)
theta_1_plot = np.arange(-15, 20, 0.5)
theta_0_plot, theta_1_plot = np.meshgrid(theta_0_plot, theta_1_plot)
Z = np.vectorize(create_J_plot)(theta_0_plot, theta_1_plot)
ax.plot_surface(theta_0_plot, theta_1_plot, Z, rstride=1, cstride=1, alpha=0.3, linewidth=0.1, cmap=cm.coolwarm)

# Animation
for line in Saved_Theta:
	# Plot the new wireframe and pause briefly before continuing
	A.append(line[0]); B.append(line[1]); C.append(line[2])
	wframe = ax.plot_wireframe(A, B, C, rstride=1, cstride=1)
	point = ax.plot([line[0]],[line[1]],[line[2]], 'ro')
	plt.pause(.02)

# Draw Contours
A = []; B = []; C = []
cont = plt.figure()
CS = plt.contour(theta_0_plot, theta_1_plot, Z)
plt.title('Contour Plot Showing Gradient Descent')

# Animation
for line in Saved_Theta:
	# Plot the new wireframe and pause briefly before continuing
	A.append(line[0]); B.append(line[1]); C.append(line[2])
	point = plt.plot([line[0]],[line[1]], 'ro')
	point = plt.plot(A,B)
	plt.pause(.02)
plt.show()