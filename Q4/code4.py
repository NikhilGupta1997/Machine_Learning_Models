'''
=============================
Gaussian Discriminant Analysis
=============================

'''

import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
from numpy.linalg import inv

''' To Read the X values '''
def Xread():
	return np.matrix([map(float, line.strip().split()) for line in open('q4x.dat')])

''' To Read the Y values '''
def Yread():
	return np.matrix([[(line.strip())] for line in open('q4y.dat')])

def LinBoundary(u0,u1,S,X):
	Mat1 = u0*inv(S) - u1*inv(S)
	Mat2 = inv(S)*(u0.T - u1.T)
	Mat = u1*inv(S)*u1.T - u0*inv(S)*u0.T
	val1 = Mat1.item(0) + Mat2.item(0)
	val2 = Mat1.item(1) + Mat2.item(1)
	x1,x2 = [],[]
	for xi in X:
		x1.append(xi)
		temp = (-val1*xi - Mat.item(0))/val2
		print temp
		x2.append(temp)
	return x1,x2

def QuadBoundary(u0,u1,S0,S1,X):
    Mat1 = inv(S0) - inv(S1)
    Mat2 = (-2)*(inv(S0) * u0.T - inv(S1) *u1.T)
    Mat3 = u0*inv(S0)*u0.T - u1*inv(S1)*u1.T
    Mat4 = math.log(math.sqrt(abs(np.linalg.det(inv(S0)) / abs(np.linalg.det(inv(S1))))))
    a = Mat1[0][0]
    b = Mat1[1][0] + Mat1[0][1]
    c = Mat1[1][1]
    d = Mat2[0].item(0)
    e = Mat2[1].item(0)
    f = (Mat3 + Mat4).item(0)
    x1,x2 = [],[]
    for xi in X:
        x1.append(xi)
        temp = -(e + b*xi) - math.sqrt((e + b*xi)**2 - 4*c*(a*xi**2 + d*xi + f))
        x2.append(temp/(2*c))
    return x1,x2

''' Read input values '''
X = Xread()
Y = Yread()

''' Normalize '''
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X-X_mean)/X_std

# Create two lists based on classification
X_one = []
X_zero = []
for i in range(len(Y.tolist())):
	if Y.item(i) == 'Canada':
		X_one.append([X.tolist()[i][0], X.tolist()[i][1]])
	else:
		X_zero.append([X.tolist()[i][0], X.tolist()[i][1]])

phi = float(len(X_one))/(len(X_one) + len(X_zero))
mean_0 = np.sum(np.matrix(X_zero), axis=0) / len(X_zero)
mean_1 = np.sum(np.matrix(X_one), axis=0) / len(X_one)

print phi, mean_0, mean_1

sigma = np.zeros((2, 2))
sigma_0 = np.zeros((2, 2))
sigma_1 = np.zeros((2, 2))

A = [1,1]
for i in range(len(X)):
	if Y.item(i) == 'Canada':
		sigma_1 += (X[i] - mean_1).T*(X[i] - mean_1)
	else:
		sigma_0 += (X[i] - mean_0).T*(X[i] - mean_0)
sigma = ( sigma_1 + sigma_0 ) / X.shape[0]
sigma_1 /= len(X_one)
sigma_0 /= len(X_zero)
print 'Sigma = \n', sigma
print 'Sigma0 = \n', sigma_0
print 'Sigma1 = \n', sigma_1

# Plot 
X_plot1 = [item[0] for item in X_one]
Y_plot1 = [item[1] for item in X_one]
plt.plot(X_plot1, Y_plot1, 'ro', label='Canada')

X_plot0 = [item[0] for item in X_zero]
Y_plot0 = [item[1] for item in X_zero]
plt.plot(X_plot0, Y_plot0, 'rx', label='Alaska')

# Plot Boundaries
x = np.arange(-2, 2, 0.1)
A , B = LinBoundary(mean_0, mean_1, sigma, x)
C , D = QuadBoundary(mean_0, mean_1, sigma_0, sigma_1, x)
plt.plot(A, B)
plt.plot(C, D)

plt.axis([min(min(X_plot0),min(X_plot1)), max(max(X_plot0),max(X_plot1)), 
	min(min(Y_plot0),min(Y_plot1)), max(max(Y_plot0),max(Y_plot1))])

# Label
plt.ylabel('x2')
plt.xlabel('x1')
plt.legend(loc='upper right')
plt.title('Classification')
plt.show()