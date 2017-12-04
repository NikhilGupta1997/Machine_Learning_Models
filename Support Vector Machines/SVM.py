'''
=======================
Support Vector Machines
=======================

'''

import numpy as np
import cvxpy as cvx
import math
import svmutil as svm

''' Read data from a file '''
def readData(filename):
	return np.matrix([map(float, line.strip().split(',')) for line in open(filename)])

''' Map the classes to either 1 or -1 '''
def conv2pn(X):
	for i in xrange(X.shape[0]):
		if X.item(i) == 1:
			X[i] = -1
		else:
			X[i] = 1
	return X

''' Return Norm Distance '''
def norm(x, y):
	return np.linalg.norm(x - y)

''' Gaussian Function '''
def gaussian(x, y):
	return math.exp(-2.5 * np.square(norm(x, y)).item())
	
''' Apply Linear Kernel '''
def kernel_lin(X, Y):
	A = np.multiply(X, Y)
	return A*A.T

''' Apply Gaussian Kernel '''
def kernel_gaussian(X, Y):
	m = Y.shape[0]
	M = [map(float, [0]) * m for _ in xrange(m)]
	for i in xrange(m):
		for j in xrange(m):
			M[i][j] = Y.item(i)*Y.item(j)*gaussian(X[i], X[j])
	return M

''' Train SVM Model '''
def SVMtrain(X, Y, str):
	C = 500 # Noise parameter
	alpha = cvx.Variable(Y.shape[0], 1); # Variable for optimization

	# Obtain kernel
	if str == "linear":
		Q = kernel_lin(X, Y)
	else:
		Q = kernel_gaussian(X, Y)

	# Define objective funtion
	obj = cvx.Maximize(cvx.sum_entries(alpha) - 0.5*(cvx.quad_form(alpha, Q))) 

	# Define constraints
	constraints = [alpha >= 0, alpha <= C, (alpha.T*Y) == 0] 
	
	# Form and solve problem.
	prob = cvx.Problem(obj, constraints)
	prob.solve()  # Returns the optimal value
	return alpha

''' Calculate the weight vector '''
def weight_vector(X, Y, a):
	mat = np.multiply(Y, a)
	return np.sum(np.multiply(X, mat), axis = 0)

''' Calculate intercept for linear model '''
def intercept(X, Y, w):
	mat = w*X.T
	maxmat = np.matrix(mat[Y.T > 0])
	minmat = np.matrix(mat[Y.T < 0])
	max = np.max(minmat)
	min = np.min(maxmat)
	return -0.5*(max + min)

''' Calculate intercept for gaussian model '''
def intercept_gaus(X, Y, a):
	m = Y.shape[0]
	mat = np.matrix([map(float, [0]) for _ in xrange(m)])
	for i in xrange(m):
		if(a[i].value > 499.9 or a[i].value < 0.1):
			continue
		temp = 0
		for j in xrange(m):
			temp += a[j].value*Y.item(j)*gaussian(X[i], X[j])
		mat[i] = temp
	maxmat = np.matrix(mat[Y > 0])
	minmat = np.matrix(mat[Y < 0])
	maxm = np.matrix(maxmat[maxmat != 0])
	minm = np.matrix(minmat[minmat != 0])
	max = np.max(minm)
	min = np.min(maxm)
	return -0.5*(max + min)

# Obtain training/testing data and labels
X = readData("svm_data/traindata.txt")
Y = readData("svm_data/trainlabels.txt")
Y = conv2pn(Y)
X1 = readData("svm_data/testdata.txt")
Y1 = readData("svm_data/testlabels.txt")

### LINEAR SVM MODEL ###

# Obtain alpha using linear kernel
alpha = SVMtrain(X, Y, "linear")
index = np.zeros((alpha.size[0], 1)) # indentify support vectors
for i in xrange(alpha.size[0]):
	index[i,0] = alpha[i].value
	if alpha[i].value > 0.1 and alpha[i].value < 499.9:
		print i

# Calculate weight vector
w = weight_vector(X, Y, index)

# Calculate intercept b
b = intercept(X, Y, w)

# Test on test data
correct = 0
count = 0
for i in xrange(Y1.shape[0]):
	val = float(w*X1[i].T) + b
	if val >= 0:
		clsfy = 2
	else:
		clsfy = 1
	if clsfy == Y1.item(i):
		correct += 1
	count += 1

print "accuracy (Linear Kernel) = ", float(correct) / float(count)


### GAUSSIAN SVM MODEL ###

# Obtain alpha using a gaussian kernel
alpha = SVMtrain(X, Y, "Gaussian")
# print alpha.value
index = np.zeros((alpha.size[0], 1)) # indentify support vectors
for i in xrange(alpha.size[0]):
	index[i,0] = alpha[i].value
	if alpha[i].value > 0.1 and alpha[i].value < 499.9:
		print i

# Calculate the intercept b
b1 = intercept_gaus(X, Y, alpha)

# Test on test data
correct = 0
count = 0
for i in xrange(Y1.shape[0]):
	temp = 0
	for j in xrange(Y.shape[0]):
		temp += alpha[j].value*Y.item(j)*gaussian(X1[i], X[j])
	val = temp + b1
	if val >= 0:
		clsfy = 2
	else:
		clsfy = 1
	if clsfy == Y1.item(i):
		correct += 1
	count += 1

print "accuracy (Gaussian Kernel) = ", float(correct) / float(count)


### LIBSVM model ###

Y2 = conv2pn(Y1)
train_data = []
train_labels = []
for i in xrange(Y.shape[0]):
	train_labels.append(int(Y.item(i)))
for i in xrange(X.shape[0]):
	param = []
	for j in xrange(X.shape[1]):
		param.append(X.item(i,j))
	train_data.append(param)
test_data = []
test_labels = []
for i in xrange(Y2.shape[0]):
	test_labels.append(int(Y2.item(i)))
for i in xrange(X1.shape[0]):
	param = []
	for j in xrange(X1.shape[1]):
		param.append(X1.item(i,j))
	test_data.append(param)

# Linear SVM Model
model = svm.svm_train(train_labels, train_data,'-t 0 -c 500')
svm.svm_predict(test_labels, test_data, model)

# Gaussian SVM Model
model = svm.svm_train(train_labels, train_data,'-g 2.5 -c 500')
svm.svm_predict(test_labels, test_data, model)

# Cross variance Accuracy
C = [1, 10, 100, 1000, 10000, 100000, 1000000]
cv_accuracies = np.zeros(7)
test_accuracies = np.zeros(7)
for i in range(7):
	test_model = svm.svm_train(train_labels, train_data,'-g 2.5 -c ' + str(C[i]))
	[predicted_label, accuracy, decision_values] = svm.svm_predict(test_labels, test_data, test_model)
	test_accuracies[i] = accuracy[1]
	cv_accuracies[i] = svm.svm_train(train_labels, train_data,'-g 2.5 -v 10 -c ' + str(C[i]))

print cv_accuracies
print test_accuracies

