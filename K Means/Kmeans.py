'''
=======
K Means
=======

'''

from mpl_toolkits.mplot3d import axes3d
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.animation as animation
import time
from time import clock
import random
from sklearn.model_selection import cross_val_score
from sklearn import metrics

# Global Variables
classes = 6
epsilon = 1e-10

''' Read X Values from File '''
def Xread():
	return np.matrix([map(float, line.strip().split()) for line in open('Data/attr.txt')])

''' Read Y Values from File '''
def Yread():
	return [map(int, line.strip().split()) for line in open('Data/label.txt')]

''' Return Norm between 2 Vectors '''
def norm(vec1, vec2):
	return np.linalg.norm(vec1 - vec2)


def error(inputs, means, classified):
	J = 0;
	for i in range(inputs.shape[0]):
		index = classified[i] - 1
		J = J + norm(inputs[i], means[index])**2
	return J

''' Find best fitting cluster to input data point '''
def best_fit(input, means):
	c = 0
	min = float("inf")
	for i in range(means.shape[0]):
		val = norm(input, means[i])**2
		if val < min:
			min = val
			c = i
	return c + 1


def update_means(inputs, classified, means):
	count = [0 for _ in xrange(classes)]
	values = np.matrix([[0.0] * inputs.shape[1] for _ in xrange(classes)])
	for i in range(inputs.shape[0]):
		index = classified[i] - 1
		count[index] += 1
		values[index] += inputs[i]
	for i in range(classes):
		if count[i] == 0:
			means[i] = values[i] / 1
		else:
			means[i] = values[i] / count[i]
	return means


def max_class(indexes, labels):
	count = [0 for _ in xrange(classes)]
	for index in indexes:
		ind = labels[index][0]-1
		count[ind] += 1
	max_count = max(count)
	return max_count


def get_accuracy(classified, labels):
	correct = 0
	total = len(classified)
	for i in range(classes):
		indexes = []
		for j in range(len(classified)):
			if classified[j] == i+1:
				indexes.append(j)
		correct += max_class(indexes, labels)
	return float(correct) / float(total)

def randomize_means(inputs, means):
	for i in range(classes):
		j = random.randint(0, inputs.shape[0]-1)
		means[i] = inputs[j];
	return means

def history(inputs, means, labels, iterations):
	saved_J = []
	saved_acc = []
	classified = [1 for _ in xrange(inputs.shape[0])]
	accuracy = get_accuracy(classified, labels)
	J = error(inputs, means, classified)
	saved_J.append(J)
	saved_acc.append(accuracy) 
	for _ in range(iterations):
		for i in range(inputs.shape[0]):
			classified[i] = best_fit(inputs[i], means)
		means = update_means(inputs, classified, means)
		J_new = error(inputs, means, classified)
		print 'J_new = ', J_new
		J = J_new
		accuracy = get_accuracy(classified, labels)
		saved_J.append(J)
		saved_acc.append(accuracy) 
	return saved_J, saved_acc

''' Run KMeans Algorithm '''
def kmeans(inputs, labels, iterations):
	saved_acc = 0
	for x in range(iterations):
		print "Iteration ", x
		means = np.random.rand(classes,inputs.shape[1])
		means = randomize_means(inputs, means)
		initial_mean = means.copy()
		classified = [1 for _ in xrange(inputs.shape[0])]
		J = error(inputs, means, classified)
		accuracy = get_accuracy(classified, labels)
		count = 0
		while True:
			count += 1
			for i in range(inputs.shape[0]):
				classified[i] = best_fit(inputs[i], means)
			means = update_means(inputs, classified, means)
			J_new = error(inputs, means, classified)
			print 'J_new = ', J_new
			if J - J_new < epsilon:
				print "final J = ", J_new
				accuracy = get_accuracy(classified, labels)
				print "accuracy = ", accuracy
				print "iterations = ", count
				break
			J = J_new
		if accuracy > saved_acc:
			saved_initial_mean = initial_mean.copy()
			saved_acc = accuracy
			saved_J = J_new
	return saved_initial_mean

def convert_to_list(inputs, labels):
	train_data = []
	train_labels = []
	for i in xrange(len(labels)):
		train_labels.append(int(labels[i][0]))
	for i in xrange(inputs.shape[0]):
		param = []
		for j in xrange(inputs.shape[1]):
			param.append(inputs.item(i,j))
		train_data.append(param)
	return train_data, train_labels

def SVM(train_data, train_labels):
	print "Setting up SVM"
	clf = svm.SVC()
	print "Training SVM"
	clf.fit(train_data, train_labels) 
	print "predicting SVM"
	scores = cross_val_score(clf, train_data, train_labels, cv=10)
	print "scores", scores
	return clf.predict(train_data)

def accuracy(prediction, labels):
	count = 0
	for i in range(len(labels)):
		if prediction[i] == labels[i]:
			count += 1
	return float(count) / float(len(labels))

inputs = Xread()
labels = Yread()

# Run KMeans Till Convergence
print "PART A"
start = clock()
kmeans(inputs, labels, 1)
end = clock()
print 'in {} secs'.format(end - start)

# Part b and c
print "\nPART B"
initial_mean = kmeans(inputs, labels, 10)
J_values, accuracies = history(inputs, initial_mean, labels, 60)
print "J = ", J_values
print "Accuracy", accuracies

# Compare accuracies with SVM
print "\nPART C"
train_data, train_labels = convert_to_list(inputs, labels)
prediction = SVM(train_data, train_labels)
print "Accuracy = ", accuracy(prediction.tolist(), train_labels)

