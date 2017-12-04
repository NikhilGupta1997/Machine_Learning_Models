'''
===============
Decision Forest
===============

'''

import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Global Variables
attribute_size = 54
PARA0 = (2, 5, 10, 15, 20, 25, 30) 	# Esitmators
PARA1 = (5, 10, 20, 30, 40, 54) 	# Features
PARA2 = (True, False)				# Bootstrap

''' Read From File '''
def read_file(filename):
	inftrain = open(filename, 'rb')
	tree_csv = csv.reader(inftrain)
	next(tree_csv)
	return list(tree_csv)

''' Get Attribute values '''
def get_X(myinput):
	return np.delete(myinput, attribute_size, axis = 1)

''' Get Class Values '''
def get_Y(myinput):
	return [line[attribute_size] for line in myinput]

''' Get accuracy between 2 sets '''
def get_accuracy(set1, set2):
	count = 0;
	for i in range(len(set1)):
		if set1[i] == set2[i]:
			count += 1;
	return (float)(count) / (float)(len(set1))

''' Find Best Training Parameters for Forest '''
def get_best(X_train, Y_train, X_valid, Y_valid):
	max_value = 0;
	parameters = (1, 1, 1)
	
	for estimators in PARA0:
		for features in PARA1:
			for bootstrap in PARA2:
				print "\nestimators = ", estimators
				print "features = ", features
				print "bootstrap = ", bootstrap
				model = RandomForestClassifier(n_estimators = estimators, max_features = features, bootstrap = bootstrap)
				model = model.fit(X_train, Y_train)
				classificaition = model.predict(X_valid)
				accuracy = get_accuracy(classificaition, Y_valid)
				print "accuracy = ", accuracy
				if (accuracy > max_value):
					parameters = (estimators, features, bootstrap)
					max_value = accuracy
	return max_value, parameters

# Get Data from Files
training_input = read_file('Data/train.dat')
test_input = read_file('Data/test.dat')
valid_input = read_file('Data/valid.dat')

X_train = get_X(training_input)
X_test = get_X(test_input)
X_valid = get_X(valid_input)

Y_train = get_Y(training_input)
Y_test = get_Y(test_input)
Y_valid = get_Y(valid_input)

# Get Best Features
max_value, parameters = get_best(X_train, Y_train, X_valid, Y_valid)

# Train Model on Best Features
model = RandomForestClassifier(n_estimators = parameters[0], max_features = parameters[1], bootstrap = parameters[2])
model = model.fit(X_train, Y_train)

# Get Accuracies
train_accuracy = get_accuracy(Y_train, model.predict(X_train))
test_accuracy = get_accuracy(Y_test, model.predict(X_test))
valid_accuracy = get_accuracy(Y_valid, model.predict(X_valid))

# Print Statistics
print '\nmax accuracy', max_value
print 'parameters', parameters
print 'Training Accuracy = ', train_accuracy
print 'Test Accuracy = ', test_accuracy
print 'Valid Accuracy = ', valid_accuracy


