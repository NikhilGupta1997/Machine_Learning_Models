'''
===============
Neural Networks
===============

'''

import math
import numpy as np
import random
import itertools

# Global Variables
LearningRate = 25
iteration = 100000
variable_rate = False
sigmoid = True
input_size = 127
output_size = 3

''' Helps read the input file into a workable foramt '''
def readFile(fileName):
	mat = np.array([line.strip().split(',') for line in open(fileName)])
	return np.array([[line[:-1]] + [line[-1]] for line in mat])

''' This helps create a single perceptron unit '''
def create_unit(size):
	return np.array([random.uniform(-0.1, 0.1) for _ in range(size)]) # Random Initialized Weights

''' Creates a whole layer of perceptrons '''
def create_layer(num, size):
	layer = np.zeros((num,size))
	for i in range(num):
		layer[i] = create_unit(size)
	return layer

''' The Sigmoid Activation function '''
def sigmoid(X):
	return np.matrix( [1 / (1 + math.exp((-X.item(i)))) for i in range(X.shape[1])] )

''' The SoftPlus Activation function '''
def softplus(X):
	return np.matrix( [math.log(1 + math.exp(X.item(i))) for i in range(X.shape[1])] )

''' Activation Function '''
def activation(X):
		if sigmoid:
			return sigmoid(X)
		else:
			return softplus(X)

''' Finds the norm of two vectors '''
def norm(vec1, vec2):
	return np.linalg.norm(vec1 - vec2)

''' Classifies the class based on the output layer values '''
def classify(output, map_output):
	saved = ''
	min = float("inf")
	for res, val in map_output.iteritems():
		if(norm(val, output) < min):
			saved = res
			min = norm(val, output)
	return saved

''' Train Model '''
def train(inputs, hidden, output, map_output):
	global iteration
	for input in inputs:
		if variable_rate: 
			iteration += 1;
			eta = float(LearningRate / float(iteration)**(1.0/2.0)) # Use for changing eta
		else: 
			eta = 0.1 # Use for constant eta

		res = map_output[input[1]]
		test_input = np.matrix(input[0].astype(np.float))
		hid_out = activation(test_input*hidden.T)
		hid_output = np.matrix(np.append([1], hid_out))
		out = activation(hid_output*output.T)
		error_out = np.multiply(np.multiply(out, (1 - out)), (res - out))
		error_hid = np.multiply(np.multiply(hid_out, (1 - hid_out)), (error_out * output)[:,1:])
		output = output + (eta * error_out.T * hid_output)
		hidden = hidden + (eta * error_hid.T * test_input)
	return hidden, output

''' Test Model '''
def test(hidden, output, tests, map_output):
	mat = np.array([['a'] + ['b'] for line in tests])
	count = -1
	for test in tests:
		count = count + 1
		test_input = np.matrix(test[0].astype(np.float))
		hid_out = activation(test_input * hidden.T)
		hid_output = np.matrix(np.append([1], hid_out))
		out = activation(hid_output * output.T)
		clsy = classify(out, map_output)
		mat[count][0] = test[1]
		mat[count][1] = clsy
	return mat

# Get input
inputs = readFile('Data/train.data')
tests = readFile('Data/test.data')

# Append 1 to input layer to give input to hidden layer
for i in range(inputs.shape[0]):
	inputs[i][0] = np.append([1], inputs[i][0])
for i in range(tests.shape[0]):
	tests[i][0] = np.append([1], tests[i][0])

# Create a mapping between different outcomes
list1, list2 = ['win', 'draw', 'loss'], [[1,0,0],[0,1,0],[0,0,1]]
map_output = dict( zip( list1, list2))

# Create train set and validation set
x = int(0.8*inputs.shape[0])
train_set = inputs[:x,:]
validation_set = inputs[x:,:]

# Define the number of hidden layer units
# C = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500] To Compare between no. of Hidden Units
C = [100]
for i in range(len(C)):
	print "no of hidden units = ", C[i]
	hidden = create_layer(C[i], input_size)
	output = create_layer(output_size, C[i] + 1)
	saved_validation_acc = [0]
	saved_test_acc = [0]
	saved_train_acc = [0]
	flag_count = 0;
	for i in itertools.count():
		# Train model for one iteration
		hidden, output = train(train_set, hidden, output, map_output)

		# Test on validation set
		result = test(hidden, output, validation_set, map_output)
		count = 0
		correct = 0
		for line in result:
			count = count + 1 
			if line[0] == line[1]:
				correct = correct + 1
		accuracy = float(correct) / float(count)
		print "accuracy = ", accuracy
		saved_validation_acc.extend([accuracy])
		print saved_validation_acc
		if accuracy <= saved_validation_acc[i]:
			flag_count += 1
		else:
			flag_count = 0
			saved_hidden = hidden
			saved_output = output
		if flag_count > 3: # If validation accuracy drops 3 consecutive times, then break
			break;

	# Classify Test
	result = test(saved_hidden, saved_output, tests, map_output)
	count = 0
	correct = 0
	for line in result:
		count = count + 1 
		if line[0] == line[1]:
			correct = correct + 1
	accuracy = float(correct) / float(count)
	print "test set accuracy = ", accuracy

	# Classify Training
	result = test(saved_hidden, saved_output, inputs, map_output)
	count = 0
	correct = 0
	for line in result:
		count = count + 1 
		if line[0] == line[1]:
			correct = correct + 1
	accuracy = float(correct) / float(count)
	print "train set accuracy = ", accuracy
