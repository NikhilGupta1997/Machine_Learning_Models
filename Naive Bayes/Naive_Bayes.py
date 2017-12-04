'''
===========
Naive Bayes
===========

'''

import math
import numpy as np

''' Read from Input File '''
def readFile(fileName):
	data = []
	old = np.array(data)
	for line in open(fileName):
		A = line.strip().split()
		B = A[:1]
		C = A[1:]
		new = np.array([B] + [C])
		old = np.concatenate((old, new))
	size = np.shape(old)[0]
	old = np.reshape(old, (size/2,2))
	return old

''' Remove Duplicate Values '''
def remove_duplicates(values):
	output = []
	temp = []
	seen = set()
	for value in values:
		temp.extend(value)
	for value in temp:
		if value not in seen:
			output.append(value)
			seen.add(value)
	return output

''' Get index of category '''
def map_category(mapping, str):
	return mapping[str]

''' Get index of word '''
def map_word(dictionary, str):
	try:
		A = dictionary[str]
		return A
	except Exception as e:
		return -1

''' Train Model '''
def train(categories, dictionary, inputs):
	cats = len(categories)
	dicts = len(dictionary)
	values = np.empty((cats, dicts, 2))
	for i in range(len(categories)):
		for j in range(len(dictionary)):
			values[i][j][0] = 1
			values[i][j][1] = 0

	word_counts = np.array([[dicts, 0]]*cats)

	for i in range(len(inputs)):
		print i
		curr = inputs[i]
		category = curr[0][0]
		words = curr[1]
		cat_index = map_category(categories, category)
		for word in words:
			word_index = map_word(dictionary, word)
			values[cat_index][word_index][0] += 1
		count = len(words)
		word_counts[cat_index][0] = int(word_counts[cat_index][0]) + count
		word_counts[cat_index][1] = int(word_counts[cat_index][1]) + 1

	for i in range(len(word_counts)):
		for line in values[i]:
			line[1] = float(line[0]) / float(word_counts[i][0])
	return values, word_counts

''' Test Model '''
def classify(trained, categories, dictionary, classifiers, tests):
	output = np.array([['a'] + ['b'] for line in tests])
	probs = np.array([[1.0] for line in categories])
	counter = -1
	for test in tests:
		counter = counter + 1
		print counter
		cat = test[0][0]
		cat_index = map_category(categories, cat)
		words = test[1]
		for i in range(len(classifiers)):
			for word in words:
				word_index = map_word(dictionary, word)
				if word_index == -1:
					continue
				probs[i][0] = float(probs[i][0]) + math.log(float(trained[i][word_index][1]))
			probs[i][0] = float(probs[i][0]) + math.log(float(classifiers[i][1]))
		index = probs.argmax()
		output[counter][0] = cat_index
		output[counter][1] = index
		for i in range(len(probs)):
			probs[i][0] = 1.0
	return output

# Read Inputs
print "Reading Training Data"
inputs = readFile('nb_data/r8-train-all-terms-new.txt')
print "Reading Test Data"
tests = readFile('nb_data/r8-test-all-terms-new.txt')

# Remove duplicates
categories = remove_duplicates(inputs[:,0])
dictionary = remove_duplicates(inputs[:,1])

# Create hash maps
mycats = dict(zip(categories, np.arange(0, len(categories), 1)))
mydict = dict(zip(dictionary, np.arange(0, len(dictionary), 1)))

# Train the data
print "Training Data"
trained_values, classifiers = train(mycats, mydict, inputs)

# Test the data
print "Testing Data"
category = classify(trained_values, mycats, mydict, classifiers, tests)

# Calculate the accuracy of the test input
for line in category:
	print line
count = 0
correct = 0
for line in category:
	count = count + 1 
	if line[0] == line[1]:
		correct = correct + 1
print "accuracy = ", float(correct) / float(count)

# Create the confusion matrix
m = len(categories)
M = [map(float, [0]) * m for _ in xrange(m)]
for line in category:
	i = int(line[0])
	j = int(line[1])
	M[i][j] += 1
N = np.matrix(M)
S = np.sum(N, axis=1)

# Create accuracy confusion matrix
for i in xrange(m):
	for j in xrange(m):
		N[i,j] = float(N.item((i, j))) / float(S.item(i))



