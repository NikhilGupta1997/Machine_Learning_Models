'''
============================
Principal Component Analysis
============================

'''

from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.animation as animation
import random
import glob
import os
import sys, getopt
from time import clock
from scipy import ndimage
from scipy import misc
from PIL import Image
from scipy.misc import toimage
from scipy.interpolate import interp1d
from sklearn.model_selection import cross_val_score

width = 0
hight = 0

def readX(file):
	global width
	global height
	inputs = []
	labels = []
	class_label = -1;
	for dirName, subdirList, fileList in os.walk(file):
		print dirName, subdirList, fileList
		class_label += 1
		for filename in fileList:
			if dirName != file and filename != ".DS_Store":
				image = Image.open(os.path.join(dirName, filename)).convert('L')
				width, height = image.size
				inputs.append(np.asarray(image).flatten().tolist())
				labels.append(class_label)
	return np.matrix(inputs), labels

def display_face(face, name):
	data = face.copy()
	data.resize(height, width)
	toimage(data).save('{}.png'.format(name))

def map_gray(eigenface):
	interpolate = interp1d([eigenface.min(), eigenface.max()],[0,255])
	return interpolate(eigenface)

def convert_to_list(inputs):
	train_data = []
	for i in xrange(inputs.shape[0]):
		param = []
		for j in xrange(inputs.shape[1]):
			param.append(inputs.item(i,j))
		train_data.append(param)
	return train_data

inputs, labels = readX(str(sys.argv[1]))
print labels

# Part A
average_face = np.mean(inputs, axis=0)
mean_faces = np.subtract(inputs, average_face)
normalized_faces = np.divide(mean_faces, np.std(mean_faces, axis=0)).T
print normalized_faces.shape
display_face(average_face, "average_face")

# Part B
U, s, V = np.linalg.svd(normalized_faces, full_matrices=False)
best_50 = np.matrix(s[:50])
with open('principal_components.txt', 'wb') as f:
	for line in best_50:
		np.savetxt(f, line, fmt='%.2f')

# Part C
for i in range(0,5):
	eigenface = U.T[i]
	display_face(map_gray(eigenface), 'eigenface{}'.format(i))

# Part D
eigenfaces = []
for i in range(0,50):
	eigenfaces.append(U.T[i][0].tolist()[0])
with open('eigenfaces.txt', 'wb') as f:
	count = 0
	for line in eigenfaces:
		count += 1
		f.write("eigenvector {}\n".format(count))
		np.savetxt(f, line, fmt='%.6f')

new_proj = np.matrix(eigenfaces) * normalized_faces
proj = new_proj / np.linalg.norm(new_proj, axis=0)
mean_proj = np.subtract(proj, np.mean(proj, axis=1))
normalized_proj = np.divide(mean_proj, np.std(mean_proj, axis=1))

reduced_images = []
for row in proj.T:
	image = map_gray(np.dot(row,eigenfaces))
	image += average_face
	image = map_gray(image)
	reduced_images.append(image.tolist()[0])
reduced_images = np.matrix(reduced_images)
with open('reduced_images.txt', 'wb') as f:
	count = 0
	for line in reduced_images:
		count += 1
		f.write("image {}\n".format(count))
		np.savetxt(f, line, fmt='%.2f')

# Part E
clf = svm.SVC(decision_function_shape='ovo')
start = clock()
scores = np.mean(cross_val_score(clf, convert_to_list(normalized_proj.T), labels, cv=10))
end = clock()
print "score1 = ", scores, 'in {} secs'.format(end - start)

clf = svm.SVC(decision_function_shape='ovo')
start = clock()
scores = np.mean(cross_val_score(clf, convert_to_list(normalized_faces.T), labels, cv=10))
end = clock()
print "score2 = ", scores, 'in {} secs'.format(end - start)

# Part F
for i in range(1,4):
	j = random.randint(1, inputs.shape[0])
	display_face(inputs[j], "input{}".format(i))
	display_face(reduced_images[j], "output{}".format(i))

