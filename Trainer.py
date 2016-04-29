# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 15:57:47 2016

@author: gbayomi
"""

from NeuralNetwork import * 
from Loader import * 

train_images, train_labels = getSample(60000)
train_images = train_images/255.0

test_images, test_labels = getTestingSample(10000)
test_images = test_images/255.0

X = train_images
Y = train_labels

testX = test_images
testY = test_labels

NN = NeuralNetwork(784, 50, 10, "sigmoid", 0.001)
NN.loadRandomWeights()
J = NN.train(X, Y, 0.0001, 1200)
print NN.getAccuracy(testX, testY)

#Binary data
np.save('files/fileW1.npy', NN.W1)
np.save('files/fileW2.npy', NN.W2)
np.save('files/fileJ.npy', J)
