# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 15:57:47 2016

@author: gbayomi
"""

from NeuralNetwork import * 
from Loader import * 

#Format the training images to [0, 1) size
train_images, train_labels = getSample(60000)
train_images = train_images/255.0

#Format the testing images to [0, 1) size
test_images, test_labels = getTestingSample(10000)
test_images = test_images/255.0

#Set X, Y for training and testX and testY for testing
X = train_images
Y = train_labels

testX = test_images
testY = test_labels

#Generate a Neural Network: input size -> 784; hidden layer->50; output->10; activation-> sigmoid; regularization-> 0.001
NN = NeuralNetwork(784, 50, 10, "sigmoid", 0.001)
#Load random weights from (-1, 1)
NN.loadRandomWeights()
#Repeat backprop algorithm 1200 times
J = NN.train(X, Y, 0.0001, 120)
print NN.getAccuracy(testX, testY)

#Save binary data
np.save('files/fileW1.npy', NN.W1)
np.save('files/fileW2.npy', NN.W2)
np.save('files/fileJ.npy', J)
