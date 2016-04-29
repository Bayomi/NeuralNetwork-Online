# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 19:01:44 2016

@author: gbayomi
"""

from NeuralNetwork import * 
from Loader import * 
import matplotlib.pyplot as plt

#Format the training images to [0, 1) size
test_images, test_labels = getTestingSample(10000)
test_images = test_images/255.0

#Format the testing images to [0, 1) size
testX = test_images
testY = test_labels
    
#Load the last saved results    
W1 = np.load('files/9271W1.npy')
W2 = np.load('files/9271W2.npy')
J = np.load('files/9271J.npy')

#Run a Neural Network: input size -> 784; hidden layer->50; output->10; activation-> sigmoid
NN = NeuralNetwork(784, 50, 10, "sigmoid")
NN.loadWeights(W1, W2)
label = NN.getAccuracy(testX, testY)

plt.plot(J)
plt.xlabel(label)
plt.show()