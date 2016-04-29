# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 19:01:44 2016

@author: gbayomi
"""

from NeuralNetwork import * 
from Loader import * 
import matplotlib.pyplot as plt

test_images, test_labels = getTestingSample(10000)
test_images = test_images/255.0

testX = test_images
testY = test_labels
    
W1 = np.load('files/fileW1.npy')
W2 = np.load('files/fileW2.npy')
J = np.load('files/fileJ.npy')

NN = NeuralNetwork(784, 50, 10, "sigmoid")
NN.loadWeights(W1, W2)
label = NN.getAccuracy(testX, testY)

plt.plot(J)
plt.xlabel(label)
plt.show()