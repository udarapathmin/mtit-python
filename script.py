#!/usr/bin/python
################################################################################
##
## Copyright 2016 Udara Karunarathna (IT13021030) and Supun Sudaraka (IT13019914).  All rights reserved.
##
################################################################################

import numpy as np
from scipy import optimize
from StringIO import StringIO
import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from neural_network import Neural_Network
from trainer import Trainer

NeuralNetwork = Neural_Network()

#Pass to Trainer Data:
X,Y = NeuralNetwork.readInputFile("input.txt")
Y = np.reshape(Y, (3,1))
print ("Input : ")
print (X)
print ("Results : ")
print (Y)

#normalized training data
X = NeuralNetwork.normalize(X)
Y = Y/100

print ("Input : ")
print (X)
print ("Results :")
print (Y)

TrainObject = Trainer(NeuralNetwork)
x, y, z = TrainObject.train(X,Y)

yHat = NeuralNetwork.forward(X)

print("YHat")
print(yHat)
print("Y")
print(Y)

print(x.shape, y.shape, z.shape)

#to take on 1000 rows of datav(limited no of rows)
x = np.resize(x, 1000)
y = np.resize(y, 1000)
z = np.resize(z, 1000)

#plot wireframe
f = plot.figure()
a = f.add_subplot(111, projection='3d')

a.plot_wireframe(x,y,z)

a.set_xlabel('W1 Values')
a.set_ylabel('W2 Values')
a.set_zlabel('W3 Values')

plot.show()