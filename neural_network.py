#!/usr/bin/python
################################################################################
##
## Copyright 2016 Udara Karunarathna (IT13021030) and Supun Sudaraka (IT13019914).  All rights reserved.
##
################################################################################

import numpy as np
from scipy import optimize
from StringIO import StringIO
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

class Neural_Network(object):
    def __init__(self):        
        #Define Hyperparameters
        self.inputLayerSize = 3
        self.outputLayerSize = 1
        self.hiddenLayer_1Size = 3
        self.hiddenLayer_2Size = 3
    
        #Randomize Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayer_1Size)
        
        self.W2 = np.random.randn(self.hiddenLayer_1Size,self.hiddenLayer_2Size)
        
        self.W3 = np.random.randn(self.hiddenLayer_2Size, self.outputLayerSize)
        
    def forward(self, X):
        #Propogate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        
        self.z3 = np.dot(self.a2, self.W2)
        self.a3 = self.sigmoid(self.z3)
        
        self.z4 = np.dot(self.a3, self.W3)
        yHat = self.sigmoid(self.z4)
        
        return yHat
        
    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector or matrix
        return 1/(1+np.exp(-z))
    
    def sigmoidPrime(self,z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)
        return J
        
    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat =self.forward(X)
        delta4 = np.multiply(-(y-self.yHat),self.sigmoidPrime(self.z4))
        dJdW3 = np.dot(self.a3.T,delta4)

        delta3 = np.dot(delta4,self.W3.T) * self.sigmoidPrime(self.z3)
        dJdW2 = np.dot(self.a2.T,delta3)
        
        delta2 = np.dot(delta3,self.W2.T) * self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T,delta2)
        
        return dJdW1, dJdW2, dJdW3
    
    #Helper Functions for interacting with other classes:
    def getParams(self):
        #Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel(), self.W3.ravel()))
        return params
    
    def setParams(self, params):
        #Set W1 and W2 and W3 using single paramater vector.
        W1_start = 0
        W1_end = self.hiddenLayerSize1 * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize1))
        
        W2_end = W1_end + self.hiddenLayerSize1*self.hiddenLayerSize2
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize1, self.hiddenLayerSize2))
        
        W3_end = W2_end + self.hiddenLayerSize2*self.outputLayerSize
        self.W3 = np.reshape(params[W2_end:W3_end], (self.hiddenLayerSize2, self.outputLayerSize))
        
    def computeGradients(self, X, y):
        dJdW1, dJdW2, dJdW3 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel(), dJdW3.ravel()))
    
    def readInputFile(self, filePathName):
        input_data = np.genfromtxt(filePathName, skip_header=1, usecols = (0, 1, 2))
        output_data = np.genfromtxt(filePathName, skip_header=1, usecols = (3))
        return input_data,output_data

    def normalize(self , A):
        A = A/np.amax(A, axis=0)
        return A;