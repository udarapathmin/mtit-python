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

#Neural network training function
class Trainer(object):
    def __init__(self, NeuralN):

        self.N = NeuralN
        
    #trainer function
    def train(self, X, y):
        cost1 = self.N.costFunction(X,y)
        dJdW1, dJdW2, dJdW3 = self.N.costFunctionPrime(X, y)

        scalar = 3
        self.N.W1 = self.N.W1 + scalar * dJdW1
        self.N.W2 = self.N.W2 + scalar * dJdW2
        self.N.W3 = self.N.W3 + scalar * dJdW3
        cost2 = self.N.costFunction(X,y)
        print("Cost 1 and Cost 2")
        print (cost1, cost2)
        
        dJdW1,dJdW2, dJdW3 = self.N.costFunctionPrime(X,y)
        scalar = 3
        self.N.W1 = self.N.W1 - scalar * dJdW1
        self.N.W2 = self.N.W2 - scalar * dJdW2
        self.N.W3 = self.N.W3 - scalar * dJdW3
        cost3 = self.N.costFunction(X,y)
        print("Cost 2 and Cost 3")
        print (cost2, cost3)

        checkCost = False
        previousCost = 1

        x_axis =[]
        y_axis =[]
        z_axis =[]
       
        x_axis=np.append(x_axis, np.average(self.N.W1))
        y_axis=np.append(y_axis, np.average(self.N.W2))
        z_axis=np.append(z_axis, np.average(self.N.W3))

        #check if all cost3 values < all cost2 values if so decreas (scalar * weight)
        if (cost3<cost2):
            while(True):
                
                dJdW1,dJdW2, dJdW3 = self.N.costFunctionPrime(X,y)
                scalar = 3
                self.N.W1 = self.N.W1 - scalar * dJdW1
                self.N.W2 = self.N.W2 - scalar * dJdW2
                self.N.W3 = self.N.W3 - scalar * dJdW3
                cost3 = self.N.costFunction(X,y)

                x_axis=np.append(x_axis, np.average(self.N.W1))
                y_axis=np.append(y_axis, np.average(self.N.W2))
                z_axis=np.append(z_axis, np.average(self.N.W3))
                            
                if(cost3>previousCost):
                    print("Cost3 > previousCost")
                    print(cost3,previousCost)
                    
                    print("Weights of exit")
                    print("W1 average")
                    print(np.average(self.N.W1))
                    print("W2 average")
                    print(np.average(self.N.W2))
                    print("W3 average")
                    print(np.average(self.N.W3))
                    break
                
                previousCost = cost3
                                
        #Check if all Cost3 values > all Cost2 values if so increas (scalar * weight)
        elif(cost3>cost2):
            while(True):
                #print("Cost3>Cost2")
                dJdW1,dJdW2, dJdW3 = self.N.costFunctionPrime(X,y)
                scalar = 3
                self.N.W1 = self.N.W1 + scalar * dJdW1
                self.N.W2 = self.N.W2 + scalar * dJdW2
                self.N.W3 = self.N.W3 + scalar * dJdW3
                cost2 = self.N.costFunction(X,y)

                x_axis=np.append(x_axis, np.average(self.N.W1))
                y_axis=np.append(y_axis, np.average(self.N.W2))
                z_axis=np.append(z_axis, np.average(self.N.W3))
                
                if(cost2>previousCost):
                    print("Cost2 > previousCost")
                    print(cost2,previousCost)

                    print("Weights of exit")
                    print("W1 average")
                    print(np.average(self.N.W1))
                    print("W2 average")
                    print(np.average(self.N.W2))
                    print("W3 average")
                    print(np.average(self.N.W3))
                    
                    break
                
                previousCost = cost2
                
        else:
            print("Error in Cost comparison in Training function");
            
        return x_axis, y_axis, z_axis   