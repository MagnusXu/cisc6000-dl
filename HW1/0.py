#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 11:02:24 2019

@author: lordxuzhiyu
"""

import numpy as np
import pandas as pd
import math

train_x = np.load('/Users/lordxuzhiyu/Desktop/mnist_data/mnist.train.npy')
train_y = np.load('/Users/lordxuzhiyu/Desktop/mnist_data/mnist.trainlabel.npy')
test_x = np.load('/Users/lordxuzhiyu/Desktop/mnist_data/mnist.test.npy')
test_y = pd.read_csv('/Users/lordxuzhiyu/Desktop/mnist_data/sample_submission.csv', usecols = ['class'])

train_y = train_y.reshape(-1, 1)
test_y = test_y.to_numpy()
test_y = test_y.reshape(-1, 1)

train_x = train_x / 255
test_x = test_x / 255

pixels = 784
train_x = train_x.reshape(train_x.shape[0], pixels)
test_x = test_x.reshape(test_x.shape[0], pixels)

class NeuralNetwork:
    def __init__(self, hidden1, hidden2, output_unit, learning_rate, n):
        self.hidden1 = self.hidden1
        self.hidden2 = self.hidden2
        self.output_unit = self.output_unit
        
        # Initialize the weights use 1/n. 
        # n is the total number of the instances from training set 
        self.weight = 1 / n
        self.weight1 = np.random.normal(self.weight, pow(self.hidden1, -0.5), (self.hidden2, self.hidden1))
        self.weight2 = np.random.normal(self.weight, pow(self.hidden2, -0.5), (self.output_unit, self.hidden2))
        
        self.learning_rate = self.learning_rate
        pass
    
    def tanh(self, x):
        s = np.tanh(x)
        return s
    
    def tanh_derivatives(self, x):
        deriv = 1.0 - np.tanh(x)**2
        return deriv
    
    def cross_entropy_loss(self, y_hat, y):
        if y == 1:
            loss = -np.log(y_hat)
        else:
            loss = -np.log(1 - y_hat)
        return loss
    
    def softmax(self, x):
        # Avoid the overflow
        exps = np.exp(x - np.max(x))
        result = exps / np.sum(exps)
        return result
    
    def cross_entropy(self, X,y):
        """
        X is the output from fully connected layer (num_examples x num_classes)
        y is labels (num_examples x 1)
        	Note that y is not one-hot encoded vector. 
        	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
        """
        m = y.shape[0]
        p = self.softmax(X)
        log_likelihood = -np.log(p[range(m),y])
        loss = np.sum(log_likelihood) / m
        return loss
    
    


        
        
        
        
        
