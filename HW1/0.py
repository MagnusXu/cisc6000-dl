#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 11:02:24 2019

@author: lordxuzhiyu
"""

import numpy as np
import pandas as pd

class NeuralNetwork:
    def __init__(self, hidden1, hidden2, output_unit, learning_rate, N):
        self.hidden1 = self.hidden1
        self.hidden2 = self.hidden2
        self.output_unit = self.output_unit
        
        # Initialize the weights use 1/N. 
        # N is the total number of the instances from training set 
        self.weight = 1 / N
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
    
    def mini_batch(self, matrix, batch_size):
        size = matrix.shape[0]
        mask = np.random.choice(size, batch_size)
        matrix_batch = matrix[mask]
        return matrix_batch
    
    def cross_entropy_loss(self, y, t):
        # y is the output of the neural network
        # t is the test data value
        batch_size = 32
        y = self.mini_batch(y, batch_size)
        t = self.mini_batch(t, batch_size)
        delta = 1e-7
        whole_loss = -np.sum(t * np.log(y + delta))
        result = whole_loss / batch_size 
        return result
    
    def train(self, inputs, targets):
        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.weight1, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.tanh(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = np.dot(self.weight2, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.tanh(final_inputs)
        
        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = np.dot(self.weight2.T, output_errors) 
        
        # update the weights for the links between the hidden and output layers
        self.weight2 += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        
        # update the weights for the links between the input and hidden layers
        self.weight1 += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
        
        pass
    
    def query(self, inputs):
        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.weight1, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.tanh(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = np.dot(self.weight2, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.tanh(final_inputs)
        
        return final_outputs
    


# Source Data
train_x = np.load('/Users/lordxuzhiyu/Desktop/mnist_data/mnist.train.npy')
train_y = np.load('/Users/lordxuzhiyu/Desktop/mnist_data/mnist.trainlabel.npy')
test_x = np.load('/Users/lordxuzhiyu/Desktop/mnist_data/mnist.test.npy')
test_y = pd.read_csv('/Users/lordxuzhiyu/Desktop/mnist_data/sample_submission.csv', usecols = ['class'])

# Define two functions for preprocessing
# One-hot encoding is used for the labels
def one_hot_encoding(a, classes):
    targets = a.reshape(-1)
    a = np.eye(classes)[targets]
    return a

# Change the shape of the array.
# (AxB * BxC = AxC) 
# e.g We will shape the 56000 * 28 * 28 3-D array to 56000 * 784 2-D array
def matrix_mul(matrix, remain_dimen, pixels):
    matrix = matrix.reshape(matrix.shape[remain_dimen], pixels)
    return matrix

# Prepare X for train and test
pixels = 784
train_x = train_x / 255
test_x = test_x / 255   
train_x = train_x.reshape(train_x.shape[0], pixels)
test_x = test_x.reshape(test_x.shape[0], pixels)        
     
# Prepare Y for train and test
train_y = train_y.reshape(-1, 1)
test_y = test_y.to_numpy()
test_y = test_y.reshape(-1, 1)
train_y = one_hot_encoding(train_y, classes = 10)
test_y = one_hot_encoding(test_y, classes = 10)       


# Parameters
hidden1 = 784
hidden2 = 256
output_unit = 10
learning_rate = 0.1
N = train_x.shape[0]


# Create instance of Class NeuralNetwork
n = NeuralNetwork(hidden1, hidden2, output_unit, learning_rate, N)

# Training
epochs = 50

for epoch in range(epochs):
    print('Train round:', epoch)
    for i in range(N):
        print(i)
        inputs = (train_x[i] / 255.0 * 0.99) + 0.01
        targets = train_y[i]
        n.train(inputs, targets)
        pass
    pass

scorecard = []

size = test_x.shape[0]
for i in range(size):
    print('Test Round:', i)
    result = test_y[i]
    
    inputs = (test_x[i] / 255.0 * 0.99) + 0.01
    outputs = n.run(inputs)
    label = np.argmax(outputs)
    
    if(label == result):
        scorecard.append(1)
        print(label)
    else:
        scorecard.append(0)
        print(result)
        pass
    pass

scorecard_array = np.asarray(scorecard)
print ("performance = ", scorecard_array.sum() / scorecard_array.size)    
