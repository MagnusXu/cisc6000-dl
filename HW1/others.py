import pandas as pd
import requests
import re, random
import nltk

import numpy as np

train_x = np.load('/Users/lordxuzhiyu/Desktop/mnist_data/mnist.train.npy')
train_y = np.load('/Users/lordxuzhiyu/Desktop/mnist_data/mnist.trainlabel.npy')
test_x = np.load('/Users/lordxuzhiyu/Desktop/mnist_data/mnist.test.npy')
test_y = pd.read_csv('/Users/lordxuzhiyu/Desktop/mnist_data/sample_submission.csv', usecols = ['class'])


train_y = train_y.reshape(-1, 1)
test_y = test_y.to_numpy()
test_y = test_y.reshape(-1, 1)

#train_y = (np.arange(train_y.max()) == train_y[...,None]-1).astype(int)

def one_hot_encoding(a, classes):
    targets = a.reshape(-1)
    a = np.eye(classes)[targets]
    return a

def matrix_mul(matrix, remain_dimen, pixels):
    matrix = matrix.reshape(matrix.shape[remain_dimen], pixels)
    return matrix

#train_y = matrix_mul(train_y, remain_dimen = 0, pixels = 9)
    
train_y = one_hot_encoding(train_y, classes = 10)
test_y = one_hot_encoding(test_y, classes = 10)

train_x = train_x / 255
test_x = test_x / 255

pixels = 784
train_x = train_x.reshape(train_x.shape[0], pixels)
test_x = test_x.reshape(test_x.shape[0], pixels)

print(type(train_x), train_x.shape)
print(type(train_y), train_y.shape)
print(type(test_x), test_x.shape)
print(type(test_y), test_y.shape)

