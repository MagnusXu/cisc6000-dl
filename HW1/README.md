# Quesion 3: Build Your Own Neural Network (30 points)
> In this exercise, you will implement a two-layer neural network model (M) to classify the handwritten digits (“0” to “9”) using a subset of MNIST dataset (provided with the assignment in Blackboard). Figure 1 illustrates the architecture of a NN.

- Your code can only use the Numpy and Math packages.
- Your NN will have two hidden layers with 784 (layer1), 256 (layer2) units respectively. The output layer will have 10 units representing the probabilities of input being “0” , “9” or other digit. For example, an output vector of [yˆ0, yˆ1,....yˆ9] represents a probability of being digit “0” to digit “9”. Your model is correct if the output unit with the highest probability corresponds to the image test label.
- Train your model to minimize the cross-entropy loss
- Initialize your weights use 1 / N , where N is the total number of instances in the training data provided in Blackboard.
- Use mini-batch gradient descent with batch size = 32
- Activation Function: tanh
- Validation Set: 20% of training

### Deliverable:
Train your model M with 50 epochs and print out your validation accuracy for each class in your Python output.
Save the predictions of M for the test data in a CSV file and submit it in you KCE. A sample submis- sion CSV file has been provided in Kaggle.

# Question 4: Tensorflow/Keras to the Rescue: (40 points)
> In this exercise, you will build the same Neural Network model (M) in Question 3 using the Tensor-
flow/Keras interface. You should be able to build your model in less than 10 lines of Keras code!

### Deliverable(s):
(a) Train your model M with 50 epochs and learning rate = 0.01. Save the predictions of M on the test data in a CSV file and submit it in your KCE.

(b) Keep other parameters unchanged and experiment with these four optimizers: SGD (with momentum = 0), Adam, Momentum (i.e., SGD with momentum = 0.8), and RMSprop. Fill in Column 1- 4 of Table 1 for each optimizer with 50 epochs of training.

(c) Your test dataset contains some flipped/rotated images as illustrated in Figure 2. In order to classify them properly, you need to implement the following two image augmentation methods and include the new images in your training data.
   
