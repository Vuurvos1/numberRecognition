# version 1.19.3
import numpy as np
import math

import csv

print("Hello world")


inputs = [1.0, 2.0, 3.0, 2.5]
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2.0, 3.0, 0.5]

output = np.dot(weights, inputs) + biases
print(output)

# Network structure
# inputs hidden layers outputs
# 3      [ 4 ]         1

"""
inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, 0.8]]

weights = [[0.2, 0.8, -0.5, 1.0],
          [0.5, -0.91, 0.26, -0.5],
          [-0.26, -0,27, 0.17, 0.87]]

biases = [2, 3, 0.5]

# output = input[0] * weights[0] + input[1] * weights[1]

# input[0] * weights[0] + input[1] * weights[1] + ... + bias
output = np.dot(inputs, np.array(weights).T) + biases
print(output)
"""


# Network structure
# inputs hidden layers outputs
# 784    [ 20 20 ]     10

def feedForward():
  return

def backPropagate():
  return

def activationFunction(x):
  return np.maximum(0, x) # Relu formula y

"""
with open('dataset/mnist_train_min.csv', newline='') as file:
  dataList = list(csv.reader(file))

  # print(dataList[0])

  # item 0 is value
  # rest is a 28x28 grid of pixel values
  # print(len(dataList[0]))
  image = []
  line = []
  # move data into 3d array for visualizing
  for i in range(len(dataList[0])):
    if i % 28 == 0:
      line.append(dataList[0][i])
      # print(line)
      image.append(line)
      line = []
    else:
      line.append(dataList[0][i])

  print(dataList[0][0])

  # 2d array using numpy
  dataList[0].pop(0)

  image = np.array(dataList[0]).reshape((28,28))
  print(image)
"""