import numpy as np  # v1.19.3
from tkinter import *  # v8.6.0
import math
import csv

print("Hello world")

root = Tk()
canvas = Canvas(root, bg="blue", height=250, width=300)


def round_rectangle(x1, y1, x2, y2, r=25, **kwargs):
    points = (x1+r, y1, x1+r, y1, x2-r, y1, x2-r, y1, x2, y1, x2, y1+r, x2, y1+r, x2, y2-r, x2, y2-r, x2,
              y2, x2-r, y2, x2-r, y2, x1+r, y2, x1+r, y2, x1, y2, x1, y2-r, x1, y2-r, x1, y1+r, x1, y1+r, x1, y1)
    return canvas.create_polygon(points, **kwargs, smooth=True)


my_rectangle = round_rectangle(50, 50, 150, 100, r=20, fill="red")

canvas.pack()
root.mainloop()

# Network structure
# inputs outputs
# 4      3
inputs = [1.0, 2.0, 3.0, 2.5]
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2.0, 3.0, 0.5]

output = np.dot(weights, inputs) + biases
# print(output)

# Network structure
# inputs hidden layers outputs
# 4      [ 3 3 ]       1


inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, 0.8]]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]

biases2 = [-1, 2, 0.5]

# output = input[0] * weights[0] + input[1] * weights[1]

# input[0] * weights[0] + input[1] * weights[1] + ... + bias
layer1Outputs = np.dot(inputs, np.array(weights).transpose()) + biases
layer2Outputs = np.dot(layer1Outputs, np.array(weights2).transpose()) + biases2
# print(layer2Outputs)


# np.random.seed(0)

inputs = [1, 2, 3, 2.5]


class NeuralLayer:
    def __init__(self, inputs, neurons):
        # inputs x neursons
        self.weights = 0.1 * np.random.randn(inputs, neurons)
        self.biases = np.zeros((1, neurons))

    def feedForward(self, inputs):
        # input[0] * weights[0] + input[1] * weights[1] + ... + bias
        self.output = np.dot(inputs, self.weights) + self.biases

    def backPropagate(self):
        pass


class activationFunctionReLu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


layer1 = NeuralLayer(4, 5)
activation1 = activationFunctionReLu()

layer2 = NeuralLayer(5, 2)
activation2 = activationFunctionReLu()

layer1.feedForward(inputs)
activation1.forward(layer1.output)
print(activation1.output)

layer2.feedForward(activation1.output)
activation2.forward(layer2.output)
print(activation2.output)


# Network structure
# inputs hidden layers outputs
# 784    [ 20 20 ]     10

def feedForward():
    return


def backPropagate():
    return


def activationFunction(inputs):
    return np.maximum(0, inputs)  # ReLu formula


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
