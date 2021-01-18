# version 1.19.3
import numpy as np
import math

import csv

print("Hello world")

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