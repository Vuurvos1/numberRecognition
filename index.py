import numpy as np  # v1.19.3
from tkinter import *  # v8.6.0
import math
import random
import csv

from PIL import Image, ImageDraw

root = Tk()
root.title("Number recognition")  # window title
root.maxsize(900, 670)  # max size the window can expand to


# Create left and right frames
leftFrame = Frame(root, width=200, height=400, bg="blue")
leftFrame.grid(row=0, column=0, padx=10, pady=5)
rightFrame = Frame(root, width=650, height=400, bg="grey")
rightFrame.grid(row=0, column=1, padx=10, pady=5)

drawCanvas = Canvas(leftFrame, bg="white", height=210, width=210)
img1 = Image.new("L", (210, 210))
draw = ImageDraw.Draw(img1)

neuralNetworkCanvas = Canvas(rightFrame, bg="white", height=650, width=400)


def setup():
    print("setup")
    # drawNode(10, 10, 0.5)

    # generate node positions in layer
    layer1 = []
    layer2 = []
    layer3 = []

    # draw lines

    # layer 1 to 2 connections
    for i in range(20):
        x1 = 10 + 12  # add half node width
        y1 = i * (24) + 40 + 12
        for j in range(20):
            x2 = 60 + 12  # add half node width
            y2 = j * (24) + 40 + 12
            drawLineAA(x1, y1, x2, y2, width=1, color="#000")

    # layer 2 to 3 connections
    for i in range(20):
        x1 = 60 + 12  # add half node width
        y1 = i * (24) + 40 + 12
        for j in range(10):
            x2 = 120 + 24
            y2 = j * (48 + 10) + 48

            drawLineAA(x1, y1, x2, y2, width=1, color="#AAA")

    # draw nodes over lines
    for i in range(20):
        drawNode(10, i * 24 + 40, random.random(), 24)

    for i in range(20):
        drawNode(60, i * 24 + 40, random.random(), 24)

    for i in range(10):
        drawNode(120, i * (48 + 10) + 20, random.random(), 48)


b1 = "up"
# xold, yold = None, None
xold = None
yold = None


def b1down(event):
    global b1, xold, yold  # acces global variables, why are you like this Python???

    b1 = "down"

    xold = event.x
    yold = event.y


def b1up(event):
    global b1, xold, yold

    b1 = "up"

    xold = None
    yold = None


def motion(event):
    global b1, xold, yold

    if b1 == "down":
        if xold is not None and yold is not None:
            # draw it smooth. neat.
            event.widget.create_line(
                xold, yold, event.x, event.y, width=5, fill='#000000', smooth=TRUE)

            # do PIL equivalent
            draw.line([xold, yold, event.x, event.y], 255, 5)

        xold = event.x
        yold = event.y


# trigger neural network on mouse up for preformance
drawCanvas.bind("<Motion>", motion)
drawCanvas.bind("<ButtonPress-1>", b1down)
drawCanvas.bind("<ButtonRelease-1>", b1up)


def lerp(a, b, t):
    return a + (b - a) * t


def round_rectangle(x1, y1, x2, y2, r=25, **kwargs):
    points = (x1+r, y1, x1+r, y1, x2-r, y1, x2-r, y1, x2, y1, x2, y1+r, x2, y1+r, x2, y2-r, x2, y2-r, x2,
              y2, x2-r, y2, x2-r, y2, x1+r, y2, x1+r, y2, x1, y2, x1, y2-r, x1, y2-r, x1, y1+r, x1, y1+r, x1, y1)
    return neuralNetworkCanvas.create_polygon(points, **kwargs, smooth=True)


def drawNode(x, y, fill=1, size=38):
    outWidth = size
    padding = 4
    # width without padding
    innerWidth = outWidth - padding * 2

    x2 = x + outWidth
    y2 = y + outWidth

    round_rectangle(x, y, x2, y2, 10, fill="white", outline="black", width=2)

    x1 = x + padding
    y1 = y + padding

    # interpolate fill value
    x2 = x1 + lerp(0, innerWidth, fill)
    y2 = y + outWidth - padding

    neuralNetworkCanvas.create_rectangle(x1, y1, x2, y2, fill="black")


def drawLineAA(x1, y1, x2, y2, width=2, color="#000"):
    # Antialiasing draw .5px thicker line with approx 33% of color intensity before rendering line
    neuralNetworkCanvas.create_line(
        x1, y1, x2, y2, width=width + 0.5, fill="#AAA")
    neuralNetworkCanvas.create_line(x1, y1, x2, y2, width=width, fill="#000")


def imgSave(event):
    print(event.char)

    if event.char == "s":
        # img1.show()

        # resize image from 210 x 210 px to 21 x 21 px
        img1small = img1.resize((21, 21), Image.ANTIALIAS)
        img1small.show()

        # convert image to (1D) numpy array
        pixels = Image.Image.getdata(img1small)
        npImg = np.array(pixels)

        # array values might need to be inverted
        print(npImg, npImg.size)

        # array = np.random.randint(255, size=(400, 400), dtype=np.uint8)
        # image = Image.fromarray(array)
        # image.save('dist/bbb.png')


root.bind("<Key>", imgSave)

setup()

neuralNetworkCanvas.pack()
drawCanvas.pack()
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
