import numpy as np  # v1.19.3
import pandas as pd  # v1.3.1
from tkinter import *  # v8.6.0
import math
from PIL import Image, ImageDraw, ImageEnhance

# tkinter stuff
root = Tk()
root.title("Number recognition")  # window title
root.maxsize(900, 670)  # max size the window can expand to


# Create left and right frames
leftFrame = Frame(root, width=200, height=400, bg="blue")
leftFrame.grid(row=0, column=0, padx=10, pady=5)
rightFrame = Frame(root, width=650, height=400, bg="grey")
rightFrame.grid(row=0, column=1, padx=10, pady=5)

drawCanvas = Canvas(leftFrame, bg="white", height=280, width=280)
img1 = Image.new("L", (280, 280))
draw = ImageDraw.Draw(img1)


def clear_canvas():
    drawCanvas.delete("all")


btn = Button(leftFrame, text="Clear",
             activebackground="green", command=clear_canvas)


neuralNetworkCanvas = Canvas(rightFrame, bg="white", height=650, width=400)


# neural network stuff
data = pd.read_csv('dataset/mnist_train.csv', header=None)
data.head()
# print(data.head())

data = np.array(data)
m, n = data.shape

# data_dev = data.

data_train = data[0:1000].T
# print(data_train)
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255

# print(X_train[:, 0].shape)
# print(Y_train)


def init_params():
    # input to layer 1
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5

    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5

    return W1, b1, W2, b2


def ReLU(Z):
    return np.maximum(0, Z)


def derivReLU(Z):
    return Z > 0


def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A


def forwardProp(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)

    # layer 2 (output)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2


def oneHot(Y):
    oneHotY = np.zeros((Y.size, Y.max() + 1))
    oneHotY[np.arange(Y.size), Y] = 1
    oneHotY = oneHotY.T
    return oneHotY


def backwardProp(Z1, A1, Z2, A2, W1, W2, X, Y):
    m = Y.size

    oneHotY = oneHot(Y)
    dZ2 = A2 - oneHotY
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)

    dZ1 = W2.T.dot(dZ2) * derivReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)

    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2

    return W1, b1, W2, b2


def get_predictions(A2):
    return np.argmax(A2, 0)


def get_accuracy(predictions, Y):
    # print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


def greadient_decent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()

    for i in range(iterations):
        Z1, A1, Z2, A2 = forwardProp(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backwardProp(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(
            W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        if i % 10 == 0:
            # print("Iteration ", i)
            print("Accuracy ", get_accuracy(get_predictions(A2), Y))

    return W1, b1, W2, b2


# train network
W1, b1, W2, b2 = greadient_decent(X_train, Y_train, 0.1, 500)


def make_prediction(X, W1, b1, W2, b2):
    _, _, _, A2 = forwardProp(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions


def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_prediction(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]

    print("Prediction ", prediction)
    print("Label ", label)


# test_prediction(0, W1, b1, W2, b2)
# test_prediction(1, W1, b1, W2, b2)
# test_prediction(2, W1, b1, W2, b2)
# test_prediction(3, W1, b1, W2, b2)
test_prediction(4, W1, b1, W2, b2)


# other stuff

def setup():
    # print("setup")
    # drawNode(10, 10, 0.5)

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
        drawNode(10, i * 24 + 40, np.random.rand(), 24)

    for i in range(20):
        drawNode(60, i * 24 + 40, np.random.rand(), 24)

    for i in range(10):
        drawNode(120, i * (48 + 10) + 20, 0, 48)


button1 = "up"
# xold, yold = None, None
xold = None
yold = None


def mouse1down(event):
    global button1, xold, yold  # acces global variables, why are you like this Python???

    button1 = "down"

    xold = event.x
    yold = event.y


def mouse1up(event):
    global button1, xold, yold

    button1 = "up"

    xold = None
    yold = None

    process_canvas()


def motion(event):
    global button1, xold, yold

    if button1 == "down":
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
drawCanvas.bind("<ButtonPress-1>", mouse1down)
drawCanvas.bind("<ButtonRelease-1>", mouse1up)


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


def process_canvas():
    # img1.show()

    # resize image to 28 x 28 px
    img1small = img1.resize((28, 28), Image.ANTIALIAS)
    # boost image contrast for better output
    img1small = ImageEnhance.Contrast(img1small).enhance(2)
    # img1small.show()

    # convert image to (1D) numpy array
    pixels = Image.Image.getdata(img1small)
    npImg = np.array(pixels)

    npImg = np.reshape(npImg, (-1, 1))
    npImg = npImg / 255

    print(make_prediction(npImg, W1, b1, W2, b2))

    _, _, _, A2 = forwardProp(W1, b1, W2, b2, npImg)
    # print(A2)

    # draw output nodes
    A2 = A2.flatten()
    for i in range(A2.size):
        drawNode(120, i * (48 + 10) + 20, A2[i], 48)


def keyPress(event):
    print(event.char)

    # if event.char == "s":
    # process_canvas()


root.bind("<Key>", keyPress)

setup()

neuralNetworkCanvas.pack()

drawCanvas.pack()
btn.pack()

root.mainloop()

"""
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


# def activationFunction(inputs):
#     return np.maximum(0, inputs)  # ReLu formula
"""
