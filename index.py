import numpy as np  # v1.19.3
import pandas as pd  # v1.3.1
from tkinter import *  # v8.6.0
import math
import json
import os
from PIL import Image, ImageDraw, ImageEnhance

# tkinter stuff
root = Tk()
root.title("Number recognition")  # window title
root.maxsize(850, 600)  # max size the window can expand to
root.geometry("850x600")  # set fixed window size

col_width = 850/2

# Create left and right frames
leftFrame = Frame(root, width=425)
leftFrame.pack(side=LEFT, expand=True)
rightFrame = Frame(root, bg="blue")
rightFrame.pack(side=RIGHT, expand=True)

drawCanvas = Canvas(leftFrame, bg="white", height=200,
                    width=200, cursor="pencil")
drawCanvas.pack(pady=(0, 16))
img1 = Image.new("L", (232, 232))
draw = ImageDraw.Draw(img1)


def clear_canvas():
    draw.rectangle([(0, 0), (280, 280)], fill="#000000")
    drawCanvas.delete("all")


btn = Button(leftFrame, text="clear",
             activebackground="#E5E5E5", borderwidth=1, relief="solid",
             cursor="hand1", padx=12, pady=4, command=clear_canvas)
btn.pack(side=RIGHT, padx=(0, 2))

neuralNetworkCanvas = Canvas(rightFrame, height=600, width=425)


# get training and test data
def format_data(path, size):
    data = pd.read_csv(path, header=None)
    # print(data.head())

    data = np.array(data)
    m, n = data.shape
    data = data[0:size].T

    x_data = data[1:n] / 255
    y_data = data[0]

    return x_data, y_data


X_train, Y_train = format_data('dataset/mnist_train.csv', 25000)
X_test, Y_test = format_data('dataset/mnist_test.csv', 1000)


class ReLU:
    def forward(self, input):
        return np.maximum(0, input)

    def backward(self, Z):
        return Z > 0


def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A


def make_prediction(x_input):
    _, _, _, A2 = net.forward(x_input)
    predictions = get_predictions(A2)
    return predictions


def get_predictions(A2):
    return np.argmax(A2, 0)


def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size


def oneHot(Y):
    oneHotY = np.zeros((Y.size, Y.max() + 1))
    oneHotY[np.arange(Y.size), Y] = 1
    oneHotY = oneHotY.T
    return oneHotY


class Neural_Network:
    def __init__(self):
        """
        # Network structure
        # inputs hidden layers outputs
        # 784    [ 12 ]        10
        """

        if os.path.exists('./output/data.json'):
            self.W1, self.b1, self.W2, self.b2 = self.get_weights()
            return

        # input to layer 1
        self.W1 = np.random.rand(12, 784) - 0.5
        self.b1 = np.random.rand(12, 1) - 0.5

        # hidden layer
        self.W2 = np.random.rand(10, 12) - 0.5
        self.b2 = np.random.rand(10, 1) - 0.5

    def forward(self, x_input):
        self.Z1 = self.W1.dot(x_input) + self.b1
        self.A1 = ReLU().forward(self.Z1)

        # layer 2 (output)
        self.Z2 = self.W2.dot(self.A1) + self.b2
        self.A2 = softmax(self.Z2)

        return self.Z1, self.A1, self.Z2, self.A2

    def backward(self, x_input, y_input, alpha):
        m = y_input.size

        oneHotY = oneHot(y_input)
        dZ2 = self.A2 - oneHotY
        dW2 = 1 / m * dZ2.dot(self.A1.T)
        db2 = 1 / m * np.sum(dZ2)

        dZ1 = self.W2.T.dot(dZ2) * ReLU().backward(self.Z1)
        dW1 = 1 / m * dZ1.dot(x_input.T)
        db1 = 1 / m * np.sum(dZ1)

        # update parameters
        self.W1 = self.W1 - alpha * dW1
        self.b1 = self.b1 - alpha * db1
        self.W2 = self.W2 - alpha * dW2
        self.b2 = self.b2 - alpha * db2

        return dW1, db1, dW2, db2

    def train(self, x_train, y_train, epochs, alpha):
        # reshuffle weights and biases
        self.W1 = np.random.rand(12, 784) - 0.5
        self.b1 = np.random.rand(12, 1) - 0.5

        self.W2 = np.random.rand(10, 12) - 0.5
        self.b2 = np.random.rand(10, 1) - 0.5

        for i in range(epochs):
            for j in range(len(x_train)):
                Z1, A1, Z2, A2 = self.forward(x_train)
                dW1, db1, dW2, db2 = self.backward(x_train, y_train, alpha)

            if i % 1 == 0:
                print("Epoch ", i, "   Accuracy",
                      get_accuracy(get_predictions(self.A2), y_train))

        self.save_weights()
        return self.W1, self.b1, self.W2, self.b2

    def predict(self, input):
        _, _, _, A2 = self.forward(input)
        return get_predictions(A2)

    def test(self, index):
        current_image = X_test[:, index, None]
        prediction = make_prediction(current_image)
        label = Y_test[index]

        print("Prediction ", prediction, "   Label ", label)

    def get_weights(self):
        # get weights and biases from file
        with open('output/data.json') as file_object:
            data = json.load(file_object)

        # convert data to numpy arrays
        W1 = np.array(data['W1'])
        b1 = np.array(data['b1'])

        W2 = np.array(data['W2'])
        b2 = np.array(data['b2'])

        return W1, b1, W2, b2

    def save_weights(self):
        # save weights and biases to json file
        data = {"W1": self.W1.tolist(), "b1": self.b1.tolist(),
                "W2": self.W2.tolist(), "b2": self.b2.tolist()}
        jsonString = json.dumps(data)
        jsonFile = open("output/data.json", "w")
        jsonFile.write(jsonString)
        jsonFile.close()

        print("Saved weights and biases")


# setup neural network
net = Neural_Network()
# net.train(X_train, Y_train, 10, 0.1)
for i in range(5):
    net.test(i)

# other stuff


def setup():
    # draw lines
    # layer 1 to 2 connections
    # for i in range(20):
    #     x1 = 10 + 12  # add half node width
    #     y1 = i * (24) + 40 + 12
    #     for j in range(20):
    #         x2 = 60 + 12  # add half node width
    #         y2 = j * (24) + 40 + 12
    #         drawLineAA(x1, y1, x2, y2, width=1, color="#000")

    # layer 2 to 3 connections
    # for i in range(20):
    #     x1 = 60 + 12  # add half node width
    #     y1 = i * (24) + 40 + 12
    #     for j in range(10):
    #         x2 = 120 + 24
    #         y2 = j * (48 + 10) + 48

    #         drawLineAA(x1, y1, x2, y2, width=1, color="#AAA")

    # draw nodes over lines
    for i in range(12):
        drawNode(48, i * (24 + 5) + 260 / 2, 0, 24)

    for i in range(10):
        drawNode(col_width / 2 - 16, i * (46 + 10) + 24, 0, 46)

        neuralNetworkCanvas.create_text(
            col_width / 2 + 60, i * (46 + 10) + 48, font="Arial 20", text=str(i))

    process_canvas()


button1 = "up"
drawCanvas.old_coords = None


def mouse1down(event):
    global button1, xold, yold  # acces global variables, why are you like this Python???

    button1 = "down"

    xold = event.x
    yold = event.y


def mouse1up(event):
    global button1

    button1 = "up"
    drawCanvas.old_coords = None

    process_canvas()


def motion(event):
    global button1, xold, yold

    if button1 == "down":
        x, y = event.x, event.y
        if drawCanvas.old_coords:
            x1, y1 = drawCanvas.old_coords

            # draw line on canvas
            drawCanvas.create_line(x, y, x1, y1, width=5)
            # draw line for neural network, add offset to create room around number
            draw.line([x+16, y+16, x1+16, y1+16], 255, 16)

        drawCanvas.old_coords = x, y


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
    x2 = x1 + lerp(0, innerWidth, np.clip(fill, 0, 1))
    y2 = y + outWidth - padding

    neuralNetworkCanvas.create_rectangle(x1, y1, x2, y2, fill="black")


def drawLineAA(x1, y1, x2, y2, width=2, color="#000"):
    # Antialiasing draw .5px thicker line with approx 33% of color intensity before rendering line
    neuralNetworkCanvas.create_line(
        x1, y1, x2, y2, width=width + 0.5, fill="#AAA")
    neuralNetworkCanvas.create_line(x1, y1, x2, y2, width=width, fill="#000")


def process_canvas():
    # resize image to 28 x 28 px, maybe remove anti aliasing
    img1small = img1.resize((28, 28), Image.ANTIALIAS)
    # img1small.show()

    # convert image to (1d) numpy array
    pixels = Image.Image.getdata(img1small)
    npImg = np.array(pixels)

    npImg = np.reshape(npImg, (-1, 1))
    npImg = npImg / 255

    # print(make_prediction(npImg, W1, b1, W2, b2))

    _, A1, _, A2 = net.forward(npImg)

    # draw hidden layer nodes
    A1 = A1.flatten()
    for i in range(A1.size):
        drawNode(48, i * (24 + 5) + 260 / 2, A1[i], 24)

    # draw output nodes
    A2 = A2.flatten()
    for i in range(A2.size):
        drawNode(col_width / 2 - 16, i * (46 + 10) + 24,  A2[i], 46)


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
