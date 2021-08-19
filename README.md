# Number Recognition

The data used for training the neural network is the [MNIST in csv](https://pjreddie.com/projects/mnist-in-csv/) dataset, which is based on the [MNIST database of handwritten digits](http://yann.lecun.com/exdb/mnist/)

![Interface of number recognition app](./docs/UI.png)

**Made using:**

- Python 3.9.6
- Numpy 1.19.5
- tkinter 8.6
- Pillow 8.3.1
- Pandas 1.3.1

**Note**
Although the neural network can get around 95% accuracy on the testing dataset it still has some trouble with recognizing the drawn digits from the canvas,
this is most likely because of the resizing and process of the canvas, which gets the input close to the dataset but not exact.
While drawing a digit it can help drawing it close to the same size as the canvas, although not always sadly.

## Sources

<!-- py -m pip -->

- 3Blue1Brown. (2017). Neural networks. https://www.3blue1brown.com/neural-networks
- Lague, S. (2018). Neural Networks. YouTube. https://www.youtube.com/playlist?list=PLFt_AvWsXl0frsCrmv4fKfZ2OQIwoUuY
- Sentdex. (2020). Neural Networks from Scratch in Python. YouTube. https://www.youtube.com/playlist?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3
- Shiffman, D. (2018, May 1). 10: Neural Networks - The Nature of Code. YouTube. https://www.youtube.com/playlist?list=PLRqwX-V7Uu6aCibgK1PTWWu9by6XFdCfh
