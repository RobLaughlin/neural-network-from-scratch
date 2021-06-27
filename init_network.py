from PIL import Image
import numpy as np
from network import Layer, NeuralNetwork, sigmoid
from data_parser import DataParser, ImageData

def format_image_data(imgdata:ImageData, index, scaledown):
    # Just pick 1 of the RGB values, it's grayscale so it doesn't matter which one.
    # We also format the matrix image data into a column vector with size equal to
    # the number of pixels.
    return imgdata.images[index][:, :, 0].flatten() / scaledown

if __name__ == '__main__':
    scaledown = 50

    imgdata_filepath = 'data/train-images.idx3-ubyte'
    labeldata_filepath = 'data/train-labels.idx1-ubyte'
    imgdata = DataParser.parse_training_image_file(imgdata_filepath)
    labeldata = DataParser.parse_training_label_file(labeldata_filepath)

    
    values = format_image_data(imgdata, 0, scaledown)
    layers = np.array([len(values), 28, 28, 10])

    y= np.zeros(10)
    y[4] = 1

    network = NeuralNetwork(layers=layers)
    network.load_data(values, y)

