import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from Network.network import NeuralNetwork, reLU, d_reLU, sigmoid, d_sigmoid
from Network.DataParser.data_parser import DataParser, ImageData, LabelData
import pickle

def format_image_data(imgdata:ImageData, index, scaledown):
    # Just pick 1 of the RGB values, it's grayscale so it doesn't matter which one.
    # We also format the matrix image data into a column vector with size equal to
    # the number of pixels.
    return imgdata.images[index][:, :, 0].flatten() / scaledown

def format_label_data(labeldata:LabelData, index):
    """
    Format the label data into a vector with the same size as the output vector
    """
    label_vec = np.zeros(10, dtype=np.uint8)
    label_vec[labeldata.labels[index]] = 1
    return label_vec

def generate_training_examples(imgdata: ImageData, labeldata: LabelData, scaledown):
    """
    Generate a list of (img, label) tuples where
    img     ~ contains the input vector with each component being a pixel
    label   ~ is a vector with the same size as the output vector
    """
    assert(imgdata.num_images == labeldata.num_labels)
    training_examples = list()
    for T in range(imgdata.num_images):
        (img, label) = (format_image_data(imgdata, T, scaledown), format_label_data(labeldata, T))
        training_examples.append((img, label))
    
    return training_examples

if __name__ == '__main__':
    scaledown = 255
    mini_batch_size = 15
    eta = 0.7
    epochs = 10
    num_outputs = 10

    # Training data
    imgdata_filepath = 'data/train-images.idx3-ubyte'
    labeldata_filepath = 'data/train-labels.idx1-ubyte'
    imgdata = DataParser.parse_training_image_file(imgdata_filepath)
    labeldata = DataParser.parse_training_label_file(labeldata_filepath)

    # Test data
    test_imgdata_filepath = 'data/testdata/t10k-images.idx3-ubyte'
    test_labeldata_filepath = 'data/testdata/t10k-labels.idx1-ubyte'
    test_imgdata = DataParser.parse_training_image_file(test_imgdata_filepath)
    test_labeldata = DataParser.parse_training_label_file(test_labeldata_filepath)

    training_examples = generate_training_examples(imgdata, labeldata, scaledown)
    test_training_examples = generate_training_examples(test_imgdata, test_labeldata, scaledown)

    layers = np.array([
        (imgdata.img_width * imgdata.img_height, sigmoid, d_sigmoid), 
        (imgdata.img_width, sigmoid, d_sigmoid), 
        (imgdata.img_height, sigmoid, d_sigmoid), 
        (num_outputs, sigmoid, d_sigmoid)
    ])
    
    network = NeuralNetwork(layers=layers)
    network.SGD(training_examples, test_training_examples, mini_batches=mini_batch_size, eta=eta, epochs=epochs)
