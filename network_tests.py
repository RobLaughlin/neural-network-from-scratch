import numpy as np
from numpy.matrixlib import matrix
import network
from data_parser import DataParser

def test_sigmoid(M:np.matrix):
    print(M)
    print("Applying sigmoid...")
    print(network.sigmoid(M))

def test_dsigmoid(M:np.matrix):
    print(M)
    print("Applying derivative of sigmoid...")
    print(network.d_sigmoid(M))

def test_imagedata_parsing(filepath):
    print(DataParser.parse_training_image_file(filepath))

def test_labeldata_parsing(filepath):
    print(DataParser.parse_training_label_file(filepath))

if __name__ == '__main__':
    # test_matrix = np.matrix(np.linspace(start=1, stop=5, num=5, dtype=np.float64))
    # test_sigmoid(test_matrix)
    # test_dsigmoid(test_matrix)
    # test_imagedata_parsing('data/train-images.idx3-ubyte')
    # test_labeldata_parsing('data/train-labels.idx1-ubyte')
    
    pass
