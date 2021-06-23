import os
from collections import OrderedDict
import numpy as np
from PIL import Image
from data_parser import DataParser

def extract_images(imgdata_file, output_dir):
    headers, images = DataParser.parse_training_image_file(imgdata_file)
    print('Extracting {0} images...'.format(headers['number_of_images']))
    print('Image size: {0}x{1}'.format(headers['width'], headers['height']))
    for i in range(len(images)):
        img = Image.fromarray(images[i])
        name = str(i) + '.png'

        print('Saving image {0}...'.format(name))
        img.save(output_dir + '/' + name)

if __name__ == '__main__':
    training_image_filepath = 'data/'
    training_image_file = training_image_filepath + 'train-images.idx3-ubyte'
    training_output_dir = training_image_filepath + 'images'
    if not os.path.exists(training_output_dir):
        os.makedirs(training_output_dir)

    test_image_filepath = 'data/testdata/'
    test_image_file = test_image_filepath + 't10k-images.idx3-ubyte'
    test_output_dir = test_image_filepath + 'images'
    if not os.path.exists(test_output_dir):
        os.makedirs(test_output_dir)

    extract_images(training_image_file, training_output_dir)
    extract_images(test_image_file, test_output_dir)