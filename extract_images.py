import os
from collections import OrderedDict
import numpy as np
from PIL import Image

def parse_headers(file):
    headers = OrderedDict({'magic_number': -1, 'number_of_images': -1, 'width': -1, 'height': -1})
    keys = list(headers)

    for k in keys:
        byte = file.read(4)
        headers[k] = int.from_bytes(byte, byteorder='big')
    
    return file, headers

def save_images(file, num_images, width, height, output_dir):
    nubyte = np.dtype(np.uint8)
    nubyte = nubyte.newbyteorder('>')

    pixels = width * height
    for i in range(num_images):
        imgdata = np.zeros((width, height, 3), dtype=np.uint8)
        buf_img = np.frombuffer(file.read(pixels), dtype=nubyte)
        buf_img = buf_img.reshape(width, height)
        imgdata[:, :, 0] = imgdata[:, :, 1] = imgdata[:, :, 2] = buf_img

        img = Image.fromarray(imgdata)
        img.save(output_dir + '/' + str(i) + '.png')

def extract_images(imgdata_file, output_dir):
    f_timages = open(imgdata_file, 'rb')
    f_timages, headers = parse_headers(f_timages)

    images = headers['number_of_images']
    width = headers['width']
    height = headers['height']

    save_images(f_timages, images, width, height, output_dir)

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