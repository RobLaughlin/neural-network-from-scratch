import numpy as np
from collections import OrderedDict

class DataParser:

    @staticmethod
    def parse_training_image_file(filepath):
        file = open(filepath, 'rb')
        headers = OrderedDict({'magic_number': -1, 'number_of_images': -1, 'width': -1, 'height': -1})
        keys = list(headers)

        for k in keys:
            byte = file.read(4)
            headers[k] = int.from_bytes(byte, byteorder='big')
        
        num_images = headers['number_of_images']
        width = headers['width']
        height = headers['height']
        pixels = width * height

        nubyte = np.dtype(np.uint8)
        nubyte = nubyte.newbyteorder('>')
        images = np.zeros((num_images, width, height, 3), dtype=np.uint8)

        for i in range(num_images):
            imgdata = np.zeros((width, height, 3), dtype=np.uint8)
            buf_img = np.frombuffer(file.read(pixels), dtype=nubyte)
            buf_img = buf_img.reshape(width, height)
            imgdata[:, :, 0] = imgdata[:, :, 1] = imgdata[:, :, 2] = buf_img
            images[i, :] = imgdata
        
        return (headers, images)
