import numpy as np
from collections import OrderedDict

class DataParser:

    @staticmethod
    def _parse_headers(file, format):
        keys = list(format)

        for k in keys:
            byte = file.read(4)
            format[k] = int.from_bytes(byte, byteorder='big')
        return file, format
    
    @staticmethod
    def parse_training_image_file(filepath):
        """
        ---Data format---

        [offset] [type]          [value]          [description]
        0000     32 bit integer  0x00000803(2051) magic number
        0004     32 bit integer  60000            number of images
        0008     32 bit integer  28               number of rows
        0012     32 bit integer  28               number of columns
        0016     unsigned byte   ??               pixel
        0017     unsigned byte   ??               pixel
        ........
        xxxx     unsigned byte   ??               pixel

        Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
        """
        
        file = open(filepath, 'rb')
        headers = OrderedDict({'magic_number': -1, 'number_of_images': -1, 'width': -1, 'height': -1})
        file, headers = DataParser._parse_headers(file, headers)

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

    @staticmethod
    def parse_training_label_file(filepath):
        """
        [offset] [type]          [value]          [description]
        0000     32 bit integer  0x00000801(2049) magic number (MSB first)
        0004     32 bit integer  10000            number of items
        0008     unsigned byte   ??               label
        0009     unsigned byte   ??               label
        ........
        xxxx     unsigned byte   ??               label

        The labels values are 0 to 9.
        """

        file = open(filepath, 'rb')
        headers = OrderedDict({'magic_number': -1, 'number_of_labels': -1})
        file, headers = DataParser._parse_headers(file, headers)
        
        num_labels = headers['number_of_labels']

        nubyte = np.dtype(np.uint8)
        nubyte = nubyte.newbyteorder('>')

        labels = np.frombuffer(file.read(num_labels), dtype=nubyte)
        return headers, labels

