import os
from PIL import Image
from Network.DataParser.data_parser import DataParser

def extract_images(imgdata_file, output_dir):
    imgdata = DataParser.parse_training_image_file(imgdata_file)

    print('Extracting {0} images...'.format(imgdata.num_images))
    print('Image size: {0}x{1}'.format(imgdata.img_width, imgdata.img_height))

    for i in range(len(imgdata.images)):
        img = Image.fromarray(imgdata.images[i])
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