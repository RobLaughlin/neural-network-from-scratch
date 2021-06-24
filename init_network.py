from PIL import Image
from network import Layer, NeuralNetwork, sigmoid
from data_parser import DataParser, ImageData

def format_image_data(imgdata:ImageData, index):
    # Just pick 1 of the RGB values, it's grayscale so it doesn't matter which one.
    # We also format the matrix image data into a column vector with size equal to
    # the number of pixels.
    return imgdata.images[index][:, :, 0].flatten().reshape(imgdata.img_width * imgdata.img_height, 1)

def generate_layers(imgdata):
    
    # Initial layer
    layer0_neuron_values = format_image_data(imgdata, 0)
    layer0 = Layer(neurons=len(layer0_neuron_values), prevalues=layer0_neuron_values)

    # Hidden layers
    layer1 = Layer(neurons=imgdata.img_width, prevalues=layer0.values)
    layer2 = Layer(neurons=imgdata.img_height, prevalues=layer1.values)

    # Output layer
    layer3 = Layer(neurons=10, prevalues=layer2.values)
    layers = [layer0, layer1, layer2, layer3]

    return layers

if __name__ == '__main__':
    imgdata_filepath = 'data/train-images.idx3-ubyte'
    labeldata_filepath = 'data/train-labels.idx1-ubyte'
    imgdata = DataParser.parse_training_image_file(imgdata_filepath)
    labeldata = DataParser.parse_training_label_file(labeldata_filepath)

    layers = generate_layers(imgdata)
    network = NeuralNetwork(layers)
    

