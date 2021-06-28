from Network.network import NeuralNetwork
from Network.DataParser.data_parser import DataParser
from Network.train_network import generate_training_examples

if __name__ == '__main__':
    scaledown = 255

    network = NeuralNetwork.load_network_object('./network.pkl')

    test_imgdata_filepath = 'data/testdata/t10k-images.idx3-ubyte'
    test_labeldata_filepath = 'data/testdata/t10k-labels.idx1-ubyte'
    test_imgdata = DataParser.parse_training_image_file(test_imgdata_filepath)
    test_labeldata = DataParser.parse_training_label_file(test_labeldata_filepath)

    test_examples = generate_training_examples(test_imgdata, test_labeldata, scaledown)
    correct, tested = network.test_predictions(test_examples)
    print("{0}/{1} digits guessed correctly ({2:.2f}%).".format(correct, tested, correct / tested * 100))