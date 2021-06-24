import numpy as np

def sigmoid(Z:np.array):
    """
    Z ~ The compact form for the weighted sum (W @ L(n-1) + b(n))
    """
    Z = 1 / (1 + np.exp(-Z))
    return Z

def d_sigmoid(Z:np.array):
    """
    Z ~ The compact form for the weighted sum (W @ L(n-1) + b(n))
    """
    denom = np.exp(Z) + 1
    Z = 1 / denom
    Z -= np.square(Z)
    return Z

class Layer:
    def __init__(self, neurons:int, prevalues:np.array, weights:np.matrix=None, biases:np.array=None, sigma=sigmoid, dsigma=d_sigmoid):
        """
        neurons     ~ the number of neurons in the layer
        prevalues   ~ the values of each neuron in the previous layer (or initial supplied data if this is the first layer)
        sigma       ~ the activation function to normalize the weighted sums
        dsigma      ~ the derivative of the sigmoid function
        weights     ~ a (N(L) x N(L-1)) matrix of weights, where N(L) is the number of neurons in layer L.
        biases      ~ a N(L) vector of float64s.
        """

        self.neurons = neurons
        prev_layer_neurons = len(prevalues)
        xavier_multiplier = np.sqrt(1/(prev_layer_neurons + neurons))

        if weights == None:
            weights = np.random.randn(neurons, prev_layer_neurons).astype(np.float64) * xavier_multiplier
        
        if biases == None:
            biases = np.zeros(shape=(neurons, 1))
        
        self.values = sigma((weights @ prevalues) + biases)
        self.weights = weights
        self.biases = biases
        self.sigma = sigma
        self.dsigma = dsigma
    
class NeuralNetwork:
    def __init__(self, layers:list):
        """
        layers is an ordinary python list of Layers where
        layers[0] is the input layer,
        layers[1...L-1] are the hidden layers,
        layers[L] is the output layer.
        """
        self.layers = layers
        pass

