from abc import ABC, abstractclassmethod
import numpy as np

def sigmoid(Z: np.array):
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

class Layer(ABC):
    def __init__(self, neurons: int, sigma=sigmoid, dsigma=d_sigmoid):
        """
        neurons     ~ The number of neurons in the layer
        sigma       ~ The differentiable function to apply to the neurons
        dsigma      ~ The derivative of the sigma function
        """
        
        self.neurons    = neurons
        self.sigma      = sigma
        self.dsigma     = dsigma

        self.values     = np.zeros(neurons, dtype=np.float64)
        self.Z          = np.zeros(neurons, dtype=np.float64) # Shorthand for the weighted sum before being passed through the sigma function

class InputLayer(Layer):
    def __init__(self, neurons: int, sigma=sigmoid, dsigma=d_sigmoid):
        super().__init__(neurons, sigma=sigma, dsigma=dsigma)
    
    def update_values(self, values: np.array):
        self.values = self.sigma(values)
        self.Z = values
    
class WeightedLayer(Layer):
    def __init__(self, neurons: int, prevLayer:Layer, sigma=sigmoid, dsigma=d_sigmoid):
        super().__init__(neurons, sigma=sigma, dsigma=dsigma)
        
        L0 = len(prevLayer.values)
        L1 = neurons
        xavier_multiplier = np.sqrt(1 / (L0 + L1))

        self.weights = np.random.randn(L0, L1) * xavier_multiplier
        self.weights_T = self.weights.T
        self.bias = np.zeros(L1).T

        self.partial_weights = np.zeros_like(self.weights)
        self.partial_bias = np.zeros_like(self.bias)
    
    def update_values(self, prevLayer: Layer):
        self.Z = (self.weights_T @ prevLayer.values) + self.bias
        self.values = self.sigma(self.Z)
    
class NeuralNetwork:
    def __init__(self, layers: np.array):
        """
        layers is an array of integers, where
        layers[0] is the number of neurons in the input layer,
        layers[1,2,...L-1] is the number of neurons in each respective hidden layer, and
        layers[L] is the number of neurons in the output layer.
        """

        # Must have at least 1 input and 1 output layer in the network
        assert(len(layers) >= 2)

        self.layers = list()

        input_layer = InputLayer(layers[0])
        self.layers.append(input_layer)

        for l in range(1, len(layers)):
            layer = WeightedLayer(layers[l], self.layers[l - 1])
            self.layers.append(layer)
        
        # X is the unscaled input data vector
        self.X = None

        # Y is the expected output vector
        self.Y = None
    
    def load_data(self, X: np.array, Y: np.array):
        assert(self.layers[0].neurons  == len(X))
        assert(self.layers[-1].neurons == len(Y))
        self.X = X
        self.Y = Y

        self.layers[0].update_values(X)
        self.feed_forward()
        
    def cost(self):
        last_layer = self.layers[-1]
        V = (last_layer.values - self.Y).flatten()

        return np.dot(V, V)
    
    def feed_forward(self):
        for l in range(1, len(self.layers)):
            self.layers[l].update_values(self.layers[l - 1])
    
    def backpropogate(self):
        pass
