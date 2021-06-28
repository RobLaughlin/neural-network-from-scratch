from abc import ABC, abstractclassmethod
import numpy as np
import random

def sigmoid(Z: np.array):
    """
    Z ~ The compact form for the weighted sum (W @ L(n-1) + b(n))
    """
    Z = 1.0 / (1.0 + np.exp(-Z))
    return Z

def d_sigmoid(Z: np.array):
    """
    Z ~ The compact form for the weighted sum (W @ L(n-1) + b(n))
    """
    return sigmoid(Z)*(1-sigmoid(Z))

def reLU(Z: np.array):
    """
    Rectified linear unit applied to the weighted sum
    """
    return np.maximum(0, Z)

def d_reLU(Z: np.array):
    """
    Piecewise derivative of reLU
    """
    return (Z > 0).astype(int)


class Layer(ABC):
    def __init__(self, neurons: int, sigma, dsigma):
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
    def __init__(self, neurons: int, sigma, dsigma):
        super().__init__(neurons, sigma=sigma, dsigma=dsigma)
    
    def update_values(self, values: np.array):
        self.values = values
        self.Z = values
    
class WeightedLayer(Layer):
    def __init__(self, neurons: int, prevLayer:Layer, sigma, dsigma):
        super().__init__(neurons, sigma=sigma, dsigma=dsigma)
        
        L0 = prevLayer.neurons
        L1 = neurons

        xavier_multiplier = 1 / np.sqrt(L0 + L1)
        self.weights = np.random.randn(L0, L1) * xavier_multiplier
        self.weights_T = self.weights.T

        self.bias = np.zeros(L1)

        self.partial_weights = np.zeros_like(self.weights)
        self.partial_bias = np.zeros_like(self.bias)

    def update_values(self, prevLayer: Layer):
        self.Z = (self.weights_T @ prevLayer.values) + self.bias
        self.values = self.sigma(self.Z)
    
    def reset_partials(self):
        self.partial_weights = np.zeros_like(self.weights)
        self.partial_bias = np.zeros_like(self.bias)
    
    def update_partials(self, pWeights: np.matrix, pBias: np.array):
        self.partial_weights = self.partial_weights + pWeights
        self.partial_bias = (self.partial_bias + pBias).flatten()
    
    def gradient_step(self, eta, mini_batch_size):
        gradient_scalar = eta / mini_batch_size
        self.weights -= (gradient_scalar * self.partial_weights)
        self.weights_T = self.weights.T
        self.bias -=(gradient_scalar * self.partial_bias)
        self.reset_partials()

class NeuralNetwork:
    def __init__(self, layers: np.array):
        """
        layers is an array of 3-tuples, where
        each tuple represents (number of neurons in respective index layer, sigma function, deriviative of respective sigma function)
        """

        # Must have at least 1 input and 1 output layer in the network
        assert(len(layers) >= 2)

        self.layers = list()

        input_layer = InputLayer(layers[0][0], layers[0][1], layers[0][2])
        self.layers.append(input_layer)

        for l in range(1, len(layers)):
            layer = WeightedLayer(layers[l][0], self.layers[l - 1], layers[l][1], layers[l][2])
            self.layers.append(layer)

        # X is the unscaled input data vector
        self.X = np.zeros(input_layer.neurons)

        # Y is the expected output vector
        self.Y = np.zeros(self.layers[-1].neurons)

    def load_data(self, X: np.array, Y: np.array):
        assert(self.layers[0].neurons  == len(X))
        assert(self.layers[-1].neurons == len(Y))
        self.X = X
        self.Y = Y

        self.layers[0].update_values(X)
        self.feed_forward()
    
    def SGD(self, training_data: list, test_data: list, mini_batches: int, eta, epochs: int = 10):
        """
        Training data is a list of size T where
        T ~ is the total number of training examples.

        For each training example, we want 
        T[0] ~ to be the input vector for our first training example, and
        T[1] ~ to be the expected output vector for our cost function.

        Test data should follow the same format as the training data.

        mini_batches    ~ The size of each mini batch
        eta             ~ The learning rate to use for gradient descent
        epochs          ~ The number of times to run through all training examples
        """

        m = mini_batches
        for e in range(epochs):
            random.shuffle(training_data)
            total_samples = len(training_data)

            for ptr in range(total_samples // mini_batches):

                batch = training_data[ptr*m : (ptr+1)*m]
                for T in batch:
                    X, Y = T[0], T[1]
                    self.load_data(X, Y)
                    self.backpropogate()
                
                for L in self.layers[1:]:
                    L.gradient_step(eta, m)
            
            correct, tests = self.test_predictions(test_data)
            print("Epoch: {0} ~ {1}/{2} correct predictions ({3:.2f}%).".format(e, correct, tests, correct / tests * 100))

    def predict(self):
        """
        Return a tuple with
        t[0] ~ the neuron value with the maximum output
        t[1] ~ the index of the neuron value in the output layer
        """
        output_layer = self.layers[-1].values
        mon_i = np.argmax(output_layer)
        max_output_neuron = output_layer[mon_i]
        
        return (max_output_neuron, mon_i)

    def test_predictions(self, test_data: np.array):
        """
        Returns a ratio of correct predictions to tested predictions.

        Test data is a list of size T where
        T ~ is the total number of test examples.

        For each test example, we want
        T[0] ~ to be the input vector for our first training example, and
        T[1] ~ to be the expected output vector for our cost function.
        """

        random.shuffle(test_data)

        correct = 0
        tested = 0
        for T in test_data:
            X, Y = T[0], T[1]
            self.load_data(X, Y)
            _, predicted_index = self.predict()

            if Y[predicted_index] == 1:
                correct += 1
            
            tested += 1
        
        return (correct, tested)
    
    def cost(self):
        last_layer = self.layers[-1]
        V = (last_layer.values - self.Y).flatten()

        return np.dot(V, V)
    
    def feed_forward(self):
        for l in range(1, len(self.layers)):
            self.layers[l].update_values(self.layers[l - 1])
    
    def backpropogate(self):
        ll = self.layers[-1] # Last layer
        
        # Delta of last layer
        delta = 2 * (ll.values - self.Y) * ll.dsigma(ll.Z)
        ll.update_partials(np.outer(self.layers[-2].values, delta), delta)

        L = len(self.layers) - 2
        for i in range(L, 0, -1):
            pl = self.layers[i - 1]
            nl = self.layers[i + 1]
            l = self.layers[i]

            delta = (nl.weights @ delta) * l.dsigma(l.Z)
            l.update_partials(np.outer(pl.values, delta), delta)


