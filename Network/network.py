from abc import ABC
import numpy as np
import random
import pickle

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
        """
        Reset all partials back to zeros.
        """
        self.partial_weights = np.zeros_like(self.weights)
        self.partial_bias = np.zeros_like(self.bias)
    
    def sum_partials(self, pWeights: np.matrix, pBias: np.array):
        """
        Sum up the partial derivatives of each layer to use for the gradient step.
        """
        self.partial_weights = self.partial_weights + pWeights
        self.partial_bias = (self.partial_bias + pBias).flatten()
    
    def gradient_step(self, eta, mini_batch_size):
        """
        Updates the weights and biases according to the respective gradient sums.
        """
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

        input_neurons = layers[0][0]
        input_sigma = layers[0][1]
        input_dsigma = layers[0][1]
        input_layer = InputLayer(input_neurons, input_sigma, input_dsigma)
        self.layers.append(input_layer)

        for l in range(1, len(layers)):
            num_neurons = layers[l][0]
            prevLayer = self.layers[l - 1]
            sigma = layers[l][1]
            dsigma = layers[l][2]

            layer = WeightedLayer(num_neurons, prevLayer, sigma, dsigma)
            self.layers.append(layer)

        # X is the unscaled input data vector
        self.X = np.zeros(input_layer.neurons)

        # Y is the expected output vector
        self.Y = np.zeros(self.layers[-1].neurons)

    def load_data(self, X: np.array, Y: np.array = None):
        assert(self.layers[0].neurons  == len(X))

        # Y is optional if we are not training the data.
        if Y is not None:
            assert(self.layers[-1].neurons == len(Y))
            self.Y = Y
        
        self.X = X
        self.layers[0].update_values(X)
        self.feed_forward()
    
    def SGD(self, training_data: list, mini_batches: int, eta: float, epochs: int = 10, save_path=None, mini_batch_callback=None, epoch_callback=None):
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
        save_path       ~ relative save path to save the network state after training
        """

        m = mini_batches
        for e in range(epochs):
            random.shuffle(training_data)
            total_samples = len(training_data)
            
            # ptr is used to set a stride for each batch in our training data
            for ptr in range(total_samples // mini_batches):

                batch = training_data[ptr*m : (ptr+1)*m]
                for T in batch:
                    X, Y = T[0], T[1]
                    self.load_data(X, Y)
                    self.backpropogate()
                
                for L in self.layers[1:]:
                    L.gradient_step(eta, m)

                if mini_batch_callback is not None:
                    mini_batch_callback(self, e, ptr)
            
            if epoch_callback is not None:
                epoch_callback(self, e, ptr)

        if save_path is not None:
            print('Done training! Saving network object...')
            NeuralNetwork.save_network_object(self, save_path)
    
    @staticmethod
    def save_network_object(network, save_path: str = './network.pkl'):
        with open(save_path, 'wb') as output:
            pickle.dump(network, output, pickle.HIGHEST_PROTOCOL)
    
    @staticmethod
    def load_network_object(load_path: str = './network.pkl'):
        with open(load_path, 'rb') as input:
            return pickle.load(input)
    
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
    
    def cost(self):
        last_layer = self.layers[-1]
        V = (last_layer.values - self.Y).flatten()

        return np.dot(V, V)
    
    def feed_forward(self):
        for l in range(1, len(self.layers)):
            self.layers[l].update_values(self.layers[l - 1])
    
    def backpropogate(self):
        # Last layer
        ll = self.layers[-1] 
        
        # Delta of last layer
        delta = 2 * (ll.values - self.Y) * ll.dsigma(ll.Z)
        ll.sum_partials(np.outer(self.layers[-2].values, delta), delta)

        L = len(self.layers) - 2
        for i in range(L, 0, -1):
            pl = self.layers[i - 1]
            nl = self.layers[i + 1]
            l = self.layers[i]

            delta = (nl.weights @ delta) * l.dsigma(l.Z)
            l.sum_partials(np.outer(pl.values, delta), delta)


