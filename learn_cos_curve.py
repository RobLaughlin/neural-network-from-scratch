import matplotlib.pyplot as plt
import numpy as np
from Network.network import NeuralNetwork, sigmoid, d_sigmoid
from math import pi
from itertools import repeat

fig, ax = plt.subplots()
ax.plot()
plt.pause(2)

def update_title(epoch: int, mini_batch: int):
    global ax
    ax.set_ylabel('cos(x)')
    ax.set_xlabel('-2pi to 2pi')
    ax.set_title("Epoch: {0} ~ Mini batch: {1}".format(epoch, mini_batch))

def epoch_callback(network: NeuralNetwork, epoch: int, mini_batch: int):
    update_title(epoch, mini_batch)
    print(epoch)
    if epoch == 0:
        plt.pause(10)

def mini_batch_callback(network: NeuralNetwork, epoch: int, mini_batch: int):
    global ax
    ax.cla()
    ax.plot(network.layers[0].values, np.cos(network.layers[0].values), 'k--')
    ax.plot(network.layers[0].values, (network.layers[-1].values * 2) - 1, linewidth=3)
    update_title(epoch, mini_batch)
    plt.pause(0.0166)

if __name__ == '__main__':
    layers = np.array([
        (100, sigmoid, d_sigmoid),
        (100, sigmoid, d_sigmoid),
        (100, sigmoid, d_sigmoid),
    ])
    
    inputs = np.linspace(-2 * pi, 2 * pi, 100)

    # Need to squeeze the outputs between 0 and 1
    outputs = (np.cos(inputs) + 1) / 2
    expected = (inputs, outputs)

    num_examples = 5000
    eta = 0.05
    epochs = 1
    mini_batches = 50

    training_examples = list(repeat(expected, num_examples))
    network = NeuralNetwork(layers=layers)
    network.SGD(
        training_examples, mini_batches=mini_batches, eta=eta, epochs=epochs, 
        epoch_callback=epoch_callback, mini_batch_callback=mini_batch_callback
    )

