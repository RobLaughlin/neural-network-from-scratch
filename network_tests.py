import numpy as np
from numpy.matrixlib import matrix
import network

def test_sigmoid(M:np.matrix):
    print(M)
    print("Applying sigmoid...")
    print(network.sigmoid(M))

def test_dsigmoid(M:np.matrix):
    print(M)
    print("Applying derivative of sigmoid...")
    print(network.d_sigmoid(M))


if __name__ == '__main__':
    # test_matrix = np.matrix(np.linspace(start=1, stop=5, num=5, dtype=np.float64))
    # test_sigmoid(test_matrix)
    # test_dsigmoid(test_matrix)

    pass
