import numpy as np


def softmax(matrix):
    denom=np.sum(np.exp(matrix),1)
    print('test')