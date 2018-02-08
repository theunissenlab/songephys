import theano

import numpy as np


def float_x(x):
    return np.asarray(x, dtype=theano.config.floatX)
