from __future__ import division, print_function

import numpy as np

import theano
import theano.tensor as T

from util import float_x


# How do we organize this code

def compute_lags(data, n_lags):
    """Compute time-lagged data

    Params
    data (tensor): n_samples x n_features
    n_lags (int): number of time lags to compute

    Returns
    result (tensor): n_samples x n_features x n_lags
    """
    return theano.scan(
        fn=lambda lag, x: T.roll(x, lag, axis=0),
        sequences=T.arange(n_lags),
        non_sequences=[data]
    )[0].dimshuffle(1, 2, 0)


def valid_slice(n_lags):
    """Return the range in which data is valid after computing lags

    Returns
    result (python slice object)
    """
    return slice(n_lags - 1, -1)


def target_slice(n_lags):
    """Return the range of data that will be the response variables"""
    return slice(n_lags // 2, -1 - n_lags // 2)


def timelagged_autocorr(data_lagged):
    """
    Params
    data_lagged (tensor): n_samples x n_elec x n_lags

    Returns
    autocorr (tensor): n_elec x n_lags x n_elec x n_lags
    """
    n_samples, n_elec, n_lags = data_lagged.shape
    data = data_lagged.reshape((n_samples, n_elec * n_lags))

    return T.dot(data.T, data).reshape((n_elec, n_lags, n_elec, n_lags))


def timelagged_xcorr(data_lagged, target):
    """
    Params
    data_lagged (tensor): n_samples x n_elec x n_lags
    target (tensor): n_samples x n_elec

    Returns
    xcorr (tensor): n_elec x n_lags x n_elec
    """
    n_samples, n_elec, n_lags = data_lagged.shape
    data = data_lagged.reshape((n_samples, n_elec * n_lags))

    return T.dot(data.T, target).reshape((n_elec, n_lags, n_elec))


'''
A normal vector autocorrelation would now look like:

>>> data = T.fmatrix()
>>> n_lags = T.iscalar()
>>> data_lagged = compute_lags(data, n_lags)
>>> data_lagged = data_lagged[valid_slice(n_lags)]
>>> target = data[target_slice(n_lags)]
>>> XX = timelagged_autocorr(data_lagged, n_lags)
>>> XY = timelagged_xcorr(data_lagged, n_lags)
'''

def remove_index(idx, mat, along_axes=(0,)):
    """Possibly hacky way to remove a single index along multiple axes

    Params
    idx (int)
    mat (tensor): input tensor
    along_axes (tuple): dimensions along which to remove idx

    Returns
    result (tensor): len(along_axes) x mat.shape
    """
    for axis in along_axes:
        if axis != 0:
            mat = T.swapaxes(mat, 0, axis)
        mat = T.roll(mat, -idx-1, axis=0)
        mat = mat[:-1]
        mat = T.roll(mat, idx, axis=0)
        if axis != 0:
            mat = T.swapaxes(mat, 0, axis)

    return mat


def remove_indices(mat, axes=(0,)):
    # All the axes should be the same length so just pick one.
    roll_amounts = mat.shape[axes[0]]

    return theano.scan(
        fn=lambda idx, mat: remove_index(idx, mat, axes=axes),
        sequences=T.arange(roll_amounts),
        non_sequences=[mat]
    )[0]


def autocorr_updated(autocorr):
    (n_elec, n_lags, _, _) = autocorr.shape
    return remove_indices(autocorr, axes=(0, 2)).reshape((
        n_elec,
        (n_elec - 1) * n_lags,
        (n_elec - 1) * n_lags
    ))


def xcorr_updated(xcorr):
    """Keep only xcorr between non-self electrodes and self"""
    (n_elec, n_lags, _) = xcorr.shape
    return remove_indices(xcorr, axes=(0,)).reshape((
        n_elec,
        (n_elec - 1) * n_lags,
        n_elec
    ))[:, T.arange(n_elec)]
    

def compute_weights(autocorr, xcorr):
    return theano.scan(
        fn=lambda a, x: T.nlinalg.pinv(a).dot(x),
        sequences=[autocorr, xcorr]
    )[0]


def compute_input_by_electrode(input_data):
    """
    """
    (n_samples, n_elec, n_lags) = input_data.shape
    return remove_indices(input_data, axes=(1,)).reshape(
        (n_elec, n_samples, (n_elec - 1) * n_lags)
    )


def compute_predictor(data_lagged, weights):
    return theano.scan(
        fn=lambda x, weights: x.dot(weights),
        sequences=[
            compute_input_by_electrode(data_lagged),
            weights
        ]
    )[0]


def generate_blocks(data, block_width):
    """Chunk into blocks along axis 0"""
    return theano.scan(
        lambda block_start, data: data[start:start + block_width],
        sequences=T.arange(0, datums.shape[0] - block_width, block_width),
        non_sequences=[data]
    )[0]


def predict(data, n_lags):
    XX = timelagged_autocorr(data, n_lags)
    XY = timelagged_xcorr(data, n_lags)
    XX_by_elec = autocorr_updated(XX)    
    XY_by_elec = xcorr_updated(XY)

    input_data = compute_input_by_electrode(data)
    weights = compute_weights(XX_by_elec, XY_by_elec)

    return compute_predictor(input_data, weights)


def filter_data_by_blocks(full_input_data, n_lags, block_width):
    blocked_input = generate_blocks(full_input_data, block_width)

    in_data = theano.scan(
        lambda block: block[target_slice(n_lags)],
        sequences=blocked_input
    )[0]

    def process_block(data, n_lags):
        return compute_lags(data, n_lags)[valid_slice(n_lags)]

    predicted_data = theano.scan(
        lambda block: predict(process_block(block), n_lags)
        sequences=blocked_input
    )[0]

    return in_data, predicted_data



'''
Now we continue...

>>> data = T.fmatrix()
>>> n_lags = T.iscalar()
>>> data_lagged = compute_lags(data, n_lags)
>>> data_lagged = data_lagged[valid_slice(n_lags)]
>>> target = data[target_slice(n_lags)]

>>> prediction = predict(data_lagged, n_lags)
>>> residual = target - prediction

Or, over blocks...
>>> data = T.fmatrix()
>>> n_lags = T.iscalar()
>>> predictions, actual = filter_data_by_blocks(data, n_lags)
'''
