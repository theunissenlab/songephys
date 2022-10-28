import numpy as np


def gaussian_convolver(std, n=3.0):
    """Create gaussian in an array form extending n std on either side

    Input:
    std (float): std of gaussian distribution
    n (float, default=3.0): number of stds to extend gaussian in each direction

    Output:
    convolver (fn): function that takes an array, and convolves it with a
        gaussian, returning the same sized array as before
    """
    extend = n * np.floor(std)
    t = np.arange(-extend, extend + 1)
    std = float(std)
    gaussian = (
        pow(np.sqrt(2 * np.pi * pow(std, 2)), -1) *
        np.exp(-pow(t, 2) / (2 * pow(std, 2)))

    )
    return lambda d: np.convolve(d, gaussian, mode="same")


def exponential_convolver(tau, n=4.0):
    """Create normalized exponential fn in array form

    Args:
    tau (float): mean of exponential distribution
    n (float, default=4.0): number of means to extend distribution out to

    convolver (fn): return a function that takes an array and convolves
        with exponential of mean tau. returns same size array as input
    """
    t = np.arange(n * tau)
    result = np.exp(-t / float(tau))
    exp = result / np.sum(result)

    # need to cut off the end since full will extend the last exponential
    # past the end

    return lambda d: np.convolve(d, exp, mode="full")[:d.size]


def conv(data, convolver, *convolver_args, **convolver_kwargs):
    """Convolve an exponential with all rows in the dataset

    Args:
    data (N_samples x N_dim):
        array of datapoint coordinates
    convolver (fn)
        Use either gaussianConvolver or expConvolver
        This function takes args and returns an array for convolution
    *convolver_args (*args)
        Arguments to pass to convolver, i.e. tau for expConvolver

    Returns:
    convolved_data (N_samples, N_dim):
        array of datapoint coordinates after convolving
    """
    return np.apply_along_axis(
            convolver(*convolver_args, **convolver_kwargs), 1, data)

