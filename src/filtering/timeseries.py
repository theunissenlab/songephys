from __future__ import division, print_function

import numpy as np
import scipy

import config


class TimeLaggedSlicing():
    """A Slice of Life"""

    @property
    def filter_width(self):
        return self.n_lags if self.causal else 2 * self.n_lags + 1

    def _valid_slice(self, data):
        """Return a tuple representing the range of data where the filter is valid.

        Considers the n_lags and causal parameters.
        """
        if self.causal:
            return slice(self.n_lags, len(data))
        else:
            return slice(self.n_lags, len(data) - self.n_lags - 1)

    def slice(self, data):
        """Slice data array according to where the filter can predict"""
        return data[self._valid_slice(data)]


class TimeSeriesRegression(object, TimeLaggedSlicing):
    """Predict vector timeseries based on another vector timeseries
    
    Because the autoregressive one in statsmodels wasn't cutting it for me...

    Here's how I envision this (e.g. predicting an electrode from others)

    >>> model = TimeSeriesRegression(n_lags=5, causal=False):
    >>> model.fit(X, Y)    # X is (n samples x k features), Y is (n samples x l features)
    >>> Y_predicted = model.predict(X)
    >>> print(model.filters)
    """

    def __init__(self, n_lags=1, causal=True, downsample=1, self_buffer=0, self_index=None):
        """Causal will make filter start at t=-1, -2
        """
        self.n_lags = n_lags
        self.causal = causal
        self.downsample = downsample
        self.self_buffer = self_buffer
        self.self_index = self_index

    def _apply_self_buffer(self, x_input, t):
        if self.self_index is None:
            return x_input
        x_input = x_input.copy()

        if self.causal:
            x = x_input[t - self.n_lags:t][::-1]
            if self.self_buffer:
                x[:self.self_buffer] = 0.0
            return x[::self.downsample]
        else:
            x_lower = x_input[t - self.n_lags:t]
            x_upper = x_input[t:t + self.n_lags + 1]

            if self.self_buffer > 0:
                x_lower[-self.self_buffer:, self.self_index] = 0.0
            x_upper[:self.self_buffer + 1, self.self_index] = 0.0  # this array includes T, which is being predicted

            x_lower = x_lower[::self.downsample]
            x_upper = x_upper[::self.downsample]

            return np.vstack([x_lower, x_upper])

    def _process_lags(self, x_input):
        valid_slice = self._valid_slice(x_input)

        lagged = np.array([
            self._apply_self_buffer(x_input, t).ravel()
            for t in range(valid_slice.start, valid_slice.stop)
        ])
        # Add a constant column
        const = np.ones((len(lagged), 1))

        return np.hstack([const, lagged])

    def fit(self, X, Y):
        x_input = self._process_lags(X)
        y_target = self.slice(Y)

        self.fitted_params = np.linalg.lstsq(x_input, y_target)[0]
        self.residual = y_target - np.dot(x_input, self.fitted_params)

    def predict(self, X):
        x_input = self._process_lags(X)
        # FIXME: this changes the shape... doesn't deal with edges
        return np.dot(x_input, self.fitted_params)

    def fit_predict(self, X, Y):
        self.fit(X, Y)
        return self.predict(X)


class PredictTimeseries(object, TimeLaggedSlicing):
    """Predict timeseries on one channel from all other channels"""

    def __init__(self, n_lags=1, causal=True, downsample=1, self_buffer=0):
        self.n_lags = n_lags
        self.causal = causal
        self.downsample = downsample
        self.self_buffer = self_buffer
        self.models = {}

    def _isolate_column(self, X, col=0):
        return X, X[:, col:col+1]  # TODO: uhh
        return np.hstack([X[:, :col], X[:, col+1:]]), X[:, col:col+1]

    def fit(self, X):
        (_, self.n_channels) = X.shape

        for channel in range(self.n_channels):
            predictors, target = self._isolate_column(X, channel)
            model = TimeSeriesRegression(n_lags=self.n_lags, causal=self.causal,
                    downsample=self.downsample, self_buffer=self.self_buffer,
                    self_index=channel)
            model.fit(predictors, target)
            self.models[channel] = model
            # print("Fit channel {}".format(channel))

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)

    def predict(self, X):
        """Return predicted response and residuals"""
        if not self.models:
            raise Exception("Have not fit model yet!")

        predictions = []
        residuals = []

        for channel in range(self.n_channels):
            model = self.models[channel]
            predictors, target = self._isolate_column(X, channel)
            predicted = model.predict(predictors)
            residual = self.slice(target) - predicted
            predictions.append(predicted)
            residuals.append(residual)

        return np.hstack(predictions), np.hstack(residuals)
