import time

import numpy as np
import theano


def float_x(x):
    return np.asarray(x, dtype=theano.config.floatX)


def convert_mins(t_sec):
    if t_sec == np.inf:
        return "a long ass time"
    if t_sec / 60.0 > 60:
        hours = np.floor(t_sec / (60.0 * 60.0))
        minutes = (t_sec / 60.0) - (60.0 * int(hours))
        return "{:.0f}h{:.0f}m".format(hours, minutes)
    else:
        minutes = (t_sec / 60.0)
        return "{:.1f}m".format(minutes)


def estimate_time_remaining(started_at, fraction_complete):
    time_elapsed = time.time() - started_at
    if fraction_complete:
        est_total_time = time_elapsed / fraction_complete
    else:
        est_total_time = np.inf
    return time_elapsed, est_total_time - time_elapsed

