import time
from functools import wraps

import numpy as np
import theano


class DeprecatedFunction(Exception):
    pass


def deprecated(msg):
    """Mark a function as deprecated. Specify what new function to use in the msg.
    """
    def _wrapper(func):
        @wraps(func)
        def _raiser(*args, **kwargs):
            raise DeprecatedFunction(msg)
        _raiser.__name__ = func.__name__
        return _raiser
    return _wrapper


def float_x(x):
    return np.asarray(x, dtype=theano.config.floatX)


def intervals_overlap(interval1, interval2):
    """Determine if two time intervals overlap

    If the time intervals are (x0, x1) and (y0, y1), returns True
    if the intervals (x0, x1) and (y0, y1) have any overlap.

    Cases:
    x0....x1.....y0...y1 -> False
    x0....y0....x1....y1 -> True
    x0....y0....y1....x1 -> True
    y0....y1.....x0...x1 -> False
    y0....x0....y1....x1 -> True
    y0....x0....x1....y1 -> True

    Params
    ======
    interval1 (tuple): x0, x1 values representing first interval
    interval2 (tuple): y0, y1 values representing second interval

    Returns
    =======
    True if interval1 and interval2 overlap in any time window
    """
    if interval1[0] <= interval2[1] < interval1[1]:
        return True
    elif interval2[0] <= interval1[1] < interval2[1]:
        return True
    else:
        return False


def convert_seconds_to_minutes(t_sec):
    """Convert time in seconds to a string in minutes or hours"""
    if t_sec == np.inf:
        return "A long time"
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
