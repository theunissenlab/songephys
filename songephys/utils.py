import logging
import time
import datetime
from functools import wraps

import numpy as np


logger = logging.getLogger(__name__)


class DeprecatedFunction(Exception):
    pass

def get_session_starttime(session_name):
    dt_str = session_name[-13:]
    return datetime.datetime.strptime(dt_str,'%y%m%d_%H%M%S')

def deprecated(msg, force=True):
    """Mark a function as deprecated. Specify what new function to use in the msg.
    """
    def _wrapper(func):
        @wraps(func)
        def _raiser(*args, **kwargs):
            if force:
                raise DeprecatedFunction(msg)
            else:
                logger.warn("{}: {}".format(DeprecatedFunction, msg))
                return func(*args, **kwargs)
        _raiser.__name__ = func.__name__
        return _raiser
    return _wrapper


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
    elif interval2[0] <= interval1[1] <= interval2[1]:
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


def pulse_edges(trace, invert=None):
    """Rising and falling edges of the pulses in a binary (digital) trace.

    Indices follow np.diff convention: an edge between samples i and i+1 is
    reported as i. A pulse already high at the start of the trace (falling edge
    before the first rising edge) is dropped; a pulse still high at the end is
    closed at len(trace).

    Params
    ======
    trace (array): binary trace (bool or 0/1 ints)
    invert (bool or None): treat the line as active-low. If None, a trace that is
        high more than half the time is assumed to be active-low.

    Returns
    =======
    rising, falling (np.ndarray): matched edge indices
    """
    x = np.asarray(trace).astype(bool)
    if invert is None:
        invert = x.mean() > .5
    if invert:
        x = ~x
    d = np.diff(x.astype(np.int8))
    rising = np.flatnonzero(d == 1)
    falling = np.flatnonzero(d == -1)
    if len(falling) and (len(rising) == 0 or falling[0] < rising[0]):
        falling = falling[1:]
    if len(rising) > len(falling):
        falling = np.append(falling, len(x))
    return rising, falling
