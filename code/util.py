import time
from functools import wraps

import numpy as np
import theano

from utils import deprecated


@deprecated("Use utils.float_x")
def float_x(x):
    return np.asarray(x, dtype=theano.config.floatX)
    

@deprecated("Use utils.convert_seconds_to_minutes instead")
def convert_mins(t_sec):
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

    
@deprecated("Use utils.estimate_time_remaining")
def estimate_time_remaining(started_at, fraction_complete):
    time_elapsed = time.time() - started_at
    if fraction_complete:
        est_total_time = time_elapsed / fraction_complete
    else:
        est_total_time = np.inf
    return time_elapsed, est_total_time - time_elapsed


@deprecated("Use plot_utils.border")
def border(ax, left=False, right=False, top=False, bottom=False):
    ax.spines["left"].set_visible(left)
    ax.spines["right"].set_visible(right)
    ax.spines["top"].set_visible(top)
    ax.spines["bottom"].set_visible(bottom)

    
@deprecated("Use plot_utils.get_square_dims")
def get_square_dims(n):
    root = np.sqrt(n)
    if root == np.floor(root):
        return int(root), int(root)
    else:
        a = int(np.ceil(root))
        b = (n + a) // a
        return a, b

