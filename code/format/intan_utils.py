"""Utility functions for dealing with idiosyncracies in Intan RHD file format
"""

import datetime
import os


def timestamp_from_intan_filename(filename):
    """Read datetime from filename into timestamp
    """
    if not (2000 <= datetime.datetime.now().year < 2100):
        raise FutureWarning("You are in the wrong century for this function.")

    basename = os.path.splitext(os.path.basename(filename))[0]
    bird, date, time = basename.split("_")
    return datetime.datetime(
        year=2000 + int(date[:2]),
        month=int(date[2:4]),
        day=int(date[4:]),
        hour=int(time[:2]),
        minute=int(time[2:4]),
        second=int(time[4:])
    )
