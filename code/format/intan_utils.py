"""Utility functions for dealing with idiosyncracies in Intan RHD file format
"""

import datetime
import os


def timestamp_from_intan_filename(filename):
    """Read datetime from filename into timestamp

    Filenames look like this:
    RedGre7107M__171221_001122

    Convert it to
    12/21/2017 at 00:11:22
    """
    if not (2000 <= datetime.datetime.now().year < 2100):
        raise FutureWarning("You are in the wrong century for this function.")

    basename = os.path.splitext(os.path.basename(filename))[0]
    date, time = basename.split("_")[-2:]
    return datetime.datetime(
        year=2000 + int(date[:2]),
        month=int(date[2:4]),
        day=int(date[4:]),
        hour=int(time[:2]),
        minute=int(time[2:4]),
        second=int(time[4:])
    )

