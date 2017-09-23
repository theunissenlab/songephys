import os


def session_basename(filename):
    return os.path.splitext(os.path.basename(filename))[0]