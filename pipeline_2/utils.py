import os


def session_basename(filename):
    if filename.endswith("/"):
        filename = filename[:-1]
    return os.path.splitext(os.path.basename(filename))[0]
