from configparser import ConfigParser
import os


config = ConfigParser(os.environ)

PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
if not PROJECT_ROOT:
    raise Exception("PROJECT_ROOT env variable not set")

config.read(os.path.join(PROJECT_ROOT, "config.ini"))
