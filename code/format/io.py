import pickle
import yaml


def save_block(filename, block):
    with open(filename, "wb+") as datafile:
        pickle.dump(block, datafile)


def load_block(filename):
    """Load a neo block from a picklefile whilst generating proper object links"""
    with open(filename, "rb") as datafile:
        block = pickle.load(datafile)
    block.create_relationship(force=True, append=True, recursive=True)
    return block


def load_yaml(filename):
    with open(filename, "r") as datafile:
        data = yaml.load(datafile.read(), Loader = yaml.FullLoader)
    return data


def save_yaml(filename, data):
    with open(filename, "w") as datafile:
        yaml.dump(data, datafile, default_flow_style=False)
