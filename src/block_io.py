try:
    import cPickle as pickle
except ImportError:
    import pickle


def save_block(filename, block):
    with open(filename, "wb+") as datafile:
        pickle.dump(block, datafile)


def load_block(filename):
    with open(filename, "rb") as datafile:
        block = pickle.load(datafile)
    block.create_relationship(force=True, append=True, recursive=True)
