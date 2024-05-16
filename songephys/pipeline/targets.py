import yaml
from ..format.io import load_yaml

from luigi.target import Target
import luigi, pandas


class YamlTarget(Target):
    """Target defined by one or more keys in a YAML configuration file"""

    def __init__(self, path, *keys):
        self.path = path
        self.keys = keys

    def exists(self):
        try:
            yaml_data = load_yaml(self.path)
        except:
            return False
        else:
            return all([k in yaml_data for k in self.keys])

    def data(self):
        return load_yaml(self.path)

class DataframeTarget(luigi.LocalTarget):
    """Target defined by a column key of a dataframe"""
    def __init__(self, path, *keys):
        self.path = path
        if len(keys) == 0:
            self.keys = None
        else:
            self.keys = keys

    def exists(self): # override LocalTarget.exists()
        try:
            df = pandas.read_pickle(self.path)
            if keys is None:
                return True
        except:
            return False
        else:
            return all([k in df.columns for k in self.keys])

    def remove(self):   # override FileSystemTarget.remove()
        if self.exists():
            df = pandas.read_pickle(self.path)
            df.drop(self.keys,axis=1,inplace=True)
            df.to_pickle(self.path)
    
    def open(self):
        return pandas.read_pickle(self.path)