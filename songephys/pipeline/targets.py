import yaml
from ..format.io import load_yaml

from luigi.target import Target


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
