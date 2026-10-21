import yaml
from ..format.io import load_yaml

from luigi.target import Target
import luigi, pandas
import os
from pynwb import NWBHDF5IO


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


class NWBTarget(luigi.Target):
    def __init__(self, nwbfile, *target_names):
        self.path = nwbfile
        self.target_names = target_names
    def exists(self):
        if not os.path.exists(self.path):
            return False
        io = NWBHDF5IO(self.path, 'r')
        nwbfile = io.read()
        children = [x.name for x in nwbfile.all_children()]
        io.close()
        return all([x in children for x in self.target_names])
    
class NWBAcquisitionTarget(luigi.Target):
    def __init__(self, nwbfile, *target_names):
        self.path = nwbfile
        self.target_names = target_names
    def exists(self):
        if not os.path.exists(self.path):
            return False
        io = NWBHDF5IO(self.path, 'r')
        nwbfile = io.read()
        exists = all([x in nwbfile.acquisition for x in self.target_names])
        io.close()
        return exists
    
    def data(self):
        io = NWBHDF5IO(self.path, 'r')
        nwbfile = io.read()
        return dict({
            x: nwbfile.acquisition[x] for x in self.target_names
        })
    
    def delete(self):
        io = NWBHDF5IO(self.path, 'r+')
        nwbfile = io.read()
        for x in self.target_names:
            try:
                nwbfile.acquisition.pop(x)
            except KeyError:
                pass
        io.write(nwbfile)
        io.close()


class NWBIntervalsTarget(luigi.Target):
    def __init__(self, nwbfile, *target_names):
        self.path = nwbfile
        self.target_names = target_names
    def exists(self):
        if not os.path.exists(self.path):
            return False
        io = NWBHDF5IO(self.path, 'r')
        nwbfile = io.read()
        exists = all([x in nwbfile.intervals for x in self.target_names])
        io.close()
        return exists
    
    def data(self):
        io = NWBHDF5IO(self.path, 'r')
        nwbfile = io.read()
        return dict({
            x: nwbfile.intervals[x] for x in self.target_names
        })
    
    def delete(self):
        io = NWBHDF5IO(self.path, 'r+')
        nwbfile = io.read()
        for x in self.target_names:
            try:
                nwbfile.intervals.pop(x)
            except KeyError:
                pass
        io.write(nwbfile)
        io.close()
class NWBProcessingTarget(luigi.Target):
    def __init__(self, nwbfile, processing_module, *target_names):
        self.path = nwbfile
        self.processing_module = processing_module
        self.target_names = target_names
    
    def exists(self):
        if not os.path.exists(self.path):
            return False
        io = NWBHDF5IO(self.path, 'r')
        nwbfile = io.read()
        if self.processing_module not in nwbfile.processing:
            io.close()
            return False
        exists = all([x in nwbfile.processing[self.processing_module].data_interfaces for x in self.target_names])
        io.close()
        return exists
    
    def data(self):
        io = NWBHDF5IO(self.path, 'r')
        nwbfile = io.read()
        return dict({
            x: nwbfile.processing[self.processing_module].data_interfaces[x] for x in self.target_names
        })
    
    def delete(self):
        io = NWBHDF5IO(self.path, 'r+')
        nwbfile = io.read()
        for x in self.target_names:
            try:
                nwbfile.processing[self.processing_module].data_interfaces.pop(x)
            except KeyError:
                pass
        io.write(nwbfile)
        io.close()


class NWBUnitsTarget(luigi.Target):
    """Exists once the NWB file has a units table."""
    def __init__(self, nwbfile):
        self.path = nwbfile

    def exists(self):
        if not os.path.exists(self.path):
            return False
        with NWBHDF5IO(self.path, 'r') as io:
            return io.read().units is not None
