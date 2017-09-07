import cPickle as pickle
import os

import numpy as np
import luigi

from filter_signals import FilterElectrodeSignals
from preproephys import fix_chIndex, extract_spikes
from spikes.sorting import sort, unravel
from spikes.write_neo import write_neo


class ExtractSpikes(luigi.Task):

    in_rhdfile = luigi.Parameter()

    def output(self):
        input_file = os.path.splitext(self.in_rhdfile)[0]
        return luigi.LocalTarget("{}-thresholded.pkl".format(input_file))

    def requires(self):
        return FilterElectrodeSignals(self.in_rhdfile)

    def run(self):
        with open(self.input().path, "rb") as inputfile:
            block = pickle.load(inputfile)
        block.create_relationship(force=True, append=True, recursive=True)

        fix_chIndex(block)
        extract_spikes(block)

        with open(self.output().path, "wb") as outputfile:
            pickle.dump(block, outputfile)


class SortSpikes(luigi.Task):

    in_rhdfile = luigi.Parameter()
    # electrode = luigi.parameter.NumericalParameter(
    #         var_type=int, min_value=0, max_value=15)
    # threshold = luigi.parameter.NumericalParameter(var_type=float,
    #         min_value=-200, max_value=0)

    def output(self):
        input_file = os.path.splitext(self.in_rhdfile)[0]
        return [
                luigi.LocalTarget("{}-e{}-sorted.npy".format(input_file, electrode))
                for electrode in range(16)
        ]
            
    def requires(self):
        return ExtractSpikes(self.in_rhdfile)

    def run(self):
        with open(self.input().path, "rb") as inputfile:
            block = pickle.load(inputfile)
        block.create_relationship(force=True, append=True, recursive=True)

        for i, output in enumerate(self.output()):
            result = sort(block, i)
            if not result:
                data = None
            else:
                t_arrs, X, V, W = result
                t_arrs, spike_times_list, X_list, V_list = unravel(t_arrs, X, V)

                data = {
                    "t_arrs": t_arrs,
                    "spike_times_list": np.array(spike_times_list),
                    "X": np.array(X_list),
                    "V": np.array(V_list),
                    "W": W
                }
            np.save(output.path, data)


class WriteSortedSpikesToNeo(luigi.Task):

    in_rhdfile = luigi.Parameter()

    def output(self):
        input_file = os.path.splitext(self.in_rhdfile)[0]
        return luigi.LocalTarget("{}-sorted.pkl".format(input_file))

    def requires(self):
        yield ExtractSpikes(self.in_rhdfile)
        yield SortSpikes(self.in_rhdfile)

    def run(self):
        with open(self.input()[0].path, "rb") as inputfile:
            block = pickle.load(inputfile)
        block.create_relationship(force=True, append=True, recursive=True)

        new_data = []
        for sorted_file in self.input()[1]:
            new_data.append(np.load(sorted_file.path)[()])

        new_block = write_neo(block, new_data)
        new_block.create_relationship(force=True, append=True, recursive=True)

        with open(self.output().path, "wb") as outputfile:
            pickle.dump(new_block, outputfile)

