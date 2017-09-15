import cPickle as pickle
import os

import numpy as np
import luigi

from filter_signals import FilterElectrodeSignals
from preproephys import fix_chIndex, extract_spikes

from sklearn.manifold import TSNE
from spikes.sorting import sort, unravel
from spikes.write_neo import write_neo, write_neo_new

from spikes.sorting.preprocess import collect_waveforms, sort_spikes_on_unit
from spikes_vis.runsort import create_server
import webbrowser


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


class CollectWaveforms(luigi.Task):

    in_rhdfile = luigi.Parameter()

    def output(self):
        input_file = os.path.splitext(self.in_rhdfile)[0]
        return [
                luigi.LocalTarget("{}-e{}-spikewaveforms.npy".format(input_file, electrode))
                for electrode in range(16)
        ]
            
    def requires(self):
        return ExtractSpikes(self.in_rhdfile)

    def run(self):
        with open(self.input().path, "rb") as inputfile:
            block = pickle.load(inputfile)
        block.create_relationship(force=True, append=True, recursive=True)

        for i, output in enumerate(self.output()):
            # a hack to get the electrode with index i
            unit = (
                channel_index.units[0]
                for channel_index in block.channel_indexes
                if channel_index.name == "Amp Elec {}".format(i)
            ).next()

            waveforms = collect_waveforms(unit.spiketrains, threshold=-60, force_single_peak=True)[0][:, 8:40]

            # remove extremely large peaks
            waveforms = waveforms[np.max(np.abs(waveforms), axis=1) < 300]
            np.save(output.path, waveforms)


class ComputeSpikeTSNE(luigi.Task):

    in_rhdfile = luigi.Parameter()
    electrode = luigi.parameter.NumericalParameter(
        var_type=int, min_value=0, max_value=15)

    def output(self):
        input_file = os.path.splitext(self.in_rhdfile)[0]
        return luigi.LocalTarget("{}-e{}-tsne.npy".format(input_file, self.electrode))
            
    def requires(self):
        return CollectWaveforms(self.in_rhdfile)

    def run(self):
        waveforms = np.load(self.input()[self.electrode].path)[()]
        tsne = TSNE(n_components=2)
        xy = tsne.fit_transform(waveforms[:10000])

        np.save(self.output().path, xy)


class JustRunAllTSNEs(luigi.Task):
    in_rhdfile = luigi.Parameter()

    def output(self):
        return None
            
    def requires(self):
        yield [ComputeSpikeTSNE(self.in_rhdfile, i) for i in range(16)]

    def run(self):
        pass


class RunManualClustering(luigi.Task):

    in_rhdfile = luigi.Parameter()
    electrode = luigi.parameter.NumericalParameter(
        var_type=int, min_value=0, max_value=15)

    def output(self):
        input_file = os.path.splitext(self.in_rhdfile)[0]
        return luigi.LocalTarget("{}-e{}-uhsorted.npy".format(input_file, self.electrode))
            
    def requires(self):
        return [ComputeSpikeTSNE(self.in_rhdfile, self.electrode), CollectWaveforms(self.in_rhdfile)]

    def run(self):

        in_tsne, in_waveforms = self.input()

        app = create_server(in_waveforms[self.electrode].path, in_tsne.path, self.output().path)
        webbrowser.open_new("http://0.0.0.0:{}/".format(8100 + self.electrode))
        app.run(host="0.0.0.0", port=8100 + self.electrode, debug=False)


class WriteSorted(luigi.Task):

    in_rhdfile = luigi.Parameter()

    def output(self):
        input_file = os.path.splitext(self.in_rhdfile)[0]
        return luigi.LocalTarget("{}-sorted.pkl".format(input_file))

    def requires(self):
        # yield ExtractSpikes(self.in_rhdfile)
        yield ExtractSpikes(self.in_rhdfile)
        for i in range(16):
            yield RunManualClustering(self.in_rhdfile, i)

    def run(self):
        with open(self.input()[0].path, "rb") as inputfile:
            block = pickle.load(inputfile)
        block.create_relationship(force=True, append=True, recursive=True)

        results = []
        for elec_id in range(16):
            unit = (
                channel_index.units[0]
                for channel_index in block.channel_indexes
                if channel_index.name == "Amp Elec {}".format(elec_id)
            ).next()

            print unit
        
            data = np.load(self.input()[elec_id + 1].path)[()]
            results.append(sort_spikes_on_unit(unit, -60, data["model"], data["units"]))

        new_block = write_neo_new(block, results)
        new_block.create_relationship(force=True, append=True, recursive=True)

        with open(self.output().path, "wb") as outputfile:
            pickle.dump(new_block, outputfile)



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

