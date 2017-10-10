import cPickle as pickle
import os

import numpy as np
import luigi

from filter_signals import FilterElectrodeSignals
from preproephys import fix_chIndex, extract_spikes

from sklearn.manifold import TSNE
from spikes.write_neo import write_neo

from spikes.sorting.preprocess import collect_waveforms, sort_spikes_on_unit
from spikes_vis.runsort import create_server
import webbrowser

import config

from utils import session_basename


class ExtractSpikes(luigi.Task):

    session_dir = luigi.Parameter()

    def output(self):
        sorting_dir = os.path.join(self.session_dir, "spikes")
        if not os.path.exists(sorting_dir):
            os.makedirs(sorting_dir)

        output_file = "{}-thresholded.pkl".format(session_basename(self.session_dir))
        return luigi.LocalTarget(os.path.join(sorting_dir, output_file))

    def requires(self):
        return FilterElectrodeSignals(self.session_dir)

    def run(self):
        with open(self.input().path, "rb") as input_file:
            block = pickle.load(input_file)
        block.create_relationship(force=True, append=True, recursive=True)

        fix_chIndex(block)
        extract_spikes(block)

        with open(self.output().path, "wb") as output_file:
            pickle.dump(block, output_file)


class CollectWaveforms(luigi.Task):

    session_dir = luigi.Parameter()

    def output(self):
        sorting_dir = os.path.join(self.session_dir, "spikes")

        return [
            luigi.LocalTarget(
                os.path.join(sorting_dir, "spike_waveforms-e{}.npy".format(electrode))
            ) for electrode in range(16)
        ]
            
    def requires(self):
        return ExtractSpikes(self.session_dir)

    def run(self):
        with open(self.input().path, "rb") as input_file:
            block = pickle.load(input_file)
        block.create_relationship(force=True, append=True, recursive=True)

        for i, output in enumerate(self.output()):
            # a hack to get the electrode with index i
            unit = (
                channel_index.units[0]
                for channel_index in block.channel_indexes
                if channel_index.name == "Amp Elec {}".format(i)
            ).next()

            waveforms = collect_waveforms(
                    unit.spiketrains,
                    force_single_peak=False
            )[0][:, config.WF_SLICE]

            # remove extremely large peaks
            waveforms = waveforms[np.max(np.abs(waveforms), axis=1) < 300]
            np.save(output.path, waveforms)


class ComputeSpikeTSNE(luigi.Task):

    session_dir = luigi.Parameter()
    electrode = luigi.parameter.NumericalParameter(
        var_type=int, min_value=0, max_value=15)

    def output(self):
        sorting_dir = os.path.join(self.session_dir, "spikes")

        return luigi.LocalTarget(os.path.join(
            sorting_dir,
            "tsne-e{}.npy".format(self.electrode)
        ))
            
    def requires(self):
        return CollectWaveforms(self.session_dir)

    def run(self):
        waveforms = np.load(self.input()[self.electrode].path)[()]
        tsne = TSNE(n_components=2)
        xy = tsne.fit_transform(waveforms[:10000])

        np.save(self.output().path, xy)


class JustRunAllTSNEs(luigi.Task):

    session_dir = luigi.Parameter()

    def output(self):
        return None
            
    def requires(self):
        yield [ComputeSpikeTSNE(self.session_dir, i) for i in range(16)]

    def run(self):
        pass


class RunManualClustering(luigi.Task):

    session_dir = luigi.Parameter()
    electrode = luigi.parameter.NumericalParameter(
        var_type=int, min_value=0, max_value=15)

    def output(self):
        sorting_dir = os.path.join(self.session_dir, "spikes")
        return luigi.LocalTarget(os.path.join(
            sorting_dir,
            "clustered-e{}.npy".format(self.electrode)
        ))
            
    def requires(self):
        return [ComputeSpikeTSNE(self.session_dir, self.electrode), CollectWaveforms(self.session_dir)]

    def run(self):

        in_tsne, in_waveforms = self.input()

        app = create_server(in_waveforms[self.electrode].path, in_tsne.path, self.output().path)
        webbrowser.open_new("http://0.0.0.0:{}/".format(8500 + self.electrode))
        app.run(host="0.0.0.0", port=8500 + self.electrode, debug=False)


class WriteSorted(luigi.Task):

    session_dir = luigi.Parameter()

    def output(self):
        sorting_dir = os.path.join(self.session_dir, "spikes")
        output_file = "{}-sorted.pkl".format(session_basename(self.session_dir))
        return luigi.LocalTarget(os.path.join(sorting_dir, output_file))

    def requires(self):
        yield ExtractSpikes(self.session_dir)
        for i in range(16):
            yield RunManualClustering(self.session_dir, i)

    def run(self):
        with open(self.input()[0].path, "rb") as input_file:
            block = pickle.load(input_file)
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
            new_times, new_wfs = sort_spikes_on_unit(unit, -50, data["model"], data["units"])
            results.append((new_times, new_wfs))
        results = zip(*results)

        new_block = write_neo(block, *results)
        new_block.create_relationship(force=True, append=True, recursive=True)

        with open(self.output().path, "wb") as output_file:
            pickle.dump(new_block, output_file)

