import cPickle as pickle
import os

import numpy as np
import scipy.stats
import luigi
import matplotlib.pyplot as plt

from filter_signals import FilterElectrodeSignals
from preproephys import fix_chIndex, extract_spikes

from sklearn.manifold import TSNE
from spikes.write_neo import write_neo

from spikes.sorting.preprocess import collect_waveforms, sort_spikes_on_unit
from spikes.sorting.merging import isosplit_cluster
# from spikes_vis.runsort import create_server
from spikes.sorting.autocluster.process_block import time_slices, process_votes, relabel_across_time, gen_figures
from spikes.sorting.autocluster import sort
from spikes.sorting.metrics import cluster_snr, interspike_interval, firing_rate, consistency, isolation
import spikes.sorting.cluster_linking as clust

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

            waveforms, times = collect_waveforms(
                    unit.spiketrains,
                    force_single_peak=False
            )
            waveforms = waveforms[:, config.WF_SLICE]

            # remove extremely large peaks
            # FIXME: remove this thing?
            # waveforms = waveforms[np.max(np.abs(waveforms), axis=1) < 800]
            # waveforms = waveforms[np.max(np.abs(waveforms), axis=1) < 800]
            np.save(output.path, {"waveforms": waveforms, "times": times})


class ComputeSpikeTSNE(luigi.Task):

    session_dir = luigi.Parameter()
    electrode = luigi.parameter.NumericalParameter(
        var_type=int, min_value=0, max_value=16)

    def output(self):
        sorting_dir = os.path.join(self.session_dir, "spikes")

        return luigi.LocalTarget(os.path.join(
            sorting_dir,
            "tsne-e{}.npy".format(self.electrode)
        ))
            
    def requires(self):
        return CollectWaveforms(self.session_dir)

    def run(self):
        waveforms = np.load(self.input()[self.electrode].path)[()]["waveforms"]
        tsne = TSNE(n_components=2)
        xy = tsne.fit_transform(waveforms[:10000])

        np.save(self.output().path, xy)


class ClusterOverTime(luigi.Task):

    session_dir = luigi.Parameter()
    step_size = luigi.IntParameter()
    iters = luigi.IntParameter(default=3)

    def output(self):
        clusters_dir = os.path.join(self.session_dir, "cluster_linking")
        for elec_id in range(16):
            yield luigi.LocalTarget(os.path.join(
                clusters_dir,
                "proposed_clusters-e{}.npy".format(elec_id)
            ))
            
    def requires(self):
        return CollectWaveforms(self.session_dir)

    def run(self):
        for elec_id, (input_file, output_file) in enumerate(zip(self.input(), self.output())):
            if not os.path.exists(os.path.dirname(output_file.path)):
                os.makedirs(os.path.dirname(output_file.path))
            data = np.load(input_file.path)[()]
            print "sorting {}".format(input_file.path)

            time_ranges, data_blocks = time_slices(data)

            # Run sorting a few times on each time range
            labels_blocks = []
            for time_range, data_block in zip(time_ranges, data_blocks):
                cluster_results = []
                for _ in range(3):  # how many times to run the clustering
                    if not len(data_block["waveforms"]):
                        cluster_results.append([])
                        continue
                    labels, data = sort.sort(data_block["waveforms"][:, 10:32],
                            initial_cluster_type="kmeans",
                            initial_cluster_count=30,
                            ordered=False,
                            merge_mode="linear",
                            reassign_mode="kmeans",
                            merge_initial_threshold=0.8,
                            merge_final_threshold=0.2,
                            max_peak_size=300,
                            min_peak_size=30,
                            peak_idx=10,
                            reassign_dims=1)
                    cluster_results.append(labels)
                labels_blocks.append(cluster_results)
                print "Just sorted one block"

            # Compute the mode of each block and try to find consistent labeling across time
            best_labels = []
            for labels_block in labels_blocks:
                best_labels.append(scipy.stats.mode(process_votes(labels_block)).mode.flatten())
            best_labels = relabel_across_time(time_ranges, data_blocks, best_labels, threshold=0.6)

            metric_blocks = []
            for _, data_block, labels, labels_block in zip(
                    time_ranges, data_blocks, best_labels, labels_blocks):
                metric = {}

                if not len(labels):
                    metric_blocks.append({})
                    continue
                
                if elec_id in [6, 13]:
                    isolation_metric = isolation(data_block["waveforms"], labels, k=6)
                else:
                    isolation_metric = {}

                for l in np.unique(labels):
                    metric[l] = {
                        "snr": cluster_snr(data_block["waveforms"][labels == l]),
                        "isi": interspike_interval(data_block["times"][labels == l], range=(0, 0.02), bins=50),
                        "fr": firing_rate(data_block["times"][labels == l]),
                        "consistency": consistency(np.array(labels_block)[:, labels == l]),
                        "isolation": isolation_metric.get(l, 0.0)
                    }

                metric_blocks.append(metric)

            result = {
                "time_ranges": time_ranges,
                "data": data_blocks,
                "labels": best_labels,
                "metrics": metric_blocks
            }

            np.save(output_file.path, result)


class GenerateClusterLinkingFigures(luigi.Task):

    session_dir = luigi.Parameter()

    def output(self):
        clusters_dir = os.path.join(self.session_dir, "cluster_linking")

        for elec_id in range(16):
            yield luigi.LocalTarget(os.path.join(
                clusters_dir,
                "images_{}".format(elec_id),
                "cluster_info.npy"
            ))
            
    def requires(self):
        return ClusterOverTime(self.session_dir, step_size=600)

    def run(self):
        for input_file, output_file in zip(self.input(), self.output()):
            output_dir = os.path.dirname(output_file.path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            print "Generating figures from {} to {}".format(input_file.path, output_dir)
            data = np.load(input_file.path)[()]

            link_dirname = "images"  # this is for the 

            figure_locations = []
            for time_range, data_block, labels, metrics in zip(
                    data["time_ranges"], data["data"], data["labels"], data["metrics"]):
                time_dir = "{}_{}".format(*time_range)
                if not os.path.exists(os.path.join(output_dir, time_dir)):
                    os.makedirs(os.path.join(output_dir, time_dir))

                fig_dict = {}
                if not len(labels):
                    figure_locations.append(fig_dict)
                    continue

                figs = gen_figures(time_range, data_block, labels, metrics)

                base_scatter = os.path.join(time_dir, "scatter.png")
                fig_dict["scatter"] = os.path.join(link_dirname, base_scatter)
                figs[0].savefig(os.path.join(output_dir, base_scatter), format="png")
                plt.close(figs[0])

                for label, fig in zip(sorted(np.unique(labels)), figs[1:]):
                    base_waveforms = os.path.join(time_dir, "waveforms_{}.png".format(label))
                    fig_dict[label] = os.path.join(link_dirname, base_waveforms)
                    fig.savefig(os.path.join(output_dir, base_waveforms), format="png")
                    plt.close(fig)

                figure_locations.append(fig_dict)
            data.update({
                "figures": figure_locations
            })

            np.save(output_file.path, data)


class RunManualClusterLinking(luigi.Task):

    session_dir = luigi.Parameter()
    electrode = luigi.parameter.NumericalParameter(
        var_type=int, min_value=0, max_value=16)

    def output(self):
        clusters_dir = os.path.join(self.session_dir, "cluster_linking")
        return luigi.LocalTarget(os.path.join(
            clusters_dir,
            "linked_clusters-e{}.npy".format(self.electrode)
        ))
            
    def requires(self):
        return GenerateClusterLinkingFigures(self.session_dir)

    def run(self):
        app = clust.create_server(list(self.input())[self.electrode].path, self.output().path)
        port = int("1{}{:02d}".format(self.session_dir.split("_")[-1][:2], self.electrode))
        webbrowser.open_new("http://0.0.0.0:{}/".format(port))
        app.run(host="0.0.0.0", port=port, debug=False)


class WriteSemiAutoSorted(luigi.Task):

    session_dir = luigi.Parameter()

    def output(self):
        sorting_dir = os.path.join(self.session_dir, "spikes")
        output_file = "{}-sorted.pkl".format(session_basename(self.session_dir))
        return luigi.LocalTarget(os.path.join(sorting_dir, output_file))

    def requires(self):
        yield ExtractSpikes(self.session_dir)
        for elec_idx in range(16):
            yield RunManualClusterLinking(self.session_dir, elec_idx)

    def run(self):
        with open(self.input()[0].path, "rb") as input_file:
            block = pickle.load(input_file)
        block.create_relationship(force=True, append=True, recursive=True)

        new_spike_times = []
        new_waveforms = []
        for elec_id, clustered in enumerate(self.input()[1:]):
            sorted_units_on_elec = np.load(clustered.path)[()]

            new_spike_times.append([])
            new_waveforms.append([])

            unit = (
                channel_index.units[0]
                for channel_index in block.channel_indexes
                if channel_index.name == "Amp Elec {}".format(elec_id)
            ).next()

            # need to iterate over spiketrains (i.e. segments) and identify the sorted 
            # units on each one
            for sorted_unit in sorted_units_on_elec:
                waveforms = sorted_unit["waveforms"]
                times = sorted_unit["times"]

                new_wfs = []
                new_times = []

                for seg_idx, st in enumerate(unit.spiketrains):
                    is_within_segment = (times >= st.t_start) & (times < st.t_stop)
                    new_wfs.append(waveforms[is_within_segment])
                    new_times.append(times[is_within_segment])

                new_spike_times[elec_id].append(np.array(new_times))
                new_waveforms[elec_id].append(np.array(new_wfs))

        for elec_id, spike_times in enumerate(new_spike_times):
            new_spike_times[elec_id] = zip(*spike_times)

        for elec_id, waveforms in enumerate(new_waveforms):
            new_waveforms[elec_id] = zip(*waveforms)
    
        for elec_id in range(16):
            print("shapes")
            print(len(new_spike_times[elec_id]))
            print(new_spike_times[elec_id])


        new_block = write_neo(block, new_spike_times, new_waveforms)
        new_block.create_relationship(force=True, append=True, recursive=True)

        with open(self.output().path, "wb") as output_file:
            pickle.dump(new_block, output_file)


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
        var_type=int, min_value=0, max_value=16)

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


class RunAutomatedClustering(luigi.Task):

    session_dir = luigi.Parameter()

    def output(self):
        sorting_dir = os.path.join(self.session_dir, "spikes")
        for elec_id in range(16):
            yield luigi.LocalTarget(os.path.join(
                sorting_dir,
                "autoclustered-e{}.npy".format(elec_id)
            ))
            
    def requires(self):
        return CollectWaveforms(self.session_dir)

    def run(self):
        for input_file, output_file in zip(self.input(), self.output()):
            waveforms = np.load(input_file.path)[()]["waveforms"]
            labels = isosplit_cluster(waveforms, threshold=1.0, k_initial=30)
            np.save(output_file.path, labels)


class TestAutomatedClustering(luigi.Task):
    """Try various settings of the autoclustering parameters to see which look the best...

    Things to try:
    feature space (regular waveform)
    initial cluster count (25)
    initial cluster method (kmeans vs gmm)
    merge thresholds (0.6 to 0.1) (0.8 to 0.1) (0.6 to 0.2) (0.8 to 0.2)
    reassignment mode (kmeans vs gmm)
    """

    session_dir = luigi.Parameter()

    def output(self):
        sorting_dir = os.path.join(self.session_dir, "spikes")
        for elec_id in range(16):
            yield luigi.LocalTarget(os.path.join(
                sorting_dir,
                "autoclustered-e{}.npy".format(elec_id)
            ))
            
    def requires(self):
        return CollectWaveforms(self.session_dir)

    def run(self):
        for input_file, output_file in zip(self.input(), self.output()):
            waveforms = np.load(input_file.path)[()]["waveforms"]
            labels = isosplit_cluster(waveforms, threshold=1.0, k_initial=30)
            np.save(output_file.path, labels)


class WriteAutoSorted(luigi.Task):

    session_dir = luigi.Parameter()

    def output(self):
        sorting_dir = os.path.join(self.session_dir, "spikes")
        output_file = "{}-sorted-auto.pkl".format(session_basename(self.session_dir))
        return luigi.LocalTarget(os.path.join(sorting_dir, output_file))

    def requires(self):
        yield ExtractSpikes(self.session_dir)
        yield CollectWaveforms(self.session_dir)
        yield RunAutomatedClustering(self.session_dir)

    def run(self):
        with open(self.input()[0].path, "rb") as input_file:
            block = pickle.load(input_file)
        block.create_relationship(force=True, append=True, recursive=True)

        new_spike_times = []
        new_waveforms = []
        for elec_id, (wf_data, labels) in enumerate(zip(self.input()[1], self.input()[2])):
            new_spike_times.append([])
            new_waveforms.append([])

            unit = (
                channel_index.units[0]
                for channel_index in block.channel_indexes
                if channel_index.name == "Amp Elec {}".format(elec_id)
            ).next()

            wf_data = np.load(wf_data.path)[()]
            waveforms = wf_data["waveforms"]
            times = wf_data["times"]
            labels = np.load(labels.path)[()]

            sorted_units = np.unique(labels)

            # need to iterate over spiketrains (i.e. segments) and identify the sorted 
            # units on each one
            for seg_idx, st in enumerate(unit.spiketrains):
                new_wfs = []
                new_times = []

                is_within_segment = (times >= st.t_start) & (times < st.t_stop)
                for unit_label in sorted_units:
                    is_unit_label = labels == unit_label

                    new_wfs.append(waveforms[is_within_segment & is_unit_label])
                    new_times.append(times[is_within_segment & is_unit_label])

                new_spike_times[elec_id].append(new_times)
                new_waveforms[elec_id].append(new_wfs)

        new_block = write_neo(block, new_spike_times, new_waveforms)
        new_block.create_relationship(force=True, append=True, recursive=True)

        with open(self.output().path, "wb") as output_file:
            pickle.dump(new_block, output_file)


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
        
            data = np.load(self.input()[elec_id + 1].path)[()]
            new_times, new_wfs = sort_spikes_on_unit(unit, -50, data["model"], data["units"])
            results.append((new_times, new_wfs))
        results = zip(*results)

        new_block = write_neo(block, *results)
        new_block.create_relationship(force=True, append=True, recursive=True)

        with open(self.output().path, "wb") as output_file:
            pickle.dump(new_block, output_file)

