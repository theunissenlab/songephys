"""Ephys Data Pipeline

``luigi -m preprocess_ephys PreprocessSite --site-dir SITE_DIR``
"""

from collections import defaultdict
import os
import time

import luigi
import numpy as np
import matplotlib.pyplot as plt
from soundsig.sound import spectrogram
from soundsig.signal import bandpass_filter
from suss.core import SpikeDataset
import suss.io

from .targets import YamlTarget
from ..format.io import load_yaml, save_yaml
from ..format.wav import AudioFile, MultiWavReader
from ..preprocess.ephys.filter import common_average_filter
from ..preprocess.ephys.detect import calculate_snippet_dimensions, extract_spikes
from ..preprocess.ephys.sort import sort, create_hierarchy

from .file_setup import Rhd2Wav, Rhd2Binary, SiteConfig, SessionConfig, SessionProjectConfig, SiteProjectConfig, DefineSessions, GenerateSiteMemmaps
from .pipeline_utils import SiteParameter, SessionParameter, get_session_sites



class FilterEphys(luigi.Task):
    """Apply filters to sections of
    """
    session_dir = SessionParameter()

    def requires(self):
        return {
            "config": SessionConfig(self.session_dir),
            "project_config": SessionProjectConfig(self.session_dir),
            "wavs": Rhd2Wav(self.session_dir)
        }

    def output(self):
        raw_dir = os.path.join(self.session_dir, "raw")
        return luigi.LocalTarget(os.path.join(raw_dir, "ephys_filtered.wav"))

    def run(self):
        session_config = self.input()["config"].data()
        project_config = self.input()["project_config"].data()
        input_file = AudioFile(self.input()["wavs"]["ephys.wav"].path)
        output_file = AudioFile.create(
            self.output().path,
            samplerate=input_file.sampling_rate,
            channels=input_file.channels)

        try:
            common_average_filter(
                input_file,
                output_file,
                batch_size=65536,
                highpass_cutoff=project_config.get("HIGHPASS_CUTOFF",300.0),
                channel_groups=[np.arange(a, b) for a, b in session_config["filtering"]["ephys.wav"]],
            )
        except:
            output_file.close()
            os.remove(output_file.path)
            raise


class DetectSpikesOnElectrode(luigi.Task):
    session_dir = SessionParameter()
    electrode_idx = luigi.IntParameter()

    def requires(self):
        return {
            "config": SessionConfig(self.session_dir),
            "project_config": SessionProjectConfig(self.session_dir),
            "ephys": FilterEphys(self.session_dir)
        }

    def output(self):
        sorting_dir = os.path.join(self.session_dir, "spikes")
        if not os.path.exists(sorting_dir):
            os.makedirs(sorting_dir)

        return luigi.LocalTarget(os.path.join(sorting_dir, "spike_waveforms-e{}.pkl".format(self.electrode_idx)))

    def run(self):
        input_file = AudioFile(self.input()["ephys"].path)
        session_config = self.input()["config"].data()
        project_config = self.input()["project_config"].data()

        if self.electrode_idx >= input_file.channels:
            raise ValueError("Cannot read electrode {} on file {} with {} channels".format(self.electrode_idx, self.input().path, input_file.channels))

        # The code used to read the variable config.SPIKE_WIDTH
        # We have updated this to (1) have longer and asymmetric spike widths, and
        # (2) the spike snippet size must be set per site
        if "spike_sorting.seconds_before_peak" not in session_config:
            raise ValueError("Config variable spike_sorting.seconds_before_peak must be set")
        if "spike_sorting.seconds_after_peak" not in session_config:
            raise ValueError("Config variable spike_sorting.seconds_after_peak must be set")

        spike_cfg = project_config.get("SPIKE_SORTING",{})
        spikes = extract_spikes(
            input_file,
            channel=self.electrode_idx,
            # Compute threshold in 5 minute windows
            batch_size=int(input_file.sampling_rate * 60 * 5),
            threshold_std=spike_cfg.get('SPIKE_THRESHOLD_STD',4),
            t_pre=session_config.get("spike_sorting.seconds_before_peak", 0.0005),
            t_post=session_config.get("spike_sorting.seconds_after_peak", 0.0015),
            min_time_between_spikes=spike_cfg.get('MIN_SPIKE_THRESHOLD_SPACING',.0003),
            z_spike_limit=spike_cfg.get("Z_SPIKE_LIMIT")
        )

        spike_dataset = SpikeDataset(
            spikes["indices"] / input_file.sampling_rate,
            spikes["waveforms"],
            sample_rate=input_file.sampling_rate,
            labels=None
        )

        suss.io.save_pickle(
            self.output().path,
            spike_dataset
        )


class DetectSpikesInSession(luigi.Task):
    session_dir = SessionParameter()

    def requires(self):
        return FilterEphys(self.session_dir)

    def output(self):
        # A bit hacky -- to know what the output is going to be, this task
        # needs to know about how many ephys channels there are. However, this
        # method might be evaluated before the first file has even been downloaded
        # or the configuration file is even first set up!
        # TODO figure out a better fix here
        site_dirs, _, _ = get_session_sites(self.session_dir)
        if not len(site_dirs):
            return DetectSpikesOnElectrode(self.session_dir, 0).output()

        config_path = os.path.join(site_dirs[0], "site_config.yaml")
        if not os.path.exists(config_path):
            return DetectSpikesOnElectrode(self.session_dir, 0).output()

        session_config = load_yaml(config_path)
        if "wavmap" not in session_config:
            return DetectSpikesOnElectrode(self.session_dir, 0).output()

        channels = 0
        for k, idx in session_config["wavmap"]["ephys.wav"]:
            if isinstance(idx, int):
                channels += idx
            else:
                channels += idx[1] - idx[0]

        for ch in range(channels):
            yield DetectSpikesOnElectrode(self.session_dir, ch).output()

    def run(self):
        input_file = AudioFile(self.input().path)
        for channel in range(input_file.channels):
            yield DetectSpikesOnElectrode(session_dir=self.session_dir, electrode_idx=channel)



class CombineSiteSpikes(luigi.Task):
    """Merge together multiple sessions via offsets

    The sessions and relative offsets must be defined in a
    sessions.yaml file in the site directory.

    Outputs one suss.core.SpikeDataset per electrode channel, each
    combining the data from all sessions defined in the site config file.
    """

    site_dir = SiteParameter()

    def requires(self):
        return {
            "site_config": DefineSessions(self.site_dir),
            "project_config": SiteProjectConfig(self.site_dir)
        }

    def output(self):
        sorting_dir = os.path.join(self.site_dir, "spikes")
        if not os.path.exists(sorting_dir):
            os.makedirs(sorting_dir)

        # TODO: this relies on the project config already being created.
        project_cfg = SiteProjectConfig(self.site_dir).output().data()

        return dict(
            (e, luigi.LocalTarget(os.path.join(
                sorting_dir,
                "spike_waveforms-e{}.pkl".format(e))))
            for e in range(project_cfg["EPHYS"]["ELECTRODE_N"])
        )

    def run(self):
        site_config = self.input()["site_config"].data()
        session_offsets = [session["t"] for session in site_config["sessions"]]
        sessions_dir = os.path.join(os.path.dirname(os.path.dirname(self.site_dir)), "sessions")

        spike_results = []
        for session in site_config["sessions"]:
            detected = yield DetectSpikesInSession(os.path.join(sessions_dir, session["id"]))
            spike_results.append(list(detected))

        n_electrodes = len(list(spike_results[0]))

        for electrode_idx in range(n_electrodes):
            spike_datasets = [
                suss.io.read_pickle(spike_result[electrode_idx].path)
                for spike_result in spike_results
            ]

            # join together the spike times and waveforms
            # add the session start time to all times in each session
            spike_times = np.concatenate([
                data.times + offset
                for data, offset in zip(spike_datasets, session_offsets)
            ])
            waveforms = np.concatenate([data.waveforms for data in spike_datasets])
            new_dataset = SpikeDataset(
                spike_times,
                waveforms,
                sample_rate=spike_datasets[0].sample_rate
            )
            suss.io.save_pickle(self.output()[electrode_idx].path, new_dataset)


class SortSpikes(luigi.Task):
    """Run spike sorting code on a single channel

    This is the automated portion of the semi-automated spike sorting,
    that generates spike clusters to be manually joined in the next step.

    This is divided across channels because it takes a while and
    might be parallelizable?

    Inputs
    ------
    site_dir : str
        Path to site
    electrode : int
        Electrode id (i.e. 8-23) (not index!)

    Outputs a suss.core.ClusterDataset with clusters of clusters of spikes
    """

    site_dir = SiteParameter()
    electrode_idx = luigi.IntParameter()
    # add upgoing /  downgoing parameter?

    def requires(self):
        return {
            "spikes": CombineSiteSpikes(self.site_dir),
            "config": SiteConfig(self.site_dir)
        }

    def output(self):
        sorting_dir = os.path.join(self.site_dir, "sort_results")
        if not os.path.exists(sorting_dir):
            os.makedirs(sorting_dir)

        return luigi.LocalTarget(
            os.path.join(sorting_dir, "sorted-e{}.pkl".format(self.electrode_idx))
        )

    def run(self):
        site_config = self.input()["config"].data()
        dataset = suss.io.read_pickle(self.input()["spikes"][self.electrode_idx].path)

        _, _, _, _, _, peak_idx = calculate_snippet_dimensions(
            t_pre=site_config.get("spike_sorting.seconds_before_peak", 0.0005),
            t_post=site_config.get("spike_sorting.seconds_after_peak", 0.0015),
            sampling_rate=dataset.sample_rate
        )

        peaks = dataset.waveforms[:, peak_idx]
        times = dataset.times[peaks < 0]
        waveforms = dataset.waveforms[peaks < 0]
        t, w, l = sort(times, waveforms)
        result = create_hierarchy(t, w, l, 1000)
        suss.io.save_pickle(self.output().path, result)


class PrepareSorting(luigi.Task):
    """Placeholder to trigger automated spike sorting for on all channels
    """

    site_dir = SiteParameter()

    def requires(self):
        # TODO: this relies on the project config already being created.
        project_cfg = SiteProjectConfig(self.site_dir).output().data()        

        for elec in range(project_cfg["EPHYS"]["ELECTRODE_N"]):
            yield ManualSorting(site_dir=self.site_dir, electrode_idx=elec)

    def run(self):
        pass

    def complete(self):
        # TODO: this relies on the project config already being created.
        project_cfg = SiteProjectConfig(self.site_dir).output().data()        
        return np.all([ManualSorting(site_dir=self.site_dir, electrode_idx=elec).complete() for elec in range(project_cfg["EPHYS"]["ELECTRODE_N"])])


class KiloSortSorting(luigi.Task):
    site_dir = SiteParameter()
    def requires(self):
        return {
            "site_config": DefineSessions(self.site_dir),
            "project_config": SiteProjectConfig(self.site_dir),
            "bin_arrays": GenerateSiteMemmaps(self.site_dir)
        }

    def output(self):
        return luigi.LocalTarget(os.path.join(self.site_dir,'kilosort_output','params.py'))


    def run(self):
        from pykilosort.io.probes import cambridge_neurotech_assy_236_h10, cambridge_neurotech_assy_236_P1
        from pykilosort import run as kilosort_run
        from pathlib import Path

        probes = dict({
            "P1": cambridge_neurotech_assy_236_P1, # 4 shank
            "H10": cambridge_neurotech_assy_236_h10 # 2 shank 
        })
        site_config = self.input()["site_config"].data()
        project_config = self.input()['project_config'].data()

        chan_names = [x[1] for x in site_config['bin_to_original_channels']['ephys.wav']]
        # offsets of each session
        session_offsets = [session["t"] for session in site_config["sessions"]]
        # paths to npy files from base folder
        session_npys = [session['raw_bin'] for session in site_config['sessions']]
        sorted_npy_paths = [Path(x) for _,x in sorted(zip(session_offsets,session_npys))]
        print(sorted_npy_paths)
        # can specify some channels are bad, should remove them from channel_names
        bad_chans = site_config.get('bad_channels',dict()).get('ephys.wav',dict()).get('RHD2000 amplifier channel',[])
        chan_names = np.array(chan_names)[~np.in1d(chan_names,bad_chans)]
        out_dir = os.path.join(self.site_dir,'kilosort_output')
        probe_fun = probes[project_config["EPHYS"].get('PROBETYPE','H10')]
        kilosort_run(sorted_npy_paths, dir_path=self.site_dir, output_dir=out_dir,
                     probe = probe_fun(chan_names = chan_names),
                     n_channels=len(chan_names),data_dtype='uint16')


class ManualSorting(luigi.Task):
    site_dir = SiteParameter()
    electrode_idx = luigi.IntParameter()

    def output(self):
        sorting_dir = os.path.join(self.site_dir, "manually_curated")
        if not os.path.exists(sorting_dir):
            os.makedirs(sorting_dir)

        return luigi.LocalTarget(os.path.join(
            sorting_dir,
            "curated-e{}.pkl".format(self.electrode_idx)
        ))

    def requires(self):
        return SortSpikes(
            site_dir=self.site_dir,
            electrode_idx=self.electrode_idx
        )

    def run(self):
        # Manual sorting here
        pass


class PreprocessSite(luigi.WrapperTask):
    """Preprocess ephys data for entire site
    """

    site_dir = SiteParameter()
    no_sort = luigi.BoolParameter()

    def requires(self):
        from .preprocess_sound import MergePlaybacksCSV
        if self.no_sort:
            return [
                DefineSessions(self.site_dir),
                MergePlaybacksCSV(self.site_dir),
                CombineSiteSpikes(self.site_dir)
            ]
        return [
            DefineSessions(self.site_dir),
            MergePlaybacksCSV(self.site_dir),
            PrepareSorting(self.site_dir)
        ]

class PreprocessOperantSite( luigi.WrapperTask ):
    """Preprocess ephys data for entire site
    This will run additional tasks <TODO INSERT TASK NAMES>
    """

    site_dir = SiteParameter()
    no_sort = luigi.BoolParameter()
    kilosort = luigi.BoolParameter()

    def requires(self):
        from .preprocess_sound import MergeOperantCSV
        if self.no_sort:
            return [
                DefineSessions(self.site_dir),
                MergeOperantCSV(self.site_dir),
                #CombineSiteSpikes(self.site_dir)
            ]
        elif self.kilosort:
            return [
                DefineSessions(self.site_dir),
                MergeOperantCSV(self.site_dir),
               # KilosortSorting(self.site_dir)
            ]
        else:
            return [
                DefineSessions(self.site_dir),
                MergeOperantCSV(self.site_dir),
                PrepareSorting(self.site_dir)
            ]

