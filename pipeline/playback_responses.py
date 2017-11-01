import cPickle as pickle
import os

import luigi
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from figure_scripts import unit_summary
from segment_analysis import MultiSegment

from spike_sorting import WriteSorted


class PlotUnitWaveforms(luigi.Task):

    session_dir = luigi.Parameter()

    def output(self):
        responses_dir = os.path.join(self.session_dir, "unit_responses")
        if not os.path.exists(responses_dir):
            os.makedirs(responses_dir)

        output_file = "unit-waveforms.png"
        return luigi.LocalTarget(os.path.join(responses_dir, output_file))

    def requires(self):
        return WriteSorted(self.session_dir)

    def run(self):
        with open(self.input().path, "rb") as input_file:
            block = pickle.load(input_file)
        block.create_relationship(force=True, append=True, recursive=True)

        multiseg = MultiSegment(np.random.choice(block.segments, replace=False, size=20))

        fig, axes = unit_summary.plot_all_waveforms(multiseg)

        # TODO: dont hardcode the y axis lims?
        for ax in axes.reshape(-1):
            ax.set_ylim(-300, 200)
        plt.savefig(self.output().path)


class PlotPlaybackResponses(luigi.Task):

    session_dir = luigi.Parameter()
    gaussian_width = luigi.IntParameter()

    def output(self):
        responses_dir = os.path.join(self.session_dir, "unit_responses", "playbacks-{}".format(self.gaussian_width))
        return luigi.LocalTarget(responses_dir)

    def requires(self):
        return WriteSorted(self.session_dir)

    def run(self):
        with open(self.input().path, "rb") as input_file:
            block = pickle.load(input_file)
        block.create_relationship(force=True, append=True, recursive=True)

        figures = unit_summary.plot_playback_responses(block, width=self.gaussian_width)

        if not os.path.exists(self.output().path):
            os.makedirs(self.output().path)
        for unit_idx, fig in enumerate(figures):
            path = os.path.join(self.output().path, "{}.png".format(unit_idx))
            fig.savefig(path)
