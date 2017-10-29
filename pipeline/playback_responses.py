import cPickle as pickle

import luigi
import numpy as np

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
        with open(self.input()[0].path, "rb") as input_file:
            block = pickle.load(input_file)
        block.create_relationship(force=True, append=True, recursive=True)

        multiseg = MultiSegment(np.random.choice(block.segments, replace=False, size=20))

        unit_summary.plot_all_waveforms(multiseg.waveforms())
        plt.savefig(self.output().path)


class PlotPlaybackResponses(luigi.Task):

    session_dir = luigi.Parameter()

    def output(self):
        responses_dir = os.path.join(self.session_dir, "unit_responses", "playbacks")
        if not os.path.exists(responses_dir):
            os.makedirs(responses_dir)

        return luigi.LocalTarget(responses_dir)

    def requires(self):
        return WriteSorted(self.session_dir)

    def run(self):
        with open(self.input()[0].path, "rb") as input_file:
            block = pickle.load(input_file)
        block.create_relationship(force=True, append=True, recursive=True)

        figures = unit_sumamry.plot_playback_responses(block)
        for unit_idx, fig in enumerate(figures):
            path = os.path.join(self.output().path, "{}.png".format(unit_idx))
            fig.savefig(path)
