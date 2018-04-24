try:
    import cPickle as pickle
except ImportError:
    import pickle
import os

import luigi
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from figure_scripts.unit_summary import (
    _get_unit_waveforms,
    _get_electrode_waveforms,
    _get_channel_index,
    plot_waveforms,
    plot_isi,
    plot_1d_by_time,
    plot_unit_playback_responses,
)
from resegment import Resegment
from spike_sorting import WriteSemiAutoSorted, CollectWaveforms


'''
class UnitSummary(luigi.Task):

    session_dir = luigi.Parameter()
    electrode = luigi.parameter.NumericalParameter(
            var_type=int, min_value=0, max_value=16)
    unit = luigi.parameter.NumericalParameter(var_type=int,
            min_value=0, max_value=1000)

    def output(self):
        responses_dir = os.path.join(
            self.session_dir,
            "unit_responses",
            "summary_e{}u{}.png".format(
                self.electrode, self.unit
            )
        )

        return luigi.LocalTarget(responses_dir)

    def requires(self):
        return [
            WriteSemiAutoSorted(self.session_dir),
            Resegment(self.session_dir),
            CollectWaveforms(self.session_dir)
        ]

    def run(self):
        with open(self.input()[0].path, "rb") as input_file:
            block = pickle.load(input_file)
        block.create_relationship(force=True, append=True, recursive=True)

        with open(self.input()[1].path, "rb") as input_file:
            segmented_block = pickle.load(input_file)
        segmented_block.create_relationship(force=True, append=True, recursive=True)

        unit_data = _get_unit_waveforms(block, self.electrode, self.unit)
        elec_data = np.load(self.input()[2][self.electrode])[()]

        fig = plt.figure(figsize=(15, 20))

        ax0 = fig.add_axes([0, 0, 0.25, 0.2])
        plot_waveforms(unit_data, ax0)
        ax0.set_ylim(-250, 200)

        ax1 = fig.add_axes([0.25, 0, 0.25, 0.2])
        plot_waveforms(elec_data, ax1, color="Gray", mean_color="Gray")
        ax1.set_ylim(*ax0.get_ylim())
        ax1.set_yticks([])

        ax2 = fig.add_axes([0.575, 0, 0.35, 0.2])
        plot_isi(unit_data, ax2)
        ax2.set_xlabel("t (seconds)")

        ax3 = fig.add_axes([0.95, 0, 0.1, 0.2])
        ax3.set_yticks([])
        ax3.set_xticks([])

        ax4 = fig.add_axes([0, -0.45, 1.05, 0.4])
        plot_1d_by_time(unit_data, elec_data, ax4, alpha=1.0)
        # ax4.set_ylim(-6, 6)

        ax4 = fig.add_axes([0, -0.85, 1.05, 0.4])
        plot_1d_by_time(unit_data, elec_data, ax4, plot_unit_data=False, alpha=1.0)
        # ax4.set_ylim(-6, 6)

        ax4.vlines([seg.t_start for seg in segmented_block.segments if seg.annotations["sound_type"] == "playback"],
                   -6, -5.9, color="Black")
        ax4.vlines([seg.t_start for seg in segmented_block.segments if seg.annotations["sound_type"] == "live"],
                   -6, -5.9, color="Green")

        plot_unit_playback_responses(segmented_block, self.electrode, self.unit, fig, [1.1, 0, 1.05, 1.0], width=15)

        ax7 = fig.add_axes([0, 0.2, 1, 0.1])
        ax7.axis("off")
        ax7.text(0, 0.2, "{} ; Electrode {} ; Unit {}".format(session, elec_idx, unit_idx), fontsize=24)

        fig.savefig(self.output().path)
'''


class RunAllUnitSummaries(luigi.Task):

    session_dir = luigi.Parameter()

    def output(self):
        responses_dir = os.path.join(self.session_dir, "unit_summaries")

        return luigi.LocalTarget(responses_dir)

    def requires(self):
        return [
            WriteSemiAutoSorted(self.session_dir),
            Resegment(self.session_dir),
            CollectWaveforms(self.session_dir)
        ]

    def run(self):
        with open(self.input()[0].path, "rb") as input_file:
            block = pickle.load(input_file)
        block.create_relationship(force=True, append=True, recursive=True)

        with open(self.input()[1].path, "rb") as input_file:
            segmented_block = pickle.load(input_file)
        segmented_block.create_relationship(force=True, append=True, recursive=True)

        elec_data_by_idx = [np.load(self.input()[2][i].path)[()] for i in range(16)]

        if not os.path.exists(self.output().path):
            os.makedirs(self.output().path)

        for elec_idx in range(16):
            try:
                elec = _get_channel_index(segmented_block, elec_idx)
            except:
                continue
            for unit_idx in range(len(elec.units)):
                unit_data = _get_unit_waveforms(block, elec_idx, unit_idx)
                elec_data = elec_data_by_idx[elec_idx]

                fig = plt.figure(figsize=(15, 20))

                ax0 = fig.add_axes([0, 0.9, 0.25, 0.2])
                plot_waveforms(unit_data, ax0)
                ax0.set_ylim(-250, 200)

                ax1 = fig.add_axes([0.25, 0.9, 0.25, 0.2])
                plot_waveforms(elec_data, ax1, color="Gray", mean_color="Gray")
                ax1.set_ylim(*ax0.get_ylim())
                ax1.set_yticks([])

                ax2 = fig.add_axes([0.575, 0.9, 0.35, 0.2])
                plot_isi(unit_data, ax2)
                ax2.set_xlabel("t (seconds)")

                ax3 = fig.add_axes([0.95, 0.9, 0.1, 0.2])
                ax3.set_yticks([])
                ax3.set_xticks([])

                ax4 = fig.add_axes([0, 0.9 -0.45, 1.05, 0.4])
                plot_1d_by_time(unit_data, elec_data, ax4, alpha=0.1)
                # ax4.set_ylim(-6, 6)

                ax4 = fig.add_axes([0, 0.9 -0.85, 1.05, 0.4])
                plot_1d_by_time(unit_data, elec_data, ax4, plot_unit_data=False, alpha=0.1)
                # ax4.set_ylim(-6, 6)

                ax4.vlines([seg.t_start for seg in segmented_block.segments if seg.annotations["sound_type"] == "playback"],
                           -6, -5.9, color="Black")
                ax4.vlines([seg.t_start for seg in segmented_block.segments if seg.annotations["sound_type"] == "live"],
                           -6, -5.9, color="Green")

                try:
                    plot_unit_playback_responses(segmented_block, elec_idx, unit_idx, fig, [1.1, 0, 1.05, 1.0], width=15)
                except:
                    pass

                ax7 = fig.add_axes([0, 1.1, 1, 0.1])
                ax7.axis("off")
                ax7.text(0, 0.2, "{} ; Electrode {} ; Unit {}".format(
                    os.path.basename(self.session_dir), elec_idx, unit_idx),
                    fontsize=24)

                path = os.path.join(self.output().path, "e{}u{}.png".format(elec_idx, unit_idx))
                fig.savefig(path, bbox_inches="tight")
