import os

import luigi

import matplotlib.pyplot as plt
import numpy as np
import suss.io

from ..analysis.playbacks.categories import generate_category_comparisons
from .preprocess_ephys import ManualSorting
from .preprocess_sound import MergeVocalPeriods

from .. import config


class GenerateCategoryPSTHs(luigi.Task):

    site_dir = luigi.Parameter()
    electrode = luigi.parameter.NumericalParameter(
        var_type=int,
        min_value=config.ELECTRODES[0],
        max_value=config.ELECTRODES[-1] + 1
    )

    def requires(self):
        # requires spike sorting tobe done and 
        return {
            "vocal_periods": MergeVocalPeriods(self.site_dir),
            "spikes": ManualSorting(self.site_dir, self.electrode)
            # and sorted spikes
        }
        
    def output(self):
        self.categories_dir = os.path.join(self.site_dir, "categories")
        if not os.path.exists(self.categories_dir):
            os.makedirs(self.categories_dir)

        return luigi.LocalTarget(os.path.join(self.categories_dir, "placeholder"))

    def run(self):
        events = np.load(self.input()["vocal_periods"].path, allow_pickle=True)[()]["playback"]
        spikes = suss.io.read_pickle(self.input()["spikes"].path)

        for unit, node in spikes.labeled_nodes:
            output_path = os.path.join(
                self.categories_dir,
                "psth_e{}u{}.png".format(
                    self.electrode, unit
                )
            )
            try:
                fig, axes = generate_category_comparisons(
                    events,
                    node.flatten().times,
                    node.flatten().waveforms
                )
                axes[-1][0].set_title("e{}u{}".format(self.electrode, unit))
                fig.savefig(output_path, format="png")
                plt.close(fig)
            except:
                print("Failed to save unit {} on elec {}".format(
                    unit, self.electrode))
                raise


class RunSiteCategories(luigi.Task):

    site_dir = luigi.Parameter()

    def requires(self):
        return [
            GenerateCategoryPSTHs(self.site_dir, elec_idx)
            for elec_idx in config.ELECTRODES
        ]
        
    def output(self):
        pass

    def run(self):
        pass
