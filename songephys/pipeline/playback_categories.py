import os

import luigi

import matplotlib.pyplot as plt
import numpy as np
import suss.io

from ..analysis.playbacks.categories import generate_category_comparisons
from .preprocess_ephys import ManualSorting
from .preprocess_sound import MergeVocalPeriods
from .file_setup import SiteProjectConfig
class GenerateCategoryPSTHs(luigi.Task):

    site_dir = luigi.Parameter()
    electrode = luigi.parameter.NumericalParameter(
        var_type=int
    )

    def requires(self):
        # requires spike sorting to be done and 
        return {
            "vocal_periods": MergeVocalPeriods(self.site_dir),
            "spikes": ManualSorting(self.site_dir, self.electrode),
            "project_config": SiteProjectConfig(self.site_dir)
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

        # get project config to make sure electrode lies in our given range
        project_cfg = self.input()['project_config'].data()
        if self.electrode > project_cfg['EPHYS']['ELECTRODE_START']+project_cfg['EPHYS']['ELECTRODE_N'] or \
            self.electrode < project_cfg['EPHYS']['ELECTRODE_START']:
            raise ValueError("Electrode %d falls outside of the range specified in project config")

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
        # get project config
        # TODO this is  a bit janky, however we need the config to get the electrode ranges
        project_cfg = SiteProjectConfig(self.site_dir).output().data()
        return [
            GenerateCategoryPSTHs(self.site_dir, elec_idx)
            for elec_idx in range(project_cfg['EPHYS']['ELECTRODE_START'],project_cfg['EPHYS']['ELECTRODE_START']+project_cfg['EPHYS']['ELECTRODE_N'])
        ]
        
    def output(self):
        pass

    def run(self):
        pass
