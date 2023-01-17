import pickle
import os

import luigi
import numpy as np

from .preprocess_ephys import ManualSorting
from ..analysis.units import generate_plots
from .file_setup import SiteProjectConfig


class UnitSummaries(luigi.Task):
    site_dir = luigi.Parameter()
    def requires(self):
        # TODO this is  a bit janky, however we need the config to get the electrode ranges
        project_cfg = SiteProjectConfig(self.site_dir).output().data()
        return dict(
            (elec, ManualSorting(self.site_dir, elec))
            for elec in range(project_cfg['EPHYS'].get('ELECTRODE_START',0),
                              project_cfg['EPHYS'].get('ELECTRODE_START',0)+project_cfg['EPHYS'].get('ELECTRODE_N',32))
        )

    def output(self):
        return None

    def run(self):
        for elec in self.input().keys():
            generate_plots(self.site_dir, elec=elec)
