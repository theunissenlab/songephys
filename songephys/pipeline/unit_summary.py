import pickle
import os

import luigi
import numpy as np

from .preprocess_ephys import ManualSorting
from ..analysis.units import generate_plots

from .. import config
try:
    cfg = global_cfg
except:
    cfg = config.get_config()

class UnitSummaries(luigi.Task):
    site_dir = luigi.Parameter()

    def requires(self):
        return dict(
            (elec, ManualSorting(self.site_dir, elec))
            for elec in range(cfg['EPHYS'].get('ELECTRODE_START',0),
                              cfg['EPHYS'].get('ELECTRODE_START',0)+cfg['EPHYS'].get('ELECTRODE_N',32))
        )

    def output(self):
        return None

    def run(self):
        for elec in self.input().keys():
            generate_plots(self.site_dir, elec=elec)
