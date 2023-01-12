import pickle
import os

import luigi
import numpy as np

from .preprocess_ephys import ManualSorting
from ..analysis.units import generate_plots

from .. import config


class UnitSummaries(luigi.Task):
    site_dir = luigi.Parameter()

    def requires(self):
        return dict(
            (elec, ManualSorting(self.site_dir, elec))
            for elec in config.ELECTRODES
        )

    def output(self):
        return None

    def run(self):
        for elec in self.input().keys():
            generate_plots(self.site_dir, elec=elec)
