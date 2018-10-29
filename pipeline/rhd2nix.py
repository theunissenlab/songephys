"""Convert RHD file to Nix file

INPUT
_.rhd

OUTPUT
_.nix
"""
import glob
import os
import numpy as np

import luigi
from neo.io import RHDIO, NixIO

from utils import session_basename
from setup import DownloadSession
from lazy import create_lazy_signals_from_nix


class Rhd2Nix(luigi.Task):

    session_dir = luigi.Parameter()

    def requires(self):
        return DownloadSession(self.session_dir)

    def output(self):
        segmentation_dir = os.path.join(self.session_dir, "segmentation")
        if not os.path.exists(segmentation_dir):
            os.makedirs(segmentation_dir)

        nix_filename = "{}.nix".format(session_basename(self.session_dir))
        return luigi.LocalTarget(os.path.join(segmentation_dir, nix_filename))

    def run(self):
        input_directory = self.input().path
        rhd_file = glob.glob(os.path.join(input_directory, "*.rhd"))[0]

        rhd = RHDIO(rhd_file)
        nix = NixIO(self.output().path)
        nix.write_block(rhd.read_block())
        nix.close()


class GenerateLazyFiles(luigi.Task):

    session_dir = luigi.Parameter()

    def requires(self):
        return Rhd2Nix(self.session_dir)

    def output(self):
        raw_dir = os.path.join(self.session_dir, "raw")
        self.mic_path = os.path.join(raw_dir, "mic.memmap")
        self.elec_path = os.path.join(raw_dir, "raw.memmap")
        return luigi.LocalTarget(os.path.join(self.session_dir, "lazy.npy"))

    def run(self):
        nix_file = self.input().path

        new_sigs = create_lazy_signals_from_nix(
            nix_file,
            self.mic_path,
            self.elec_path,
            batch_size=30000
        )

        np.save(self.output().path, new_sigs)
