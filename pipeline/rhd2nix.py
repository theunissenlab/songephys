"""Convert RHD file to Nix file

INPUT
_.rhd

OUTPUT
_.nix
"""
import glob
import os

import luigi
from neo.io import RHDIO, NixIO

from utils import session_basename
from setup import DownloadSession


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
        # TODO (kevin): When I fix the validate step above
        input_directory = self.input().path
        rhd_file = glob.glob(os.path.join(input_directory, "*.rhd"))[0]
        rhd = RHDIO(rhd_file)
        nix = NixIO(self.output().path)

        # check for if pyoperant file is needed, warn otherwise

        nix.write_block(rhd.read_block())
        nix.close()
