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


'''
class ValidateInputData(luigi.Task):

    session_dir = luigi.Parameter()

    def output(self):
        input_directory = os.path.join(self.session_dir, "raw")

        if not os.path.exists(input_directory):
            raise Exception("Input directory {} not found".format(input_directory))

        rhd_files = glob.glob(os.path.join(input_directory, "*.rhd"))

        if len(rhd_files) == 0:
            raise Exception("No RHD file found in {}".format(input_directory))

        if len(rhd_files) > 1:
            raise Exception("Multiple RHD files found in {}".format(input_directory))

        # TODO: look for pyoperant file

        return [
            luigi.LocalTarget(rhd_files[0])
        ]

    def run(self):
        pass
'''


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
