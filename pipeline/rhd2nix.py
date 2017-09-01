"""Convert RHD file to Nix file

INPUT
_.rhd

OUTPUT
_.nix
"""
import os

import luigi
from neo.io import RHDIO, NixIO


class Rhd2Nix(luigi.Task):

    in_rhdfile = luigi.Parameter()

    def output(self):
        rhdfile_path = os.path.splitext(self.in_rhdfile)[0]
        return luigi.LocalTarget("{}.nix".format(rhdfile_path))

    def run(self):
        rhd = RHDIO(self.in_rhdfile)
        nix = NixIO(self.output().path)
        nix.write_block(rhd.read_block())
        nix.close()
