"""Literally just a template for a simple task"""
import os

import luigi


class DummyTask(luigi.Task):

    in_rhd_file = luigi.Parameter()

    def requires(self):
        # return UpstreamTask(self.in_rhdfile)

    def output(self):
        input_file = os.path.splitext(self.in_rhdfile)[0]
        # return luigi.LocalTarget("{}-something.pkl".format(input_file))

    def run(self):
        # load file from self.input().path
        # do something
        # write file to self.output().path
