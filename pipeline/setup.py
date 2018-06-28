from __future__ import print_function

try:
    import cPickle as pickle
except ImportError:
    import pickle
import os

import glob
import luigi
import pandas as pd

import config
from gdrive_access import download

from utils import session_basename


class DownloadSession(luigi.Task):
    
    session_dir = luigi.Parameter()
    pyoperants = luigi.ListParameter(default=[])
    notes = luigi.Parameter(default=None)

    def output(self):
        self.session_name = session_basename(self.session_dir)
        bird_name = self.session_name.split("_")[0]
        output_directory = os.path.join(self.session_dir, "raw")

        return luigi.LocalTarget(output_directory)

    def run(self):
        download.download_session(
            self.session_name,
            save_to=self.session_dir,
            pyoperant_files=self.pyoperants)
        if self.notes:
            with open(os.path.join(self.output().path, "NOTES"), "a+") as notefile:
                notefile.write(self.notes)


'''
class DownloadSessions(luigi.Task):

    file_summary = luigi.Parameter()

    def requires(self):
        with open(self.file_summary, "rb") as picklefile:
            self.sessions = pickle.load(picklefile)

        return [
            DownloadSession(session, pyoperants, notes=notes)
            for session, pyoperants, notes in self.sessions
        ]

    def run(self):
        pass
'''


