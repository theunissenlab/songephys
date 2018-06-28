from __future__ import print_function

import os

import luigi

from gdrive_access import download
from utils import session_basename


class DownloadSession(luigi.Task):

    session_dir = luigi.Parameter()
    pyoperants = luigi.ListParameter(default=[])

    def output(self):
        self.session_name = session_basename(self.session_dir)
        output_directory = os.path.join(self.session_dir, "raw")

        return luigi.LocalTarget(output_directory)

    def run(self):
        if download.drive is None:
            raise Exception("Not authenticated with Google Drive; "
                            "cannot download raw data files")

        download.download_session(
            self.session_name,
            save_to=self.session_dir,
            pyoperant_files=self.pyoperants)
