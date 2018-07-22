import os

import luigi

from block_io import load_yaml
from resegment import ResegmentSite
from setup import DownloadSession

import config


class RunSite(luigi.Task):

    site_dir = luigi.Parameter()

    def requires(self):
        self.session_info = load_yaml(
                os.path.join(self.site_dir, "sessions.yaml")
        )

        for session in self.session_info["sessions"]:
            yield DownloadSession(
                    os.path.join(
                        self.site_dir,
                        "..",
                        "..",
                        "sessions",
                        session["id"]
                    ),
                    session.get("playbacks", [])
            )

        yield ResegmentSite(self.site_dir)

    def run(self):
        pass
