import os

import luigi

from block_io import load_yaml
from lazy import create_lazy_site

from resegment import ResegmentSite
from filter_signals import FilterElectrodeSignals
from setup import DownloadSession

import config


class GenerateLazyFiles(luigi.Task):

    site_dir = luigi.Parameter()

    def requires(self):
        self.session_info = load_yaml(
            os.path.join(self.site_dir, "sessions.yaml")
        )
        for session in self.session_info["sessions"]:
            yield FilterElectrodeSignals(os.path.join(
                self.site_dir,
                "..",
                "..",
                "sessions",
                session["id"]))

    def run(self):
        create_lazy_site(self.site_dir)


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

        # yield ResegmentSite(self.site_dir)
        # commented out because sort sit eis now made to do both
        yield SortSite(self.site_dir)
        yield GenerateLazyFiles(self.site_dir)

    def run(self):
        pass
