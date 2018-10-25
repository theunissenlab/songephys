import os
import pickle
import luigi

from block_io import load_yaml
from lazy import create_lazy_site

from resegment import ResegmentSite
from spike_sorting import SortSite
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

    def output(self):
        return luigi.LocalTarget(os.path.join(self.site_dir, "lazy_data.pkl"))

    def run(self):
        new_site = create_lazy_site(self.site_dir)
        with open(self.output().path, "wb") as picklefile:
            pickle.dump(new_site, picklefile)


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
