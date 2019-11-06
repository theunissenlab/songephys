import os

import pickle

import numpy as np
import luigi

from preprocess_ephys import FilterSignals
from file_setup import GenerateLazyFiles

from format.io import load_yaml
from format.lazy import (
    LazySite,
)


class MergeLazyFiles(luigi.Task):
    """Create memmapped array on disk for lazy loading

    Inputs
    ------
    session_dir : str
        Path to session

    Outputs memmapped files for the microphone and electrode data arrays
    """

    site_dir = luigi.Parameter()

    def requires(self):
        self.session_info = load_yaml(
            os.path.join(self.site_dir, "sessions.yaml")
        )

        lazy_signals = []
        for session in self.session_info["sessions"]:
            session_dir = os.path.join(
                    self.site_dir,
                    "..",
                    "..",
                    "sessions",
                    session["id"]
            )
            lazy_signals.append(FilterSignals(session_dir=session_dir))

        return {
            "lazy_signals": lazy_signals
        }

    def output(self):
        return luigi.LocalTarget(os.path.join(self.site_dir, "lazy_data.pkl"))

    def run(self):
        lazy_signals_raw = []
        lazy_signals_mic = []
        lazy_signals_filtered = []
        signal_times = []
        for session_output, session in zip(self.input()["lazy_signals"], self.session_info["sessions"]):
            session_sigs = np.load(session_output["lazy_signals"].path, allow_pickle=True)[()]
            lazy_signals_mic.append(session_sigs["mic"])
            lazy_signals_raw.append(session_sigs["raw"])
            lazy_signals_filtered.append(session_sigs["filtered"])
            signal_times.append(session["t"])
            
        new_site = LazySite(
            signal_times,
            mic=lazy_signals_mic,
            raw=lazy_signals_raw,
            filtered=lazy_signals_filtered
        )

        print(new_site)

        with open(self.output().path, "wb") as picklefile:
            pickle.dump(new_site, picklefile)
