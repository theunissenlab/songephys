"""Pipeline tasks for packaging ephys data for use in a separate repo
"""

import os
import glob
import luigi
import shutil
import requests
from pathlib import Path

import suss.io
import numpy as np
import pandas as pd

from postprocess.trials import package_unit_lightweight, package_stims_and_renditions
from postprocess.memory_test import compute_coherences, lookup_reward
from preprocess_ephys import ManualSorting, MergeLazySignals


BASE_PATH = "pecking_test_processed_files"


class SetupSiteList(luigi.Task):
    """Manually create a list of all the site ids"""
    def output(self):
        return luigi.LocalTarget(os.path.join(BASE_PATH, "sitelist"))

    def run(self):
        raise Exception


class SetupStimFolder(luigi.Task):
    """Copy all stim files into a directory
    """
    def output(self):
        stim_folder = os.path.join(BASE_PATH, "stimuli")
        return luigi.LocalTarget(stim_folder)

    def requires(self):
        return SetupSiteList()

    def run(self):
        with open(self.input().path, "r") as sitelist:
            sites = sitelist.readlines()
        
        sites = [s.strip() for s in sites]

        # Filenames contain some relation information (e.g. unfamiliar, familiar)
        # So there are duplicates of the same filenames with alternating familiar/unfamiliarr
        # strings. I'll just keep both of those by going through and copying all stimuli files
        # that are found that we have not found before.

        try:
            os.makedirs(self.output().path)
            for site in sites:
                stimuli_path = os.path.join(
                    "data",
                    "birds",
                    site.split("_")[0],
                    "sessions",
                    site,
                    "stimuli",
                    "SongLadderStimsShort_fs30k"
                )
                for filename in glob.glob(os.path.join(stimuli_path, "*.wav")):
                    wavname = os.path.basename(filename)
                    destpath = os.path.join(self.output().path, wavname)
                    if not os.path.exists(destpath):
                        shutil.copy(filename, destpath)
        except:
            shutil.rmtree(self.output().path)
            raise


class DownloadPeckingTestData(luigi.Task):

    def output(self):
        # This file will never exist, so you can always run this reset task
        trial_data_file = os.path.join(BASE_PATH, "TrialData.csv")
        return luigi.LocalTarget(trial_data_file)

    def run(self):
        # The csv of all trials used in the pecking memory tests
        resp = requests.get("https://raw.githubusercontent.com/theunissenlab/zebra-finch-memory-lesion/master/TrialData.csv")
        with open(self.output().path, "wb") as trial_file:
            trial_file.write(resp.content)


class PackageUnits(luigi.Task):
    """This task also injects the reward in... maybe a better place to do it but why not here for now... and always
    """
    site_dir = luigi.Parameter()

    def output(self):
        packaged_dir = os.path.join(self.site_dir, "packaged_unit_stim_responses")
        waveform_dir = os.path.join(self.site_dir, "packaged_unit_waveforms")
        finished_marker = os.path.join(self.site_dir, "packaged_unit_stim_responses", ".luigi_task_finished")

        return {
            "packaged_folder": luigi.LocalTarget(packaged_dir),
            "waveform_folder": luigi.LocalTarget(waveform_dir),
            "task_finished": luigi.LocalTarget(finished_marker)
        }

    def requires(self):
        return {
            "sort_results": [
                ManualSorting(site_dir=self.site_dir, electrode=i)
                for i in range(32)
            ],
            "lazy": MergeLazySignals(site_dir=self.site_dir),
            "pecking_test_data": DownloadPeckingTestData(),
        }

    def run(self):
        pecking_test_data = pd.read_csv(self.input()["pecking_test_data"].path)

        packaged_dir = self.output()["packaged_folder"].path
        if not os.path.exists(packaged_dir):
            os.makedirs(packaged_dir)

        waveform_dir = self.output()["waveform_folder"].path
        if not os.path.exists(waveform_dir):
            os.makedirs(waveform_dir)

        site = os.path.basename(self.site_dir)

        for elec, sort_result in enumerate(self.input()["sort_results"]):
            sorted_file = suss.io.read_pickle(sort_result.path)
            clusters = sorted(list(dict(sorted_file.labeled_nodes).keys()))
            for cluster in clusters:
                out = package_unit_lightweight(site, elec, cluster)
                if out is None:
                    print("Skipped {} {} {}".format(site, elec, cluster))
                    continue

                np.savez(
                    os.path.join(self.output()["waveform_folder"].path, out["id"]),
                    **out["waveforms"]
                )

                # fill in rewards
                reward_class = []
                subject = site.split("_")[0]
                for i in range(len(out["responses"])):
                    row = out["responses"].iloc[i]
                    if row["relation"] == "test":
                        reward = lookup_reward(subject, row["file"], pecking_test_data)
                        reward_class.append(reward)
                    else:
                        reward_class.append(None)
                out["responses"]["reward_class"] = reward_class

                out["responses"].to_pickle(
                    os.path.join(self.output()["packaged_folder"].path, out["id"])
                )
                print("Saved {} {} {}".format(site, elec, cluster))

        Path(self.output()["task_finished"].path).touch()



class PackageAllUnits(luigi.Task):
    def output(self):
        return luigi.LocalTarget("dumy")

    def requires(self):
        sitelist_path = os.path.join(BASE_PATH, "sitelist")
        with open(sitelist_path, "r") as sitelist:
            sites = sitelist.readlines()
        sites = [s.strip() for s in sites]

        for site in sites:
            subject = site.split("_")[0]
            site_dir = os.path.join("data", "birds", subject, "sites", site)
            yield PackageUnits(site_dir)

    def run(self):
        pass


class StackAllUnits(luigi.Task):
    def output(self):
        return luigi.LocalTarget(os.path.join(BASE_PATH, "UnitData.pkl"))

    def requires(self):
        return SetupSiteList()

    def run(self):
        with open(self.input().path, "r") as sitelist:
            sites = sitelist.readlines()
        sites = [s.strip() for s in sites]

        dfs = []
        for site in sites:
            subject = site.split("_")[0]
            site_dir = os.path.join("data", "birds", subject, "sites", site)
            for f in glob.glob(os.path.join(site_dir, "packaged_unit_stim_responses", "*")):
                dfs.append(pd.read_pickle(f))

        df = pd.concat(dfs, axis=0)
        df["stimulus_file"] = df["file"]
        df = df.drop("file", axis=1)
        df.to_pickle(self.output().path)


class PackageStimuli(luigi.Task):
    def output(self):
        return luigi.LocalTarget(os.path.join(BASE_PATH, "StimData.pkl"))

    def requires(self):
        return SetupStimFolder()

    def run(self):
        stim_files = glob.glob(os.path.join(self.input().path, "*.wav"))
        df = package_stims_and_renditions(stim_files)
        df.to_pickle(self.output().path)


class PackageUnitWaveforms(luigi.Task):
    def output(self):
        return luigi.LocalTarget("asdf")

    def requires(self):
        return SetupSiteList()

    def run(self):
        with open(self.input().path, "r") as sitelist:
            sites = sitelist.readlines()
        sites = [s.strip() for s in sites]

        save_path = os.path.join(BASE_PATH, "unit_waveforms")
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for site in sites:
            subject = site.split("_")[0]
            site_dir = os.path.join("data", "birds", subject, "sites", site)
            for f in glob.glob(os.path.join(site_dir, "packaged_unit_waveforms", "*")):
                shutil.copy(f, os.path.join(save_path, os.path.basename(f)))
