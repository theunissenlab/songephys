"""Pipeline tasks for packaging ephys data for use in a separate repo
"""
import asyncio
import os
import functools
import glob
import logging
import requests
import shutil
import signal
import sys
from pathlib import Path

import luigi
import tqdm

import suss.io
import numpy as np
import pandas as pd

from ..postprocess.trials import package_unit_lightweight, package_stims_and_renditions, from_unit_id
from ..preprocess.ephys.detect import recut_waveforms
from ..postprocess.memory_test import lookup_reward
from .preprocess_ephys import ManualSorting, MergeLazySignals


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
    force = luigi.BoolParameter(default=False)

    def output(self):
        packaged_dir = os.path.join(self.site_dir, "packaged_unit_stim_responses")
        threshold_dir = os.path.join(self.site_dir, "packaged_unit_threshold_metadata")
        waveform_dir = os.path.join(self.site_dir, "packaged_unit_waveforms")
        if not self.force:
            finished_marker = os.path.join(self.site_dir, "packaged_unit_stim_responses", ".luigi_task_finished")
        else:
            finished_marker = "asdfasdfasdfblahsldkfhalsdjfa"

        return {
            "packaged_folder": luigi.LocalTarget(packaged_dir),
            "threshold_metadata_folder": luigi.LocalTarget(threshold_dir),
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

        threshold_dir = self.output()["threshold_metadata_folder"].path
        if not os.path.exists(threshold_dir):
            os.makedirs(threshold_dir)

        site = os.path.basename(self.site_dir)

        for elec, sort_result in tqdm.tqdm(
                    enumerate(self.input()["sort_results"]),
                    total=len(self.input()["sort_results"])
                ):
            sorted_file = suss.io.read_pickle(sort_result.path)
            clusters = sorted(list(dict(sorted_file.labeled_nodes).keys()))
            for cluster in clusters:
                out = package_unit_lightweight(site, elec, cluster)
                if out is None:
                    # print("Skipped {} {} {}".format(site, elec, cluster))
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

                np.savez(
                    os.path.join(self.output()["threshold_metadata_folder"].path, out["id"]),
                    **out["threshold_metadata"]
                )

        if not self.force:
            Path(self.output()["task_finished"].path).touch()


class RecutWaveformSnippets(luigi.Task):
    """Takes unit waveforms produced in PackageUnitUnits and recuts them with a different time window
    """
    site_dir = luigi.Parameter()
    t_pre = luigi.FloatParameter()
    t_post = luigi.FloatParameter()

    def output(self):
        waveform_dir = os.path.join(self.site_dir, "recut_unit_waveforms_{}_{}".format(self.t_pre, self.t_post))
        finished_marker = os.path.join(waveform_dir, ".luigi_task_finished")

        return {
            "waveform_folder": luigi.LocalTarget(waveform_dir),
            "task_finished": luigi.LocalTarget(finished_marker)
        }

    def requires(self):
        return {
            "packaged_units": PackageUnits(site_dir=self.site_dir),
            "lazy": MergeLazySignals(site_dir=self.site_dir),
        }

    def run(self):
        from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

        waveform_dir = self.output()["waveform_folder"].path
        if not os.path.exists(waveform_dir):
            os.makedirs(waveform_dir)

        executor = ProcessPoolExecutor(max_workers=16)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.async_run(executor))

    async def async_run(self, executor):
        try:
            import tqdm
        except ImportError:
            tqdm = None

        futures = []
        wf_files = glob.glob(os.path.join(self.input()["packaged_units"]["waveform_folder"].path, "e*u*s*.npz"))
        wf_files = sorted(wf_files, key=os.path.getsize, reverse=True)
        lazy_site = np.load(self.input()["lazy"].path, allow_pickle=True)[()]
        loop = asyncio.get_event_loop()

        for idx, wf_file in enumerate(wf_files):
            unit_id = os.path.splitext(os.path.basename(wf_file))[0]
            with np.load(wf_file) as data:
                wf_data = data
                spike_times = wf_data["spike_times"]
            unit_info = from_unit_id(unit_id)

            future = loop.run_in_executor(executor, functools.partial(recut_waveforms,
                lazy_site,
                spike_times,
                elec_id=unit_info["electrode"],
                t_pre=self.t_pre,
                t_post=self.t_post,
                tqdm_fn=functools.partial(tqdm.tqdm, position=idx, desc="{} : {}".format(idx, unit_id)) if tqdm is not None else None
            ))
            futures.append(future)

        results = await asyncio.gather(*futures)

        for wf_file, (t_arr, wfs, peak_idx, sampling_rate) in zip(wf_files, results):
            unit_id = os.path.splitext(os.path.basename(wf_file))[0]
            with np.load(wf_file) as data:
                wf_data = data
                spike_times = wf_data["spike_times"]

            outfile = os.path.join(self.output()["waveform_folder"].path, unit_id)
            np.savez(
                outfile,
                spike_times=spike_times,
                spike_waveforms=wfs,
                peak_idx=peak_idx,
                t_arr=t_arr,
                sampling_rate=sampling_rate,
            )
            print("Saved {}".format(outfile))

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


class PackageRecutUnitWaveforms(luigi.Task):
    t_pre = luigi.FloatParameter()
    t_post = luigi.FloatParameter()

    def output(self):
        return luigi.LocalTarget("asdf")

    def requires(self):
        sitelist_path = os.path.join(BASE_PATH, "sitelist")
        with open(sitelist_path, "r") as sitelist:
            sites = sitelist.readlines()
        sites = [s.strip() for s in sites]

        for site in sites:
            subject = site.split("_")[0]
            site_dir = os.path.join("data", "birds", subject, "sites", site)
            yield RecutWaveformSnippets(site_dir, t_pre=self.t_pre, t_post=self.t_post)

    def run(self):
        logger = logging.getLogger('luigi-interface')
        sitelist_path = os.path.join(BASE_PATH, "sitelist")
        with open(sitelist_path, "r") as sitelist:
            sites = sitelist.readlines()
        sites = [s.strip() for s in sites]

        save_path = os.path.join(BASE_PATH, "unit_waveforms_{}_{}".format(self.t_pre, self.t_post))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for site in sites:
            subject = site.split("_")[0]
            site_dir = os.path.join("data", "birds", subject, "sites", site)
            for f in glob.glob(os.path.join(site_dir, "recut_unit_waveforms_{}_{}".format(self.t_pre, self.t_post), "*")):
                logger.info("Copying {} to {}".format(f, os.path.join(save_path, os.path.basename(f))))
                shutil.copy(f, os.path.join(save_path, os.path.basename(f)))


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


class BackupPackageUnits(luigi.Task):
    def output(self):
        return luigi.LocalTarget("imacrazyperson")

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
            old_dir = os.path.join(site_dir, "packaged_unit_stim_responses")
            shutil.move(old_dir, "{}.0.bak".format(old_dir))


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
