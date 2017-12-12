from __future__ import print_function

import cPickle as pickle
import os

import glob
import luigi
import pandas as pd

import config
from gdrive_access import download
from segmentation.audio_segment import combine_stim_csvs
from segmentation.audio_segment import load_stimulus_database, save_stimulus_database


class DownloadSession(luigi.Task):
    
    session_name = luigi.Parameter()
    pyoperants = luigi.ListParameter()
    notes = luigi.Parameter(default=None)

    def output(self):
        bird_name = self.session_name.split("_")[0]
        output_directory = os.path.join(
                config.DATADIR,
                "birds",
                bird_name,
                self.session_name,
                "raw")

        return luigi.LocalTarget(output_directory)

    def run(self):
        download.download_session(
            self.session_name,
            self.pyoperants)
        if self.notes:
            with open(os.path.join(self.output().path, "NOTES"), "a+") as notefile:
                notefile.write(notes)


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


class CombineTrialMetadata(luigi.Task):

    session_dir = luigi.Parameter()

    def requires(self):
        return DownloadSession(os.path.basename(self.session_dir))

    def output(self):
        segmentation_dir = os.path.join(self.session_dir, "segmentation")
        if not os.path.exists(segmentation_dir):
            os.makedirs(segmentation_dir)

        combined_db = "playbacks_db"
        combined_csv = "playbacks.csv"

        return [
            luigi.LocalTarget(os.path.join(segmentation_dir, combined_db)),
            luigi.LocalTarget(os.path.join(segmentation_dir, combined_csv))
        ]


    def run(self):
        input_directory = os.path.join(self.session_dir, "raw")
        stimuli_directory = os.path.join(self.session_dir, "stimuli")

        csv_files = glob.glob(os.path.join(input_directory, "*.csv"))
        if not len(csv_files):
            # TODO: im just creating empty files here if nothing
            pd.DataFrame([]).to_csv(self.output()[1].path)
            with open(output()[0].path, "a"):
                pass
            return

        # Identify the stimulus directory (e.g. TutFamBos) associated
        # with each pyoperant csv
        stim_dirs = []
        possible_stim_dirs = [os.path.basename(f) for f in glob.glob(os.path.join(stimuli_directory, "*"))]

        for csv_file in csv_files:
            print("Looking for stim dir for {}".format(csv_file))
            stimulus_names = pd.read_csv(csv_file)["stimulus_name"]
            
            stim_dir_used = None
            for stim_dir in possible_stim_dirs:
                if any([(stim_dir in x) for x in stimulus_names]):
                    print("Matched {}".format(stim_dir))
                    stim_dir_used = os.path.join(stimuli_directory, stim_dir)
                    break

            if stim_dir_used is None:
                raise Exception("Couldn't find the stim dir for {}".format(csv_file))

            stim_dirs.append(stim_dir_used)

        # Save the combined csv and also the database
        new_csv = combine_stim_csvs(csv_files, stim_dirs)
        new_csv.to_csv(self.output()[1].path)

        print("Saving stim database {}".format(self.output()[0].path))
        save_stimulus_database(
                self.output()[1].path,
                self.output()[0].path)
