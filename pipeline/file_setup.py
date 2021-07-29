import os

import numpy as np
import luigi
import soundfile
from neo.io import RHDIO, NixIO

from format.download import downloaders
from format.io import load_yaml, save_yaml
from format.lazy import create_lazy_signals_from_nix
from format.wav import get_wavfile_data_from_nix
from targets import YamlTarget
from pipeline_utils import (
    associate_pyoperant_csvs_to_rhds,
    get_session_sites,
    parse_filelist,
    session_basename,
)


class SiteFileList(luigi.Task):
    """Checks if a file list file exists

    sites/
        SITE/
            filelist [output]
    """
    site_dir = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(os.path.join(self.site_dir, "filelist"))

    def run(self):
        raise Exception("The filelist file must be manually created")


class SiteConfig(luigi.Task):
    site_dir = luigi.Parameter()
    required_keys = luigi.ListParameter(default=["download_type", "experiment_name", "filelist"])

    def output(self):
        return YamlTarget(os.path.join(self.site_dir, "site_config.yaml"), *self.required_keys)

    def run(self):
        raise Exception("The site_config.yaml file with keys {} must be manually created."
                " One or more keys are missing".format(self.required_keys))


class SessionConfig(luigi.Task):
    session_dir = luigi.Parameter()
    required_keys = luigi.ListParameter(default=["download_type", "experiment_name", "filelist", "sessions"])

    def output(self):
        site_dirs, _ = get_session_sites(self.session_dir)
        if not len(site_dirs):
            raise Exception("No valid site with a config file found for session {}".format(self.session_dir))
        return YamlTarget(os.path.join(site_dirs[0], "site_config.yaml"), *self.required_keys)

    def run(self):
        raise Exception("The site_config.yaml file with keys {} must be manually created."
                " One or more keys are missing".format(self.required_keys))


class SignalsConfig(luigi.Task):
    site_dir = luigi.Parameter()

    def output(self):
        return YamlTarget(
            os.path.join(self.site_dir, "site_config.yaml"),
            "nix2wav",
            "filtering",
        )

    def run(self):
        raise Exception("""The nix2wav and filtering keys in site_config.yaml must be manually created.

        Example:

        nix2wav:
            ephys.wav:
                - ["Amplifier", [0, 15]]
                - ["Amplifier", [16, 31]]
            mic.wav:
                - ["Amplifier", 15]
                - ["Amplifier", 31]
                - ["Board ADC", [0, 3]]
            motion.wav:
                - ["AUX Input", [0, 6]]
        filtering:
            ephys.wav:
                - [0, 15]
                - [15, 30]
        """)


class DefineSessions(luigi.Task):
    """Determine how rhd and csv files should be combined into sessions

    Uses timestamps of RHD filenames and timestamps within pyoperant CSV
    files to figure out which CSVs go with which RHDs
    """
    site_dir = luigi.Parameter()

    def requires(self):
        return SiteConfig(self.site_dir)

    def output(self):
        return YamlTarget(os.path.join(self.site_dir, "site_config.yaml"), "sessions")

    def run(self):
        config = self.input().data()

        if config["download_type"] not in downloaders.keys():
            raise ValueError("The 'download_type' in {} must be one of {}".format(
                self.input().path, ", ".join(downloaders.keys())
            ))

        session_dicts = associate_pyoperant_csvs_to_rhds(
            csv_files=[f for f in config["filelist"] if f.endswith(".csv")],
            rhd_files=[f for f in config["filelist"] if f.endswith(".rhd")],
            experiment_name=config["experiment_name"],
            download_type=config["download_type"],
        )
        config["sessions"] = session_dicts

        save_yaml(self.output().path, config)


class DownloadSession(luigi.Task):
    """
    Download a session and pyoperant data for that session from google drive.

    Looks in google drive to create a directory

    /birds/<BIRDNAME>/sessions/<session_dir>/raw/

    and puts in an rhd file and 0+ pyoperant files
    """

    session_dir = luigi.Parameter()
    force = luigi.BoolParameter(default=False)

    def requires(self):
        self.session_name = session_basename(self.session_dir)
        output_directory = os.path.join(self.session_dir, "raw")
        self.site_dirs, self.pyoperants = get_session_sites(self.session_dir)

        return DefineSessions(self.site_dirs[0])

    def output(self):
        if self.force:
            return luigi.LocalTarget("dummy")
            
        self.session_name = session_basename(self.session_dir)
        output_directory = os.path.join(self.session_dir, "raw")
        self.site_dirs, self.pyoperants = get_session_sites(self.session_dir)

        return {
            "rhd": luigi.LocalTarget(os.path.join(
                output_directory, "{}.rhd".format(self.session_name)
            )),
            "pyoperants": [
                luigi.LocalTarget(os.path.join(
                    output_directory, "{}.csv".format(filename)
                )) for filename in self.pyoperants
            ]
        }

    def run(self):
        """Use the yaml file to download RHDs and pyoperant csvs for this session
        """
        config = self.input().data()

        if not len(self.site_dirs):
            raise Exception("No sites were found with site_config.yaml containing {}".format(self.session_dir))

        downloader = downloaders[config["download_type"]]

        downloader.download_session(
            config["experiment_name"],
            self.session_name,
            save_to=self.session_dir,
            pyoperant_files=self.pyoperants,
        )


class DownloadSite(luigi.Task):
    site_dir = luigi.Parameter()
    force = luigi.BoolParameter(default=False)

    def output(self):
        # Force this task to always run if requested
        return luigi.LocalTarget("dummy")

    def requires(self):
        return DefineSessions(self.site_dir)

    def run(self):
        config = self.input().data()

        for session in config["sessions"]:
            session_dir = os.path.join(os.path.dirname(os.path.dirname(self.site_dir)), "sessions", session["id"])
            if not os.path.exists(session_dir):
                os.makedirs(session_dir)
            yield DownloadSession(session_dir, force=self.force)


class Rhd2Nix(luigi.Task):
    """Converts an RHD file to a NIX file, and clean up the RHD file

    Inputs
    ------
    session_dir : str
        Path to session

    Outputs a nix file as raw/{session_name}.nix
    """

    session_dir = luigi.Parameter()

    def requires(self):
        return {
            "files": DownloadSession(self.session_dir),
            "config": SessionConfig(self.session_dir),
        }

    def output(self):
        raw_dir = os.path.join(self.session_dir, "raw")
        if not os.path.exists(raw_dir):
            os.makedirs(raw_dir)

        nix_filename = "{}.nix".format(session_basename(self.session_dir))

        # Temporarily move nix file from segmentation dir to raw dir
        if os.path.exists(os.path.join(self.session_dir, "segmentation", nix_filename)):
            import shutil
            shutil.move(os.path.join(self.session_dir, "segmentation", nix_filename), os.path.join(raw_dir, nix_filename))

        return luigi.LocalTarget(os.path.join(raw_dir, nix_filename))

    def run(self):
        rhd_file = self.input()["files"]["rhd"].path
        config = self.input()["config"].data()

        # Delete the digital input form the rhd file if from Seewiesen.
        # These are not used for stim id but to sync with bat logger
        # TODO: this is not a great place for this in case we want to keep
        # the digital codes at some point? and seewiesen also doesnt apply to the owncloud data either...
        if (config["download_type"] == "seewiesen"):
            print("Seewiesen Data: digital events in rhd will be deleted")
            storeDIG = False
        else:
            storeDIG = True

        rhd = RHDIO(rhd_file)
        blk = rhd.read_block(storeDIG=storeDIG)
        nix = NixIO(self.output().path)
        nix.write_block(blk)
        nix.close()

        # Remove the RHD file now that we have a NIX file
        # os.remove(inputs["rhd"].path)


class Nix2Wav(luigi.Task):
    """Write analogsignals to disk as wav files (RF64 format)

    Outputs files for the microphone and electrode data arrays,
    and also writes the mapping between the channels in these files
    to their original channels into the configuration file.

    The microphone data is also normalized and zero-centered according
    to the filter groups defined in the config "filtering" key

    Inputs
    ------
    session_dir : str
        Path to session
    """

    session_dir = luigi.Parameter()

    def requires(self):
        return {
            "nix": Rhd2Nix(self.session_dir),
            "config": SessionConfig(self.session_dir, ["nix2wav", "filtering"])
        }

    def output(self):
        raw_dir = os.path.join(self.session_dir, "raw")
        mic_path = os.path.join(raw_dir, "mic.wav")
        elec_path = os.path.join(raw_dir, "ephys.wav")
        motion_path = os.path.join(raw_dir, "motion.wav")

        site_dirs, _ = get_session_sites(self.session_dir)
        config_path = os.path.join(site_dirs[0], "site_config.yaml")

        return {
            "mic.wav": luigi.LocalTarget(mic_path),
            "ephys.wav": luigi.LocalTarget(elec_path),
            "motion.wav": luigi.LocalTarget(motion_path),
            "config_original_channels": YamlTarget(config_path, "wav_to_original_channels")
        }

    def run(self):
        outputs = self.output()
        nix_file = self.input()["nix"].path
        config = self.input()["config"].data()

        arrays = get_wavfile_data_from_nix(self.input()["nix"].path, config["nix2wav"])

        with soundfile.SoundFile(
                self.output()["mic.wav"].path,
                "w+",
                samplerate=arrays["mic"]["sampling_rate"],
                format="RF64",
                subtype="FLOAT",
                channels=arrays["mic"]["data"].shape[1]) as f:
            print(type(arrays["mic"]["data"]))
            arrays["mic"]["data"] = np.array(arrays["mic"]["data"])
            arrays["mic"]["data"] -= np.median(arrays["mic"]["data"], axis=0)
            for group_start, group_end in config["filtering"]["mic.wav"]:
                arrays["mic"]["data"][:, group_start:group_end] /= np.max(np.abs(arrays["mic"]["data"][:, group_start:group_end]))
            f.write(arrays["mic"]["data"])

        if not os.path.exists(self.output()["ephys.wav"].path):
            with soundfile.SoundFile(
                    self.output()["ephys.wav"].path,
                    "w+",
                    samplerate=arrays["ephys"]["sampling_rate"],
                    format="RF64",
                    subtype="FLOAT",
                    channels=arrays["ephys"]["data"].shape[1]) as f:
                f.write(arrays["ephys"]["data"])

        if "motion" in arrays and arrays["motion"]["data"] is not None:
            with soundfile.SoundFile(
                    self.output()["motion.wav"].path,
                    "w+",
                    samplerate=arrays["motion"]["sampling_rate"],
                    format="RF64",
                    subtype="FLOAT",
                    channels=arrays["motion"]["data"].shape[1]) as f:
                f.write(arrays["motion"]["data"])
        else:
            open(self.output()["motion.wav"].path, 'a').close()

        # Finally, write the channel name mappings to the configuration file
        # This is kind of an extra thing; not sure if I should put it in the
        # task outputs / set it as a new task, or what?

        # FIXME: The NixIO seems to change the order of the signals...
        # The order in the .rhd file is fine, why is it when saved to/loaded from nix
        # the order gets messed up? This can mess up any tasks that rely
        # on plain integer indexing (like filtering) so need to investigate this
        config_out = self.output()["config_original_channels"].data()
        config_out["wav_to_original_channels"] = {
            "{}.wav".format(k): v["channel_names"] for k, v in arrays.items()
        }
        save_yaml(self.output()["config_original_channels"].path, config_out)


class Site2Wav(luigi.Task):
    """Save all of site's raw data into wav files
    """
    site_dir = luigi.Parameter()

    def requires(self):
        return SiteConfig(site_dir=self.site_dir, required_keys=["nix2wav", "filtering"])

    def run(self):
        config = self.input().data()
        for session in config["sessions"]:
            session_dir = os.path.join(
                os.path.dirname(os.path.dirname(self.site_dir)), "sessions", session["id"]
            )
            yield Nix2Wav(session_dir=session_dir)
