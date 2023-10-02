import os
import datetime

import numpy as np
import luigi
import soundfile
from neo.io import NixIO, IntanIO, NWBIO

from ..format.download import downloaders
from ..format.io import load_yaml, save_yaml
from ..format.wav import get_wavfile_data_from_neo,get_raw_data_from_neo
from .targets import YamlTarget
from .pipeline_utils import (
    SessionParameter,
    SiteParameter,
    associate_pyoperant_csvs_to_rhds,
    get_session_sites,
    parse_filelist,
    session_basename,
)


class SiteConfig(luigi.Task):
    """Verify config file exists and specific keys exist

    Use as a dependency in requires() to make sure a task can only be run
    if the given configuration keys are defined

    Example Usage
    -------------
    >>> class ExampleTask(luigi.Task):
    ...     REQUIRED_KEYS = ["required_key_1", "required_key_2"]
    ...     site_dir = SiteParameter()
    ...     def requires(self):
    ...         return SiteConfig(self.site_dir, required_keys=self.REQUIRED_KEYS)
    ...     def run(self):
    ...         config = self.input().data()
    ...         assert "required_key_1" in config
    ...         assert "required_key_2" in config

    Arguments
    ---------
    site_dir : str
        Path to site
    required_keys : List[str] (default = ["download_type", "experiment_name", "filelist"])
        List of keys that must exist in the config file

    Output
    ------
    The keys in `required_keys` defined in `{site_dir}/site_config.yaml`.
    """
    site_dir = SiteParameter()
    required_keys = luigi.ListParameter(
        default=["download_type", "experiment_name", "filelist"])

    def config_path(self):
        return os.path.join(self.site_dir, "site_config.yaml")

    def output(self):
        return YamlTarget(self.config_path(), *self.required_keys)

    def run(self):
        raise Exception("""
        The keys {} are required but not present in the config file found for {}
        Config file: {}
        """.format(self.required_keys, self.site_dir, self.config_path()))


class SessionConfig(luigi.Task):
    """Verify config file exists and specific keys exist

    The configuration is located by finding a site whose config contains
    the given session as one of its sessions.

    Use as a dependency in requires() to make sure a task can only be run
    if the given configuration keys are defined

    Example Usage
    -------------
    >>> class ExampleTask(luigi.Task):
    ...     REQUIRED_KEYS = ["required_key_1", "required_key_2"]
    ...     session_dir = SessionParameter()
    ...     def requires(self):
    ...         return SessionConfig(self.session_dir, required_keys=self.REQUIRED_KEYS)
    ...     def run(self):
    ...         session_config = self.input().data()
    ...         assert "required_key_1" in session_config
    ...         assert "required_key_2" in session_config

    Arguments
    ---------
    session_dir : str
        Path to session
    required_keys : List[str] (default = ["download_type", "experiment_name", "filelist"])
        List of keys that must exist in the config file

    Output
    ------
    The keys in `required_keys` defined in `{site_dir}/site_config.yaml`, where site_dir is
    a site whose site_config.yaml contains `session_dir` in its `sessions` list.
    """
    session_dir = SessionParameter()
    required_keys = luigi.ListParameter(
        default=["download_type", "experiment_name", "filelist", "sessions"])

    def config_path(self):
        site_dirs, _, _ = get_session_sites(self.session_dir)
        if not len(site_dirs):
            raise Exception(
                "No valid site with a config file found for session {}".format(self.session_dir))
        return os.path.join(site_dirs[0], "site_config.yaml")

    def output(self):
        return YamlTarget(self.config_path(), *self.required_keys)

    def run(self):
        raise Exception("""
        The keys {} are required but not present in the config file found for {}
        Config file: {}
        """.format(self.required_keys, self.session_dir, self.config_path()))


class SignalsConfig(luigi.Task):
    """Verify config file exists with electrode layout configured properly

    Use as a dependency in requires() to make sure a task can only be run
    if the config variables already exist

    Example Usage
    -------------
    >>> class ExampleTask(luigi.Task):
    ...     site_dir = SiteParameter()
    ...     def requires(self):
    ...         return SignalsConfig(self.site_dir)
    ...     def run(self):
    ...         site_config = self.input().data()
    ...         assert "wavmap" in site_config
    ...         assert "filtering" in site_config
    ...         assert "playback_mic_channels" in site_config

    Arguments
    ---------
    site_dir : str
        Path to site

    Output
    ------
    "wavmap", "filtering", and "playback_mic_channels" must exist in the config file.
    """
    site_dir = SiteParameter()

    def output(self):
        return YamlTarget(
            os.path.join(self.site_dir, "site_config.yaml"),
            "wavmap",
            "filtering",
        )

    def run(self):
        raise Exception("""The nix2wav and filtering keys in site_config.yaml must be manually created.

        Example:

        wavmap:
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


class SiteProjectConfig(luigi.Task):
    """Verify config file exists and specific keys exist

    Use as a dependency in requires() to make sure a task can only be run
    if the given configuration keys are defined

    Example Usage
    -------------
    >>> class ExampleTask(luigi.Task):
    ...     REQUIRED_KEYS = ["required_key_1", "required_key_2"]
    ...     site_dir = SiteParameter()
    ...     def requires(self):
    ...         return ProjectConfig(self.site_dir, required_keys=self.REQUIRED_KEYS)
    ...     def run(self):
    ...         config = self.input().data()
    ...         assert "required_key_1" in config
    ...         assert "required_key_2" in config

    Arguments
    ---------
    site_dir : str
        Path to site
    required_keys : List[str] (default = ["EXPERIMENT_NAME", "DATADIR", "EPHYS"])
        List of keys that must exist in the config file

    Output
    ------
    The keys in `required_keys` defined in `{site_dir}/site_config.yaml`.
    """
    site_dir = SiteParameter()
    required_keys = luigi.ListParameter(
        default=["EXPERIMENT_NAME", "DATADIR", "EPHYS"])

    def config_path(self):
        # get project_dir from site_dir
        project_dir = self.site_dir.split("sites")[0]
        return os.path.join(project_dir, "project_config.yaml")

    def output(self):
        return YamlTarget(self.config_path(), *self.required_keys)

    def run(self):
        raise Exception("""
        The keys {} are required but not present in the config file found for {}
        Config file: {}
        """.format(self.required_keys, self.site_dir, self.config_path()))


class SessionProjectConfig(luigi.Task):
    """Verify config file exists and specific keys exist

    Use as a dependency in requires() to make sure a task can only be run
    if the given configuration keys are defined

    Example Usage
    -------------
    >>> class ExampleTask(luigi.Task):
    ...     REQUIRED_KEYS = ["required_key_1", "required_key_2"]
    ...     session_dir = SessionParameter()
    ...     def requires(self):
    ...         return SessionProjectConfig(self.session_dir, required_keys=self.REQUIRED_KEYS)
    ...     def run(self):
    ...         config = self.input().data()
    ...         assert "required_key_1" in config
    ...         assert "required_key_2" in config

    Arguments
    ---------
    site_dir : str
        Path to site
    required_keys : List[str] (default = ["EXPERIMENT_NAME", "DATADIR", "EPHYS"])
        List of keys that must exist in the config file

    Output
    ------
    The keys in `required_keys` defined in `{site_dir}/site_config.yaml`.
    """
    session_dir = SessionParameter()
    required_keys = luigi.ListParameter(
        default=["EXPERIMENT_NAME", "DATADIR", "EPHYS"])

    def config_path(self):
        # get project_dir from session_dir
        project_dir = self.session_dir.split("sessions")[0]
        return os.path.join(project_dir, "project_config.yaml")

    def output(self):
        return YamlTarget(self.config_path(), *self.required_keys)

    def run(self):
        raise Exception("""
        The keys {} are required but not present in the config file found for {}
        Config file: {}
        """.format(self.required_keys, self.session_dir, self.config_path()))


class DefineSessions(luigi.Task):
    """Converts a list of RHD and CSV files into a list of session dicts

    Uses timestamps of RHD filenames and timestamps within pyoperant CSV
    files to figure out which CSVs go with which RHDs

    Example
    -------
    luigi --module file_setup DefineSessions --site-dir SITE_DIR

    Output
    ------
    "sessions" key in site_config.yaml, each element is a dict with "id", "sessions", and "playbacks"
    """
    site_dir = SiteParameter()

    def requires(self):
        return { 
            "site_config": SiteConfig(self.site_dir), 
            "project_config": SiteProjectConfig(self.site_dir)
            }

    def output(self):
        return YamlTarget(os.path.join(self.site_dir, "site_config.yaml"), "sessions")

    def run(self):
        site_config = self.input()['site_config'].data()
        project_config = self.input()['project_config'].data()

        if site_config["download_type"] not in downloaders.keys():
            raise ValueError("The 'download_type' in {} must be one of {}".format(
                self.input().path, ", ".join(downloaders.keys())
            ))
        # instantiate the downloader
        downloader = downloaders[site_config['download_type']](cfg = project_config)

        session_dicts = associate_pyoperant_csvs_to_rhds(
            csv_files=[f for f in site_config["filelist"]
                       if f.endswith(".csv")],
            rhd_files=[f for f in site_config["filelist"]
                       if f.endswith(".rhd")],
            experiment_name=site_config["experiment_name"],
            downloader=downloader,
            include_events=np.any([f.endswith("events.log") for f in site_config['filelist']])
        )
        site_config["sessions"] = session_dicts

        save_yaml(self.output().path, site_config)


class DownloadSession(luigi.Task):
    """
    Download a session and pyoperant data for that session using a downloader defined in format.download

    Creates a directory under `DATA_DIR/birds/EXPERIMENT_NAME/sessions/SESSION_DIR/raw` and
    saves the rhd file, any number of pyoperant CSV files, and stimulus folders

    Example
    -------
    ``luigi --module file_setup DownloadSession --session-dir SESSION_DIR``

    Arguments
    ---------
    session_dir : str
    force : bool (default False)
        Force the task to run whether or not the files had been downloaded already

    Output
    ------
    Folders: `session_dir/raw/SESSION_NAME.rhd`, `session_dir/raw/*.csv`, and
    `session_dir/stimuli/*`
    """

    session_dir = SessionParameter()
    force = luigi.BoolParameter(default=False)

    def requires(self):
        self.session_name = session_basename(self.session_dir)
        output_directory = os.path.join(self.session_dir, "raw")
        self.site_dirs, self.pyoperants, self.events = get_session_sites(self.session_dir)

        return {
            "define_sessions": DefineSessions(self.site_dirs[0]),
            "project_config": SessionProjectConfig(self.session_dir)
        }

    def output(self):
        if self.force:
            return luigi.LocalTarget("dummy")

        self.session_name = session_basename(self.session_dir)
        output_directory = os.path.join(self.session_dir, "raw")
        self.site_dirs, self.pyoperants, self.events = get_session_sites(self.session_dir)

        return {
            "rhd": luigi.LocalTarget(os.path.join(
                output_directory, "{}.rhd".format(self.session_name)
            )),
            "pyoperants": [
                luigi.LocalTarget(os.path.join(
                    output_directory, "{}.csv".format(filename)
                )) for filename in self.pyoperants
            ],
            "events": [
                luigi.LocalTarget(os.path.join(
                    output_directory, "{}.log".format(filename)
                )) for filename in self.events
            ]
        }

    def run(self):
        """Use the yaml file to download RHDs and pyoperant csvs for this session
        """
        session_config = self.input()['define_sessions'].data()
        project_cfg = self.input()['project_config'].data()
        if not len(self.site_dirs):
            raise Exception(
                "No sites were found with site_config.yaml containing {}".format(self.session_dir))

        # instantiate the downloader with the project_config
        downloader = downloaders[session_config["download_type"]](cfg = project_cfg)

        downloader.download_session(
            session_config["experiment_name"],
            self.session_name,
            save_to=self.session_dir,
            pyoperant_files=self.pyoperants,
            event_files = self.events
        )


class DownloadSite(luigi.Task):
    site_dir = SiteParameter()
    force = luigi.BoolParameter(default=False)

    def output(self):
        # Force this task to always run if requested
        return luigi.LocalTarget("dummy")

    def requires(self):
        return DefineSessions(self.site_dir)

    def run(self):
        site_config = self.input().data()

        for session in site_config["sessions"]:
            session_dir = os.path.join(os.path.dirname(
                os.path.dirname(self.site_dir)), "sessions", session["id"])
            if not os.path.exists(session_dir):
                os.makedirs(session_dir)

            if self.force:
                try:
                    DownloadSession(session_dir, force=self.force).run()
                except Exception as e:
                    print("Download session {} had an error {}. But going to try to download all because force=True".format(
                        session_dir,
                        str(e)))
            else:
                yield DownloadSession(session_dir, force=self.force)


class GenerateSiteMemmaps(luigi.Task):
    """
    Generate numpy binary arrays of ephys data.
    
    This is used as input to kilosort
    Generates on .dat file for each session containing all channels of ephys
    """
    site_dir = SiteParameter()
    force = luigi.BoolParameter(default=False)

    def requires(self):
        return {
            "site_config": DefineSessions(self.site_dir),
            "project_config": SiteProjectConfig(self.site_dir)
        }
    
    def output(self):
        site_config = self.input()["site_config"].path
        # Force this task to always run if requested
        # TODO: Fix this by checking for files
        if self.force:
            return luigi.LocalTarget("dummy")
        return YamlTarget(site_config,'bin_arrays_generated')
    
    def run(self):
        site_config = self.input()["site_config"].data()
        sessions_dir = os.path.join(os.path.dirname(os.path.dirname(self.site_dir)), "sessions")
        all_sessions = site_config["sessions"]
        for i,session in enumerate(all_sessions):
            bin_output = yield Rhd2Binary(os.path.join(sessions_dir, session["id"]))
            all_sessions[i]['raw_bin'] = bin_output["ephys.bin"].path
        site_config['sessions'] = all_sessions
        site_config['bin_arrays_generated'] = True
        save_yaml(self.input()["site_config"].path, site_config)

class Rhd2Nix(luigi.Task):
    """Converts an RHD file to a NIX file, and clean up the RHD file

    Inputs
    ------
    session_dir : str
        Path to session

    Outputs a nix file as raw/{session_name}.nix
    """

    session_dir = SessionParameter()

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
            shutil.move(os.path.join(self.session_dir, "segmentation",
                        nix_filename), os.path.join(raw_dir, nix_filename))

        return luigi.LocalTarget(os.path.join(raw_dir, nix_filename))

    def run(self):
        rhd_file = self.input()["files"]["rhd"].path
        session_config = self.input()["config"].data()

        # Delete the digital input form the rhd file if from Seewiesen.
        # These are not used for stim id but to sync with bat logger
        # TODO: this is not a great place for this in case we want to keep
        # the digital codes at some point? and seewiesen also doesnt apply to the owncloud data either...
        if (session_config["download_type"] == "seewiesen"):
            print("Seewiesen Data: digital events in rhd will be deleted")
            storeDIG = False
        else:
            storeDIG = True

        rhd = IntanIO(rhd_file)
        blk = rhd.read_block()
        blk.segments[0].annotations['signal_channels'] = [x[0]
                                                          for x in rhd.header['signal_channels']]
        blk.segments[0].annotations['event_channels'] = [x[0]
                                                         for x in rhd.header['event_channels']]
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

    session_dir = SessionParameter()

    def requires(self):
        return {
            "nix": Rhd2Nix(self.session_dir),
            "config": SessionConfig(self.session_dir, ["wavmap", "filtering"])
        }

    def output(self):
        raw_dir = os.path.join(self.session_dir, "raw")
        mic_path = os.path.join(raw_dir, "mic.wav")
        elec_path = os.path.join(raw_dir, "ephys.wav")
        motion_path = os.path.join(raw_dir, "motion.wav")

        site_dirs, _, _ = get_session_sites(self.session_dir)
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
        session_config = self.input()["config"].data()

        arrays = get_wavfile_data_from_neo(
            self.input()["nix"].path, session_config["wavmap"])

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
            for group_start, group_end in session_config["filtering"]["mic.wav"]:
                arrays["mic"]["data"][:, group_start:group_end] /= np.max(
                    np.abs(arrays["mic"]["data"][:, group_start:group_end]))
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


class Rhd2Wav(luigi.Task):
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

    session_dir = SessionParameter()

    def requires(self):
        return {
            "files": DownloadSession(self.session_dir),
            "config": SessionConfig(self.session_dir),
        }

    def output(self):
        raw_dir = os.path.join(self.session_dir, "raw")
        mic_path = os.path.join(raw_dir, "mic.wav")
        elec_path = os.path.join(raw_dir, "ephys.wav")
        motion_path = os.path.join(raw_dir, "motion.wav")

        site_dirs, _, _ = get_session_sites(self.session_dir)
        config_path = os.path.join(site_dirs[0], "site_config.yaml")

        return {
            "mic.wav": luigi.LocalTarget(mic_path),
            "ephys.wav": luigi.LocalTarget(elec_path),
            "motion.wav": luigi.LocalTarget(motion_path),
            "config_original_channels": YamlTarget(config_path, "wav_to_original_channels")
        }

    def run(self):

        rhd_file = self.input()["files"]["rhd"].path
        session_config = self.input()["config"].data()

        arrays = get_wavfile_data_from_neo(rhd_file, session_config["wavmap"])

        with soundfile.SoundFile(
                self.output()["mic.wav"].path,
                "w+",
                samplerate=arrays["mic"]["sampling_rate"],
                format="RF64",
                subtype="FLOAT",
                channels=arrays["mic"]["data"].shape[1]) as f:
            arrays["mic"]["data"] = np.array(arrays["mic"]["data"])
            arrays["mic"]["data"] -= np.median(arrays["mic"]["data"], axis=0)
            for group_start, group_end in session_config["filtering"]["mic.wav"]:
                arrays["mic"]["data"][:, group_start:group_end] /= np.max(
                    np.abs(arrays["mic"]["data"][:, group_start:group_end]))
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

class Rhd2Numpy(luigi.Task):
    """Write analogsignals to disk as numpy memmaps

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

    session_dir = SessionParameter()

    def requires(self):
        return {
            "files": DownloadSession(self.session_dir),
            "config": SessionConfig(self.session_dir),
        }

    def output(self):
        sess_name = os.path.basename(self.session_dir)
        raw_dir = os.path.join(self.session_dir, "raw")
        elec_path = os.path.join(raw_dir, "%s_16bit.npy"%sess_name)

        site_dirs, _, _ = get_session_sites(self.session_dir)
        config_path = os.path.join(site_dirs[0], "site_config.yaml")

        return {
            "ephys.npy": luigi.LocalTarget(elec_path),
            "config_original_channels": YamlTarget(config_path, "dat_to_original_channels")
        }

    def run(self):

        rhd_file = self.input()["files"]["rhd"].path
        session_config = self.input()["config"].data()

        arrays = get_raw_data_from_neo(rhd_file, session_config["wavmap"])

        if not os.path.exists(self.output()["ephys.npy"].path):
            with open(self.output()["ephys.npy"].path,'wb') as f:
                np.save(f,arrays['ephys']['data'])
        

        # Finally, write the channel name mappings to the configuration file
        # This is kind of an extra thing; not sure if I should put it in the
        # task outputs / set it as a new task, or what?

        # FIXME: The NixIO seems to change the order of the signals...
        # The order in the .rhd file is fine, why is it when saved to/loaded from nix
        # the order gets messed up? This can mess up any tasks that rely
        # on plain integer indexing (like filtering) so need to investigate this
        config_out = self.output()["config_original_channels"].data()
        config_out["dat_to_original_channels"] = {
            "{}.wav".format(k): v["channel_names"] for k, v in arrays.items()
        }
        
        save_yaml(self.output()["config_original_channels"].path, config_out)

class Rhd2Binary(luigi.Task):
    """Write analogsignals to disk as binary memmaps

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

    session_dir = SessionParameter()

    def requires(self):
        return {
            "files": DownloadSession(self.session_dir),
            "config": SessionConfig(self.session_dir),
        }

    def output(self):
        sess_name = os.path.basename(self.session_dir)
        raw_dir = os.path.join(self.session_dir, "raw")
        elec_path = os.path.join(raw_dir, "%s.bin"%sess_name)

        audio_path = os.path.join(raw_dir, "%s_audio.wav"%sess_name)
        
        motion_path = os.path.join(raw_dir, "%s_motion.npy"%sess_name)

        lfp_path = os.path.join(raw_dir,'%s_LFP.bin'%sess_name)
        site_dirs, _, _ = get_session_sites(self.session_dir)
        config_path = os.path.join(site_dirs[0], "site_config.yaml")

        return {
            "ephys.bin": luigi.LocalTarget(elec_path),
            "audio.wav": luigi.LocalTarget(audio_path),
            "motion.bin": luigi.LocalTarget(motion_path),
            "config_original_channels": YamlTarget(config_path, "bin_to_original_channels")
        }

    def run(self):

        rhd_file = self.input()["files"]["rhd"].path
        session_config = self.input()["config"].data()
        
        bad_chans = session_config.get('bad_channels',None)
        print("BAD CHANNS %s"%bad_chans)
        arrays = get_raw_data_from_neo(rhd_file, session_config["wavmap"],bad_chans)

        if not os.path.exists(self.output()["ephys.bin"].path):
            arrays['ephys']['data'].tofile(self.output()["ephys.bin"].path)
                
        if not os.path.exists(self.output()['audio.wav'].path):
            with soundfile.SoundFile(
                            self.output()["audio.wav"].path,
                            "w+",
                            samplerate=arrays["mic"]["sampling_rate"],
                            format="RF64",
                            subtype="FLOAT",
                            channels=arrays["mic"]["data"].shape[1]) as f:
                arrays["mic"]["data"] = np.array(arrays["mic"]["data"])
                arrays["mic"]["data"] -= np.median(arrays["mic"]["data"], axis=0)
                for group_start, group_end in session_config["filtering"]["mic.wav"]:
                    arrays["mic"]["data"][:, group_start:group_end] /= np.max(
                        np.abs(arrays["mic"]["data"][:, group_start:group_end]))
                f.write(arrays["mic"]["data"])
        
        if not os.path.exists(self.output()['motion.bin'].path):
            if "motion" in arrays and arrays["motion"]["data"] is not None:
                np.save(self.output()['motion.bin'].path,arrays['motion']['data'])
                # previously it was a wav file
                # with soundfile.SoundFile(
                #         self.output()["motion.bin"].path,
                #         "w+",
                #         samplerate=arrays["motion"]["sampling_rate"],
                #         format="RF64",
                #         subtype="FLOAT",
                #         channels=arrays["motion"]["data"].shape[1]) as f:
                #     f.write(arrays["motion"]["data"])

        # Finally, write the channel name mappings to the configuration file
        # This is kind of an extra thing; not sure if I should put it in the
        # task outputs / set it as a new task, or what?

        # FIXME: The NixIO seems to change the order of the signals...
        # The order in the .rhd file is fine, why is it when saved to/loaded from nix
        # the order gets messed up? This can mess up any tasks that rely
        # on plain integer indexing (like filtering) so need to investigate this
        config_out = self.output()["config_original_channels"].data()
        config_out["bin_to_original_channels"] = {
            "{}.wav".format(k): v["channel_names"] for k, v in arrays.items()
        }
        
        save_yaml(self.output()["config_original_channels"].path, config_out)


class GenerateLFP(luigi.Task):
    session_dir = SessionParameter()
    def requires(self):
        return Rhd2Binary(self.session_dir)
    def output(self):
        sess_name = os.path.basename(self.session_dir)
        raw_dir = os.path.join(self.session_dir, "raw")
        lfp_path = os.path.join(raw_dir,'%s_LFP.npy'%sess_name)
        return luigi.LocalTarget(lfp_path)
    def run(self):
        from scipy.signal import butter, filtfilt,decimate
        in_data = np.fromfile(self.input()['ephys.bin'].path,dtype=np.uint16)
        in_data = in_data.reshape((in_data.shape[0]//64,64))
        hp_freq = 40
        lp_freq = 300
        filt_order = 4
        FS = 30000
        Fn = FS/2.
        LFP_freq = 1000
        b,a = butter(filt_order,(hp_freq/Fn,lp_freq/Fn),btype='bandpass')
        filtered_data = decimate(filtfilt(b=b,a=a,x=in_data,axis=0),int(FS//LFP_freq),axis=0)
        np.save(self.output().path,filtered_data.astype(np.uint16))



class Rhd2Nwb(luigi.Task):
    """Converts an RHD file to a NWB file, and clean up the RHD file

    Inputs
    ------
    session_dir : str
        Path to session

    Outputs a nix file as raw/{session_name}.nix
    """

    session_dir = SessionParameter()

    def requires(self):
        return {
            "files": DownloadSession(self.session_dir),
            "config": SessionConfig(self.session_dir),
        }

    def output(self):
        raw_dir = os.path.join(self.session_dir, "raw")
        if not os.path.exists(raw_dir):
            os.makedirs(raw_dir)

        nwb_filename = "{}.nwb".format(session_basename(self.session_dir))

        # Temporarily move nwb file from segmentation dir to raw dir
        if os.path.exists(os.path.join(self.session_dir, "segmentation", nwb_filename)):
            import shutil
            shutil.move(os.path.join(self.session_dir, "segmentation",
                        nwb_filename), os.path.join(raw_dir, nwb_filename))

        return luigi.LocalTarget(os.path.join(raw_dir, nwb_filename))

    def run(self):
        rhd_file = self.input()["files"]["rhd"].path
        session_config = self.input()["config"].data()

        # Delete the digital input form the rhd file if from Seewiesen.
        # These are not used for stim id but to sync with bat logger
        # TODO: this is not a great place for this in case we want to keep
        # the digital codes at some point? and seewiesen also doesnt apply to the owncloud data either...
        if (session_config["download_type"] == "seewiesen"):
            print("Seewiesen Data: digital events in rhd will be deleted")
            storeDIG = False
        else:
            storeDIG = True

        rhd = IntanIO(rhd_file)
        blk = rhd.read_block()
        # the end of rhd files is the timestamp
        rhd_split = os.path.basename(rhd_file).split('.')[0].split('_')
        t_start = datetime.datetime.strptime(
            "%s%s" % (rhd_split[-2], rhd_split[-1]), '%y%m%d%H%M%S')

        nwb = NWBIO(self.output().path, mode='w')
        nwb.write_block(blk)
        nwb.close()

        # Remove the RHD file now that we have a NIX file
        # os.remove(inputs["rhd"].path)


class Nwb2Wav(luigi.Task):
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

    session_dir = SessionParameter()

    def requires(self):
        return {
            "nix": Rhd2Nix(self.session_dir),  # TODO NWB
            "config": SessionConfig(self.session_dir, ["wavmap", "filtering"])
        }

    def output(self):
        raw_dir = os.path.join(self.session_dir, "raw")
        mic_path = os.path.join(raw_dir, "mic.wav")
        elec_path = os.path.join(raw_dir, "ephys.wav")
        motion_path = os.path.join(raw_dir, "motion.wav")

        site_dirs, _, _ = get_session_sites(self.session_dir)
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
        session_config = self.input()["config"].data()

        arrays = get_wavfile_data_from_nix(
            self.input()["nix"].path, session_config["wavmap"])

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
            for group_start, group_end in session_config["filtering"]["mic.wav"]:
                arrays["mic"]["data"][:, group_start:group_end] /= np.max(
                    np.abs(arrays["mic"]["data"][:, group_start:group_end]))
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
    site_dir = SiteParameter()

    def requires(self):
        return SiteConfig(site_dir=self.site_dir, required_keys=["wavmap", "filtering"])

    def run(self):
        site_config = self.input().data()
        for session in site_config["sessions"]:
            session_dir = os.path.join(
                os.path.dirname(os.path.dirname(self.site_dir)
                                ), "sessions", session["id"]
            )
            yield Nix2Wav(session_dir=session_dir)
