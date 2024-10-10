import os
from .format.io import load_yaml, save_yaml

#############
### Paths ###
#############

def generate_config_file(experiment_name, out_path=None, data_dir = os.environ.get("DATA_DIR")):
    """ Generates a config file in the current directory for use with the songephys pipeline

    Keyword arguemnts:
    experiment_name -- Name of the experiment, also will be appended to Data Dir for default RHD locations
    out_path -- location to save config to (defaults to SONGEPHYS_DIR/experiment_name/)
    data_dir -- Root data folder (default checks DATA_DIR environment variable)
    """
    project_cfg = dict()
    if not experiment_name:
        raise Exception("Must provide experiment name")
    if not data_dir:
        raise Exception("DATA_DIR env variable not set")
    if out_path is None:
        songephys_dir = os.environ.get("SONGEPHYS_DIR")
        if not songephys_dir:
            raise Exception("SONGEPHYS_DIR env variable not set")
        out_path = os.path.join(os.environ.get("SONGEPHYS_DIR"), experiment_name)
    expeiriment_cfg = dict()
    expeiriment_cfg["EXPERIMENT_NAME"] = experiment_name
    expeiriment_cfg["DATADIR"] = data_dir


    ########################
    ### File downloading ###
    ########################
    expeiriment_cfg["DOWNLOADS"] = dict()
    download_cfg = expeiriment_cfg["DOWNLOADS"] 
    download_cfg["GDRIVE_CREDENTIALS_DIR"] = os.path.join(expeiriment_cfg["DATADIR"], "tlab_gdrive_credentials")
    download_cfg["OWNCLOUD_URL"] = "https://owncloud.gwdg.de"
    download_cfg["OWNCLOUD_CREDENTIALS_FILE"] = os.path.join(expeiriment_cfg["DATADIR"], "owncloud.yaml")

    # Raw file locations (can be used instead of downloading the files) used in
    # songephys.code.format.download.LocalSymlinker
    # Fill in with paths to local locations of files, with {bird} as a wildcard for the subject folder name
    expeiriment_cfg["LOCAL_RAW_FILES"] = {
        "rhd_drive": os.path.join(expeiriment_cfg["DATADIR"], f"{experiment_name}"),
        "pyoperant_drive": os.path.join(expeiriment_cfg["DATADIR"], f"{experiment_name}"),
        "stimuli_drive": os.path.join(expeiriment_cfg["DATADIR"], f"{experiment_name}", "Stimuli"),
    }

    #################
    ### filtering ###
    #################

    expeiriment_cfg["HIGHPASS_CUTOFF"] = 300.0

    #################################
    ### Audio segmentation params ###
    #################################

    # TODO: eliminate song padding parameters
    expeiriment_cfg['AUDIO_SEGMENTATION'] = dict()
    audio_cfg = expeiriment_cfg['AUDIO_SEGMENTATION']
    audio_cfg["PAD_SONG_BEFORE"] = .5   # Time to pad detected song segments (in seconds)
    audio_cfg["AUTODETECT_FUSE_DURATION"] = audio_cfg["PAD_SONG_BEFORE"]
    audio_cfg["AUTODETECT_SHORTEST_DURATION"] = 0.5
    audio_cfg["AUTODETECT_IGNORE_DURATION"] = 0.05
    audio_cfg["MIN_SEGMENT_LENGTH"] = 1.0
    audio_cfg["MAX_SEGMENT_LENGTH"] = 10.0
    audio_cfg["VOCALIZATION_AMPLITUDE_THRESHOLD"] = 1.5

    ############################
    ### spike sorting params ###
    ############################
    expeiriment_cfg['SPIKE_SORTING'] = dict()
    spike_cfg = expeiriment_cfg['SPIKE_SORTING']
    spike_cfg["SPIKE_THRESHOLD_STD"] = 4
    spike_cfg["Z_THRESHOLD"] = 4.0       # number of standard deviations to threshold initial spike detection
    spike_cfg["Z_SPIKE_LIMIT"] = 30.0    # upper bound of spikes to detect (because larger are probably noise...)
    # TODO: This is being replaced by the variable spike_sorting.seconds_before_peak and spike_sorting.seconds_after_peak
    # in the site_config.yaml files
    spike_cfg["SPIKE_WIDTH"] = 0.0006    # in seconds
    spike_cfg["MIN_SPIKE_THRESHOLD_SPACING"] = 0.0003

    # Define electrode numbers in order
    # - for our 16 channel recordings this is range(8, 24)
    # - for our 32 channel recordings this is range(0, 32)
    expeiriment_cfg['EPHYS'] = dict()
    ephys_cfg = expeiriment_cfg['EPHYS']
    ephys_cfg["ELECTRODE_START"] = 0
    ephys_cfg["ELECTRODE_N"] = 32
    ephys_cfg["ML"] = 0 # Medial-Lateral
    ephys_cfg["AP"] = 0 # Anterior-Posterior

    #################
    ### web views ###
    #################

    expeiriment_cfg["DEBUG"] = False
    cfg_path = os.path.join(out_path,"project_config.yaml")
    print(f"Saving Config to {cfg_path}")
    save_yaml(cfg_path, expeiriment_cfg)

def get_config(experiment_name=None):
    if experiment_name is None:
        # loads config from current directory
        return load_yaml("project_config.yaml")
    else:
        # loads config from the experiment path
        songephys_dir = os.environ.get("SONGEPHYS_DIR")
        if not songephys_dir:
            raise Exception("SONGEPHYS_DIR env variable not set")
        cfg_path = os.path.join(os.environ.get("SONGEPHYS_DIR"), experiment_name,"project_config.yaml")

        if not os.path.exists(cfg_path):
            raise  Exception("Project config does not exist for %s\n Please run `song create-config %s"%(experiment_name,experiment_name))
        return load_yaml(cfg_path)