import os


#############
### Paths ###
#############


PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
if not PROJECT_ROOT:
    raise Exception("PROJECT_ROOT env variable not set")
DATADIR = os.path.join(PROJECT_ROOT, "Data")


########################
### File downloading ###
########################
GDRIVE_CREDENTIALS_DIR = os.path.join(PROJECT_ROOT, "tlab_gdrive_credentials")
OWNCLOUD_URL = None
OWNCLOUD_URL = "https://owncloud.gwdg.de"
OWNCLOUD_CREDENTIALS_FILE = os.path.join(PROJECT_ROOT, "owncloud.yaml")

# Raw file locations (can be used instead of downloading the files) used in
# songephys.code.format.download.LocalSymlinker
# Fill in with paths to local locations of files, with {bird} as a wildcard for the subject folder name
LOCAL_RAW_FILES = {
    "rhd_drive": os.path.join(DATADIR, "{experiment_name}"),
    "pyoperant_drive": os.path.join(DATADIR, "{experiment_name}"),
    "stimuli_drive": os.path.join(DATADIR, "{experiment_name}", "Stimuli"),
}

#################
### filtering ###
#################

HIGHPASS_CUTOFF = 300.0

#################################
### Audio segmentation params ###
#################################

# TODO: eliminate song padding parameters
PAD_SONG_BEFORE = .5   # Time to pad detected song segments (in seconds)
AUTODETECT_FUSE_DURATION = PAD_SONG_BEFORE
AUTODETECT_SHORTEST_DURATION = 0.5
AUTODETECT_IGNORE_DURATION = 0.05
MIN_SEGMENT_LENGTH = 1.0
MAX_SEGMENT_LENGTH = 10.0
VOCALIZATION_AMPLITUDE_THRESHOLD = 1.5

############################
### spike sorting params ###
############################

SPIKE_THRESHOLD_STD = 4
Z_THRESHOLD = 4.0       # number of standard deviations to threshold initial spike detection
Z_SPIKE_LIMIT = 30.0    # upper bound of spikes to detect (because larger are probably noise...)
# TODO: This is being replaced by the variable spike_sorting.seconds_before_peak and spike_sorting.seconds_after_peak
# in the site_config.yaml files
SPIKE_WIDTH = 0.0006    # in seconds
MIN_SPIKE_THRESHOLD_SPACING = 0.0003

# Define electrode numbers in order
# - for our 16 channel recordings this is range(8, 24)
# - for our 32 channel recordings this is range(0, 32)
ELECTRODES = range(0, 32)
N_ELECTRODES = 32

#################
### web views ###
#################

DEBUG = True
