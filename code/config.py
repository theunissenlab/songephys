import os


#############
### Paths ###
#############

PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
if not PROJECT_ROOT:
    raise Exception("PROJECT_ROOT env variable not set")
DATADIR = os.path.join(PROJECT_ROOT, "data")
TMPDIR = os.path.join(PROJECT_ROOT, "data", "tmp")
GDRIVE_CREDENTIALS_DIR = os.path.join(PROJECT_ROOT, "tlab_gdrive_credentials")

#################
### filtering ###
#################

USE_TENSORFLOW = False
HIGHPASS_CUTOFF = 300.0
NOISE_WINDOW = 0.1
NOISE_CORR_THRESHOLD = 0.005
FILTER_BLOCK = 600
N_LAGS = 9

#################################
### Audio segmentation params ###
#################################

# TODO: eliminate song padding parameters
PAD_SONG_BEFORE = .5   # Time to pad detected song segments (in seconds)
PAD_SONG_AFTER = .5
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
# TODO: replace this with time pre-spike and time-post spike (allow for asymmetric spike snippets)
SPIKE_WIDTH = 0.0006    # in seconds
MIN_SPIKE_THRESHOLD_SPACING = 8

# Define electrode numbers in order
# - for our 16 channel recordings this is range(8, 24)
# - for our 32 channel recordings this is range(0, 32)
ELECTRODES = range(0, 32)

#################
### web views ###
#################

DEBUG = True
