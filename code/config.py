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

####################
### model params ###
####################

# TODO remove? now unnecessary
SONG_CLASSIFIER_PARAMS = os.path.join(
    DATADIR,
    "params",
    "songvnoise_classifier_noz_big_nopca.pkl"
)

#################################
### Audio segmentation params ###
#################################

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
WF_START = 10           # start index to slice waveforms at
WF_STOP = 50            # stop index to slice waveforms at
# WF_START = 8  # these are values for the 25kHz sampled data
# WF_STOP = 40
WF_SIZE = WF_STOP - WF_START
WF_SLICE = slice(WF_START, WF_STOP)
ELEC_SAMPLE_RATE = 30000.0
SPIKE_WIDTH = 0.0008    # in seconds
# ELEC_SAMPLE_RATE = 25000.0
MIN_SPIKE_THRESHOLD_SPACING = 3
ELECTRODES = range(8, 24)


#################
### web views ###
#################

DEBUG = True

