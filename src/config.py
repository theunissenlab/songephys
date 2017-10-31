import os


#############
### Paths ###
#############

PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
if not PROJECT_ROOT:
    raise Exception("PROJECT_ROOT env variable not set")
DATADIR = os.path.join(PROJECT_ROOT, "data")


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

STIM_DIR = "TutFamBos"
PAD_SONG_BEFORE = 1.0   # Time to pad detected song segments (in seconds)
PAD_SONG_AFTER = 1.0 
AUTODETECT_FUSE_DURATION = 0.2
AUTODETECT_SHORTEST_DURATION = 0.6
MIN_SEGMENT_LENGTH = 1.0
MAX_SEGMENT_LENGTH = 6.0


############################
### spike sorting params ###
############################

Z_THRESHOLD = 2.0       # number of standard deviations to threshold initial spike detection
WF_START = 10           # start index to slice waveforms at
WF_STOP = 50            # stop index to slice waveforms at
WF_START = 8  # these are values for the 25kHz sampled data
WF_STOP = 40
WF_SIZE = WF_STOP - WF_START
WF_SLICE = slice(WF_START, WF_STOP)


#################
### web views ###
#################

DEBUG = True

