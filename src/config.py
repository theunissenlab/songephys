import os


### Paths ###

PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
if not PROJECT_ROOT:
    raise Exception("PROJECT_ROOT env variable not set")
DATADIR = os.path.join(PROJECT_ROOT, "data")

### model params ###

SONG_CLASSIFIER_PARAMS = os.path.join(
    DATADIR,
    "params",
    "songvnoise_classifier_noz_big_nopca.pkl"
)

### Audio segmentation params ###
PAD_SONG_BEFORE = 0.2   # Time to pad detected song segments (in seconds)
PAD_SONG_AFTER = 0.2


### spike sorting params ###

Z_THRESHOLD = 2.0       # number of standard deviations to threshold initial spike detection
WF_START = 10           # start index to slice waveforms at
WF_STOP = 50            # stop index to slice waveforms at
WF_SIZE = WF_STOP - WF_START
WF_SLICE = slice(WF_START, WF_STOP)

### web views ###

DEBUG = True
