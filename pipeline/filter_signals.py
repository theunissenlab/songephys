from __future__ import print_function

try:
    import cPickle as pickle
except ImportError:
    import pickle

import os
import time

import luigi

from detect_sound_segments import DetectSoundSegments
# from preproephys import high_low_filter
from filtering.filters import filter_segment, detect_noisy_epochs, convert_to_epoch_obj

from utils import session_basename


def estimate_time_remaining(started_at, fraction_complete):
    time_elapsed = time.time() - started_at
    if fraction_complete:
        est_total_time = time_elapsed / fraction_complete
    else:
        est_total_time = 1000000000.0
    return time_elapsed, est_total_time - time_elapsed


class FilterElectrodeSignals(luigi.Task):

    session_dir = luigi.Parameter()

    def requires(self):
        return DetectSoundSegments(self.session_dir)

    def output(self):
        segmentation_dir = os.path.join(self.session_dir, "segmentation")
        output_filename = "{}-filtered.pkl".format(session_basename(self.session_dir))
        return luigi.LocalTarget(os.path.join(segmentation_dir, output_filename))

    def run(self):
        with open(self.input()[0].path, "rb") as inputfile:
            block = pickle.load(inputfile)
        block.create_relationship(force=True, append=True, recursive=True)

        start = time.time()        

        for i, seg in enumerate(block.segments):
            elapsed, remaining = estimate_time_remaining(start, float(i) / len(block.segments))
            print("Filtering Segment {}/{} in {:.0f} min... Est. remaining: {:.0f} min".format(i, len(block.segments), elapsed / 60.0, remaining / 60.0))
            clean_signal, predicted_signal = filter_segment(seg)
            noise_epochs = detect_noisy_epochs(clean_signal, predicted_signal)
            epoch = convert_to_epoch_obj(noise_epochs)
            epoch.segment = seg
            seg.epochs.append(epoch)
            seg.create_relationship(force=True)

        with open(self.output().path, "wb") as output_file:
            pickle.dump(block, output_file, protocol=4)
