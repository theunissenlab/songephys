from __future__ import print_function

import os
import time
import numpy as np

import luigi

import block_io
from detect_sound_segments import SplitUp
from filtering.filters import (
        convert_to_epoch_obj,
        detect_noisy_epochs,
        filter_segment,
)

from utils import session_basename


def estimate_time_remaining(started_at, fraction_complete):
    time_elapsed = time.time() - started_at
    if fraction_complete:
        est_total_time = time_elapsed / fraction_complete
    else:
        est_total_time = np.inf
    return time_elapsed, est_total_time - time_elapsed


class FilterElectrodeSignals(luigi.Task):

    session_dir = luigi.Parameter()

    def requires(self):
        return SplitUp(self.session_dir)

    def output(self):
        segmentation_dir = os.path.join(self.session_dir, "segmentation")
        output_filename = "{}-filtered.pkl".format(
                session_basename(self.session_dir)
        )
        return luigi.LocalTarget(
                os.path.join(segmentation_dir, output_filename)
        )

    def run(self):
        block = block_io.load_block(self.input().path)

        start = time.time()
        for i, seg in enumerate(block.segments):
            elapsed, remaining = estimate_time_remaining(
                    start, float(i) / len(block.segments)
            )
            print(
                "Filtering Segment {}/{} in {:.0f} min... "
                "Est. remaining: {:.0f} min".format(
                    i, len(block.segments), elapsed / 60.0, remaining / 60.0
                )
            )
            clean_signal, predicted_signal = filter_segment(seg)

            # Identify noise epochs based on the residuals
            noise_epochs = detect_noisy_epochs(clean_signal, predicted_signal)
            # TODO: save noise epochs in a separate file
            epoch = convert_to_epoch_obj(noise_epochs)
            epoch.segment = seg
            seg.epochs.append(epoch)
            seg.create_relationship(force=True)

        block_io.save_block(self.output().path, block)
