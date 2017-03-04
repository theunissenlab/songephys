import os
import quantities as pq
from zeebeez.tdt2neo import stim, import_multielectrode
from neo.core import EpochArray
from neo.io import RHDIO, NeoHdf5IO
from neosound import sound_manager
from neosound.sound_store import HDF5Store

import numpy as np

experiment_dir = "/srv/media/datasets/ephys/chronicv2/LbY6074"
experiment_file = os.path.join(experiment_dir, "RHD", "LbY6074__161215_145633.rhd")
stimulus_file = os.path.join(experiment_dir, "pyoperant", "")
stimulus_dir = os.path.join(experiment_dir, "stimuli", "")


def rhd_to_neo(experiment_dir, rhd_file, stimulus_file=None):
    experiment_file = os.path.join(experiment_dir, "RHD", rhd_file)
    rhd_importer = RHDIO(experiment_file)
    block = rhd_importer.read_block()
    stimulus_h5_filename = os.path.join(experiment_dir,
                                        os.path.basename(
                                            experiment_file).replace(".rhd", "_stimuli.h5"))

    if stimulus_file is not None:
        csv_reader = stim.StimCSVReader()

        # Read all of the metadata and trial data
        csv_reader.read(stimulus_file, root_dir=stimulus_dir)
        csv_reader.write(stimulus_h5_filename)

        write_playback_stim_to_neo(block, stimulus_h5_filename)
    h5_filename = os.path.join(experiment_dir,
                               os.path.basename(experiment_file).replace(".rhd", "_neo.h5"))
    h5_exporter = NeoHdf5IO(h5_filename)
    h5_exporter.save(block)
    print 'Block path is', block.hdf5_path
    h5_exporter.close()


def write_playback_stim_to_neo(block, stim_file_name):
    # Write sounds to a neosound database
    sm = sound_manager.SoundManager(HDF5Store, stim_file_name)

    # The block only has one segment
    segment = block.segments[0]

    # The list of epocharrays should be empty but force it here
    segment.epocharrays = []

    # Get digital signal eventarray and process it
    digital_signals = segment.eventarrays[0]
    sampling_rate = digital_signals.annotations["sampling_rate"]
    on_times = np.round(digital_signals.times * sampling_rate).astype("int")
    codes = import_multielectrode.convert_digital_to_metadata(on_times)

    # Loop through each stimulus stored in metadata and add an EpochArray instance
    for ii, (start, metadata) in enumerate(codes):
        # must be a unit quantity to work in the hdf5 file
        start = (float(start) / sampling_rate).rescale(pq.s)
        trial = metadata["trial"]
        # One experiment forgot to add metadata, so trial number is always 0.
        # Use count instead, converted to 1-index.
        if trial == 0:
            trial = ii + 1

        # Find the stimulus in the database using the trial number
        print trial
        stim_id = sm.database.filter_ids(index=trial)[0]
        label = str(stim_id)
        duration = float(sm.database.get_annotations(stim_id)["duration"]) * pq.s
        # Warning: don't make the arguments into list -
        # otherwise the writing into hdf5 files will fail
        epoch = EpochArray(times=start,
                           labels=label,
                           durations=duration,
                           stim_id=stim_id,
                           **metadata)

        segment.epocharrays.append(epoch)
        print 'Epoch:', ii, 'Trial:', trial, 'Stim:', stim_id

    print 'Found', len(segment.epocharrays), 'trials'

    # This next line make the parent of all the children of the segment
    # (in this case the epocharrays) be the segment
    segment.create_many_to_one_relationship()


def read_audio_from_neo(neo_filename):
    h5_exporter = NeoHdf5IO(neo_filename)
    block = h5_exporter.get("/Block_0")
    sound_recording = block.segments[0].analogsignalarrays[1]
    norm_sound = sound_recording / np.max(sound_recording)
    return norm_sound
