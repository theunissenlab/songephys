import os
import quantities as pq
from zeebeez.tdt2neo import stim
from neo.core import Epoch
from neo.io import RHDIO, NixIO
from neo.io.tools import convert_digital_to_metadata
from neosound import sound_manager
from neosound.sound_store import HDF5Store

import numpy as np

experiment_dir = "/srv/media/datasets/ephys/chronicv2/LbY6074"
experiment_file = os.path.join(experiment_dir, "RHD", "LbY6074__161215_145633.rhd")
stimulus_file = os.path.join(experiment_dir, "pyoperant", "")
stimulus_dir = os.path.join(experiment_dir, "stimuli", "")


def rhd_to_nix(experiment_dir, rhd_file, stimulus_file=None):
       
    # Check to see if RHD file exists.
    experiment_file = os.path.join(experiment_dir, "RHD", rhd_file)
    if not(os.path.isfile(experiment_file)):
        print 'Error:', experiment_file, 'does not exist'
        return 1
    
    # Check to see if NIX file exists.
    experiment_file = os.path.join(experiment_dir, "RHD", rhd_file)
    h5_filename = os.path.join(experiment_dir, "NIX",
                               os.path.basename(experiment_file).replace(".rhd", ".nix"))
    if os.path.isfile(h5_filename):
        print 'Error:', h5_filename, 'exists'
        return 1
    
    # Read RHD file.   
    rhd_importer = RHDIO(experiment_file)
    block = rhd_importer.read_block()
    
    
    # stimulus_h5_filename = os.path.join(experiment_dir,
    #                                    os.path.basename(
    #                                        experiment_file).replace(".rhd", "_stimuli.h5"))
    # if stimulus_file is not None:
    #     csv_reader = stim.StimCSVReader()
    #
    #     # Read all of the metadata and trial data
    #     csv_reader.read(stimulus_file, root_dir=stimulus_dir)
    #     csv_reader.write(stimulus_h5_filename)
    #
    #     write_playback_stim_to_nix(block, stimulus_h5_filename)
    
    # Write NIX File  
    h5_exporter = NixIO(h5_filename)
    h5_exporter.write_block(block)
    # print 'Block path is', block.hdf5_path
    h5_exporter.close()
    return 0



def write_playback_stim_to_nix(block, stim_file_name):
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
    codes = convert_digital_to_metadata(on_times)

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
        epoch = Epoch(times=start,
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


def read_audio_from_nix(experiment_dir, nix_filename):
    h5_filename = os.path.join(experiment_dir, 'NIX', nix_filename)
    
    if not(os.path.isfile(h5_filename)):
        print 'Error:', nix_filename, 'is missing'
        return None
    
    h5_exporter = NixIO(h5_filename)
    block = h5_exporter.read_block()
    sound_recording = block.segments[0].analogsignals[1]
    norm_sound = sound_recording / np.max(sound_recording)
    h5_exporter.close()
    return norm_sound


def add_vocal_periods_to_nix(experiment_dir, nix_filename,
                             vocal_periods_dict, sampling_rate=25000.0):
    """
    Adds annotated vocal periods to the nix file as an epoch array. Mutates the passed file, with
    no return value.

    Parameters
    ----------
    experiment_dir: string.
        Root of the directory where all of the files for a particular bird are stored.
    nix_filename: string.
        The actual filename of the particular experiment session. Should be of the form:
        <bird id><timestamp>_neo.h5
    vocal_periods_dict: dictionary.
        A dictionary with keys "student" and "tutor" containing lists of intervals [(int, int)].
    """
    full_filename = os.path.join(experiment_dir, 'NIX', nix_filename)
    
    if not(os.path.isfile(full_filename)):
        print 'Error:', full_filename, 'is missing'
        return 1
    
    exporter = NixIO(full_filename)
    block = exporter.read_block()
    segment = block.segments[0]
    if segment.epochs is None:
        segment.epochs = []
    for k in ['student', 'tutor']:
        intervals = vocal_periods_dict[k]
        starts = np.array(map(lambda x: x[0], intervals))
        durations = np.array(map(lambda p: p[1] - p[0], intervals))
        labels = np.array([k]*len(starts))
        epocharr = Epoch(times=starts/sampling_rate*pq.s, durations=durations/sampling_rate*pq.s,
                         labels=labels, name=k+'_birdsong')
        segment.epochs.append(epocharr)
    segment.create_many_to_one_relationship()
    exporter.write_block(block)
    exporter.close()
    return 0
