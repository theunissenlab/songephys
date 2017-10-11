import os
import sys
import numpy as np
import quantities as pq
try:
   import cPickle as pickle
except:
   import pickle

from neo.core import Segment
from neo.io import NixIO
from neosound import sound_manager
from neosound.sound_store import HDF5Store
from zeebeez.tdt2neo import stim, import_multielectrode


import autodetect
import config


def save_stimulus_database(pyoperant_file, stim_dir, db_name):
    csv_reader = stim.StimCSVReader()

    # Read all of the metadata and trial data
    csv_reader.read(pyoperant_file, root_dir=stim_dir)

    # Write sounds to a neosound database
    db_path = os.path.join(stim_dir, db_name)
    csv_reader.write(db_path)

    return sound_manager.SoundManager(HDF5Store, db_path)


def generate_stim_dicts(digital_signals, sound_db):
    """Generate segment data from digital signal data and sound (stim) database

    Args:
        digital_signals (neo.core.Events)
        sound_db (neosound.sound_store.HDF5Store)
    
    Returns:
        list of dicts: List of segment information
            Each element is a dict defining segment start and end,
            and metadata to include in Segment annotations
            See docstring for audio_segment.segment() for required fields
    """
    sample_rate = digital_signals.annotations["sampling_rate"]
    on_times = np.round(digital_signals.times * sample_rate).astype("int")

    digital_codes = import_multielectrode.convert_digital_to_metadata(on_times)

    stim_dicts = []

    for trial_id, (start_time, metadata) in enumerate(digital_codes):
        start_time = float(start_time) / sample_rate
        trial = metadata.get("trial", trial_id + 1)

        stim_id = sound_db.filter_ids(index=trial)[0]
        duration = sound_db.get_annotations(stim_id)["duration"]

        metadata = sound_db.get_metadata(stim_id)
        annotations = sound_db.get_annotations(stim_id)

        stim_dicts.append({
            "name": "playback_{}".format(annotations["emitter"]),
            "description": "Created by segmentation.audio_segment.generate_stim_dicts",
            "sound_type": "playback",
            "vocalization_type": annotations.get("callid", "UV"),
            "vocalizer": annotations.get("source"),  # need a helper to make this more specific
            "playback_file": annotations.get("original_filename"),
            "stim_id": stim_id,
            "start_time": start_time.rescale(pq.s),
            "duration": duration * pq.s
        })

    return stim_dicts


def generate_vocal_periods_dicts(audio, sampling_rate):
    """Generate segment data by detecting periods of acoustic activity

    Does not include any classification/categorization step
    """
    vocal_periods, _ = autodetect.find_vocal_periods(
        audio,
        fs=sampling_rate,
        fuse_duration=config.AUTODETECT_FUSE_DURATION,
        shortest_duration=config.AUTODETECT_SHORTEST_DURATION
    )

    seg_dicts = []
    for start_time, stop_time in vocal_periods:
        start_time = float(start_time) / sampling_rate
        stop_time = float(stop_time) / sampling_rate
        seg_dicts.append({
            "name": "live_unknown",
            "description": "Created by segmentation.audio_segment.generate_vocal_periods_dicts",
            "sound_type": "live",
            "vocalization_type": "unknown",
            "vocalizer": "unknown",
            "playback_file": None,
            "start_time": start_time * pq.s,
            "duration": (stop_time - start_time) * pq.s
        })

    return seg_dicts


def generate_live_vocalization_dicts(df):
    """Generate segment data by detecting periods of CLASSIFIED acoustic activity

    Args:
        -block (RHD)
        -df -a pandas dataframe which is produced upstream, mostly by autodetect.populate_dataframe
            -contains:
                -vocalization start and stop times : df.columns[-3]  (as pq.Quantity tuple)
                -vocalizer  :                        df.columns[-2]
                -vocal_category  :                   df.columns[-1]
    Returns:
        list of dicts: List of segment information
            Each element is a dict defining segment start and end,
            and metadata to include in Segment annotations
            See docstring for audio_segment.segment() for required fields
    """
    vocal_periods = df.iloc[:, -3].values
    vocalizer = df.iloc[:, -2].values
    vocal_category = df.iloc[:, -1].values

    seg_dicts = []
    for (start_time, stop_time), emitter, category in zip(vocal_periods, vocalizer, vocal_category):
        seg_dicts.append({
            "name": "live_{}".format(emitter),
            "description": "Created by segmentation.audio_segment.generate_live_vocalization_dicts",
            "sound_type": "live",
            "vocalization_type": category,
            "vocalizer": emitter,
            "playback_file": None,
            "start_time": start_time,
            "duration": stop_time - start_time
        })
            
    return seg_dicts


def merge_segments(segments_1, segments_2, dedupe=True):
    """Combine segments with optional de-duplication"""
    if dedupe:
        # give segments_1 priority
        combined_segments = segments_1[:]
        seg_1_starts = np.array([s["start_time"] for s in segments_1])
        seg_1_durations = np.array([s["duration"] for s in segments_1])

        for seg in segments_2:
            # there are three conditions for overlap:
            # Case 1: my start time occurs between your start and end times
            start_diffs = seg["start_time"].item() - seg_1_starts
            start_overlaps = (start_diffs > 0) & (start_diffs < seg_1_durations)

            # Case 2: my end time occurs between your start and end times
            end_diffs = seg["start_time"].item() + seg["duration"].item() - seg_1_starts
            end_overlaps = (end_diffs > 0) & (end_diffs < seg_1_durations)

            # Case 3: my start time occurs before your start time and my end time occurs after your end time
            encompasses = (
                    (seg["start_time"].item() < seg_1_starts) &
                    (seg["start_time"].item() + seg["duration"].item() - seg_1_starts > seg_1_durations)
            )

            if np.any(start_overlaps + end_overlaps + encompasses):
                continue

            combined_segments.append(seg)
    else:
        combined_segments = segments_1 + segments_2

    return sorted(combined_segments, key=lambda s: s["start_time"])


def segment(block, segment_dicts):
    """Segment a block in-place

    Args:
        block (neo.core.Block):
            The original neo block to be segmented
        segment_dicts (list of dict): List of segment information
            Each element is a dict defining segment start and end,
            and metadata to include in Segment annotations
            Each dict requires the following keys:
                name - {sound_type}_{vocalizer}_index
                description - Created by <file.py> + hashcode of file
                file_origin - original rhd file  (this could come from the block itself already though?)
                sound_type - playback, live, or silence
                vocalization_type - So, Te, DC, UV (Song, Tet, Distance Call, Unknown Vocalization)
                vocalizer - self, tutor, mom, sister, con_male, con_female
                playback_file - stimulus id?
                start_time (pq.Quantity) - start time of segment
                duration (pq.Quantity) - duration of segment
    
    Returns:
        neo.core.Block: The original block, modified with new segments
    """
    main_segment = block.segments[0]

    # Remove the link to main analogsignal from the Amplifier and Board ADC channel indexes
    for ci in block.channel_indexes:
        if ci.name == 'Amplifier':
            ci.analogsignals = []       
        if ci.name == 'ADC':
            ci.analogsignals = []

    if len(block.segments) > 1:
        raise Exception("This block already has more than one segment. What are you doing?")

    for trial_id, segment_dict in enumerate(segment_dicts):
        start_time = segment_dict["start_time"]
        duration = segment_dict["duration"]

        new_segment = Segment(
                name="{}_{}".format(segment_dict["name"], trial_id),  # must be unique!
                description=segment_dict["description"],
                file_origin=main_segment.file_origin,
                file_datetime=main_segment.file_datetime,
                rec_datetime=main_segment.rec_datetime,
                index=trial_id,
                times=start_time,
                sound_type=segment_dict.get("sound_type"),
                vocalization_type=segment_dict.get("vocalization_type"),
                vocalizer=segment_dict.get("vocalizer"),
                playback_file=segment_dict.get("playback_file")
        )

        for signal in main_segment.analogsignals:
            sliced_signal = signal.time_slice(
                start_time - config.PAD_SONG_BEFORE * pq.s,
                start_time + segment_dict["duration"] + config.PAD_SONG_AFTER * pq.s
            )
            for channel_index in block.channel_indexes:
                # TODO: preserve the channel number (e.g. 8 instead of 0)
                if channel_index.name in signal.name:  # Amplifier or Board ADC
                    channel_index.analogsignals.append(sliced_signal)

            new_segment.analogsignals.append(sliced_signal)

        new_segment.create_many_to_one_relationship()
        block.segments.append(new_segment)

    del block.segments[0]
    block.create_many_to_one_relationship(force=True, recursive=True)
    block.create_many_to_many_relationship()
    block.create_relationship(force=True, recursive=True, append=True)

    return block

