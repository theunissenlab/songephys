import os
import numpy as np
import pandas as pd
import quantities as pq

from neo.core import AnalogSignal, SpikeTrain, Segment, Epoch
from .autodetect import find_vocal_periods

import config


def binary_converter(code, bits_per_char):
    num_vals = len(code) // bits_per_char
    vals = np.zeros(num_vals)
    for ii in range(num_vals):
        binvec = code[ii * bits_per_char: (ii + 1) * bits_per_char]
        vals[ii] = np.sum((2 ** np.arange(bits_per_char - 1, -1, -1)) * binvec)

    return "".join(map(chr, vals.astype(np.uint8)))


def convert_digital_to_metadata(on_times, code_length=384, bits_per_char=8):
    """Detect periods of digital codes and decode stimulus metadata

    Copied from zeebeez because of a problem sometimes where the digital code starts
    with "on" state instead of "off" state
    """
    # There is a certain bug where the pulses start in the on state
    # and the first pulse is missed.
    gaps = np.hstack([np.array(np.inf), np.diff(on_times)])
    starts = on_times[gaps > code_length]
    # If starts[0] is 0, then we probably have that bug.
    # Instead, start on the first time that we see a "off"
    if starts[0] == 0:
        starts[0] = int(np.argmax(np.diff(on_times) > 1.0))

    # TODO: automatically detect bad cases and correct them...
    # TODO: kevin, i forgot what that TODO means
    codes = list()
    for ii, start in enumerate(starts):
        metadata = dict()
        signal = np.zeros(code_length, dtype=np.int8)
        on_bits = on_times[((on_times - start) > 0) & ((on_times - start) < code_length)] - start - 1
        signal[on_bits] = 1
        code = binary_converter(signal, bits_per_char)
        metadata["code"] = code
        metadata["object"] = code[:4]
        metadata["action"] = code[4:8]
        try:
            metadata["repetition"] = int(code[8:12])
        except ValueError:
            print("Repetition is not an integer! Setting to 0.")
            metadata["repetition"] = 0
        try:
            metadata["trial"] = int(code[12:16])
        except ValueError:
            print("Trial is not an integer! Setting to 0.")
            metadata["trial"] = 0
        metadata["md5"] = code[16:]
        codes.append((start, metadata))

    return codes


def get_vocalizer_relation(annotations):
    familiarity = annotations.get("source")
    if familiarity == "tutor":
        return "tutor"
    elif annotations.get("source_sex") == "m":
        return "con_male"
    elif annotations.get("source_sex") == "f":
        return "con_female"
    else:
        return "unfamiliar"


def validate_digital_codes(digital_signals, trials_csv):
    """Validate that the extracted digital codes are consistent with data in csvs

    for now just check that the times between trials is consistent
    """
    sample_rate = digital_signals.annotations["sampling_rate"]
    on_times = np.array(np.round(digital_signals.times * sample_rate)).astype("int")
    digital_codes = convert_digital_to_metadata(on_times)

    code_diffs = np.diff([start_time / 30000.0 for (start_time, _) in digital_codes])
    csv_diffs = [t.item() for t in np.diff(trials_csv["time"])]

    # sometimes the first or last code_diff will be way larger than the other ones...
    # that means we probbaly want to ignore it or somethign

    if len(code_diffs) != len(csv_diffs):
        # raise Exception("Different number of digital codes and csv trials")
        pass

    return code_diffs[:len(csv_diffs)] / 30000.0 - csv_diffs[:len(code_diffs)]


def generate_stim_dicts(digital_signals, playbacks_db):
    """Generate segment data from digital signal data and sound (stim) database

    Args:
        digital_signals (neo.core.Events)
        playbacks_db (pandas.DataFrame): dataframe of stimuli played

    Returns:
        list of dicts: List of segment information
            Each element is a dict defining segment start and end,
            and metadata to include in Segment annotations
            See docstring for audio_segment.segment() for required fields
    """
    sample_rate = digital_signals.annotations["sampling_rate"]
    on_times = np.array(np.round(digital_signals.times * sample_rate)).astype("int")

    digital_codes = convert_digital_to_metadata(on_times)
    # TODO: if the digital codes dont match up with the csv codes... we need to correct digital_codes by removing the bad ones

    digital_codes = [codes for codes in digital_codes if codes[1]["action"] == "queu"]

    stim_dicts = []

    # Use the trial only as a validaiton
    current_trial = None
    for _, (start_time, metadata) in enumerate(digital_codes):
        if current_trial is None:
            current_trial = metadata.get("trial")
        else:
            current_trial += 1

        start_time = float(start_time) / sample_rate

        # can we VALIDATE the digital code start time with the
        # data in sounddb or the csv?

        # Query the database for the correct trial
        stim = playbacks_db[playbacks_db["trial"] == current_trial]
        if len(stim) == 0:
            # TODO (kevin): We couldn't find this trial. Raise error?
            break
        else:
            stim_id = stim.index[0]
            stim = dict(stim.iloc[0])

        vocalizer_relation = get_vocalizer_relation(stim)

        stim_dicts.append({
            "name": "playback_{}".format(stim["emitter"]),
            "description": "Created by segmentation.audio_segment.generate_stim_dicts",
            "sound_type": "playback",
            "vocalization_type": stim.get("callid", "UV"),
            "vocalizer": vocalizer_relation,
            "playback_file": stim.get("original_filename"),
            "stim_id": stim_id,
            "start_time": start_time.rescale(pq.s) - config.PAD_SONG_BEFORE * pq.s,
            "duration": (stim["duration"] + config.PAD_SONG_AFTER + config.PAD_SONG_BEFORE) * pq.s
        })

    return stim_dicts


def generate_vocal_periods_dicts(audio, sampling_rate):
    """Generate segment data by detecting periods of acoustic activity

    Does not include any classification/categorization step
    """

    vocal_periods, _ = find_vocal_periods(
        audio,
        fs=sampling_rate,
        fuse_duration=config.AUTODETECT_FUSE_DURATION,
        shortest_duration=config.AUTODETECT_SHORTEST_DURATION
    )

    total_duration = len(audio) / sampling_rate

    seg_dicts = []
    for start_time, stop_time in vocal_periods:
        start_time = float(start_time) / sampling_rate
        stop_time = float(stop_time) / sampling_rate

        if stop_time > total_duration:
            print("Warning: segment from {} to {} extends beyond {}".format(start_time, stop_time, total_duration))
            continue

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


def generate_all_segment_dicts(block, t_width=10.0):
    """Segment the whole block into segments that cover the whole thing.

    Can remerge them later but need to do this for processing for now.
    """
    main_segment = block.segments[0]

    seg_dicts = []
    start_times = np.arange(main_segment.t_start.item(), main_segment.t_stop.item(), t_width)
    for i, start_time in enumerate(start_times):
        seg_dicts.append({
            "name": "segment_{}".format(i),
            "description": "Created by segmentation.audio_segment.generate_all_segment_dicts",
            "sound_type": None,
            "vocalization_type": None,
            "vocalizer": None,
            "playback_file": None,
            "start_time": start_time * pq.s,
            "duration": t_width * pq.s
        })

    return seg_dicts


def _one_long_segment(block):
    """Return a block to its original one-segment state"""
    # merge analogsignals
    # merge spiketrains

    new_segment = Segment(
            name="all_encompassingly",
            description="the whole data",
            index=0,
            times=block.segments[0].t_start
    )

    sigs = [seg.analogsignals[0] for seg in block.segments]
    new_analogsignal = AnalogSignal(
            np.concatenate([s.as_array() for s in sigs]) * sigs[0].units,
            name="all_encompassingly_analogsignal",
            sampling_rate=sigs[0].sampling_rate,
            t_start=sigs[0].t_start,
            units=sigs[0].units,
            **sigs[0].annotations)
    new_analogsignal.segment = new_segment
    new_segment.analogsignals.append(new_analogsignal)

    epochs = [seg.epochs[0] for seg in block.segments]
    new_epoch = Epoch(
        times=np.concatenate([epoch.times for epoch in epochs]) * epochs[0].times.units,
        durations=np.concatenate([epoch.durations for epoch in epochs]) * epochs[0].durations.units,
        labels=np.concatenate([epoch.labels for epoch in epochs])
    )
    new_segment.epochs.append(new_epoch)

    spiketimes = [
        np.concatenate([
            st.times.as_array() for st in sts
        ])
        for sts in zip(*[seg.spiketrains for seg in block.segments])
    ]

    wfs = [
        np.concatenate([
            st.waveforms for st in sts
        ], axis=0)
        for sts in zip(*[seg.spiketrains for seg in block.segments])
    ]

    for unit_idx, template_st in enumerate(block.segments[0].spiketrains):
        new_spiketrain = SpikeTrain(
            spiketimes[unit_idx],
            t_start=new_analogsignal.t_start,
            t_stop=new_analogsignal.t_stop,
            name=template_st.unit.name,
            waveforms=wfs[unit_idx],
            # unit=template_st.unit,
            units=template_st.units
        )
        template_st.unit.spiketrains = [new_spiketrain]
        new_spiketrain.unit = template_st.unit
        new_spiketrain.segment = new_segment
        new_segment.spiketrains.append(new_spiketrain)

    new_segment.create_many_to_one_relationship()

    return new_segment


def one_long_segment(block):
    new_segment = _one_long_segment(block)

    block.segments = [new_segment]

    block.create_many_to_one_relationship(force=True, recursive=True)
    block.create_many_to_many_relationship()
    block.create_relationship(force=True, recursive=True, append=True)

    return block


def consolidate_segments(segments):
    """Merge segments that overlap so that there are no overlaps
    """

    segments.sort(key=lambda seg: seg["start_time"])

    new_segments = []

    current_segment = None
    for segment in segments:
        next_end = segment["start_time"] + segment["duration"]

        if current_segment is None:
            current_segment = dict(**segment)
        elif segment["start_time"] <= (current_segment["start_time"] + current_segment["duration"]):
            current_segment["duration"] = next_end - current_segment["start_time"]
        else:
            new_segments.append(current_segment)
            current_segment = dict(**segment)

    new_segments.append(current_segment)

    return new_segments


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
    # main_segment = block.segments[0]

    # Remove the link to main analogsignal from the Amplifier and Board ADC channel indexes
    for ci in block.channel_indexes:
        if ci.name == 'Amplifier':
            ci.analogsignals = []       
        if ci.name == 'ADC':
            ci.analogsignals = []

    if len(block.segments) > 1:
        # TODO (kevin): raise a warning if the block has more than one segment?
        print("warn warn warn")
        # raise Exception("This block already has more than one segment. What are you doing?")

    new_segments = []
    curr_seg = 0

    for channel_index in block.channel_indexes:
        for unit in channel_index.units:
            del unit.spiketrains[:]

    for trial_id, segment_dict in enumerate(segment_dicts):
        start_time = segment_dict["start_time"]
        end_time = start_time + segment_dict["duration"]

        old_segment = block.segments[curr_seg]

        ends_after =  (
            end_time >= block.segments[curr_seg].t_stop or
            np.isclose(end_time.item(), block.segments[curr_seg].t_stop.item())
        )

        while ends_after:
            curr_seg += 1
            ends_after = (curr_seg < len(block.segments)) and (
                end_time >= block.segments[curr_seg].t_stop or
                np.isclose(end_time.item(), block.segments[curr_seg].t_stop.item())
            )

        new_segment = Segment(
            name="{}_{}".format(segment_dict["name"], trial_id),  # must be unique!
            description=segment_dict["description"],
            file_origin=old_segment.file_origin,
            file_datetime=old_segment.file_datetime,
            rec_datetime=old_segment.rec_datetime,
            index=trial_id,
            times=start_time,
            sound_type=segment_dict.get("sound_type"),
            vocalization_type=segment_dict.get("vocalization_type"),
            vocalizer=segment_dict.get("vocalizer"),
            playback_file=segment_dict.get("playback_file")
        )

        for signal in old_segment.analogsignals:
            stop_time = end_time if end_time <= signal.t_stop else None
            sliced_signal = signal.time_slice(start_time, stop_time)
            for channel_index in block.channel_indexes:
                # TODO: preserve the channel number (e.g. 8 instead of 0)
                if channel_index.name in signal.name:  # Amplifier or Board ADC
                    channel_index.analogsignals.append(sliced_signal)

            new_segment.analogsignals.append(sliced_signal)
            sliced_signal.segment = new_segment
            
        for spiketrain in old_segment.spiketrains:
            stop_time = end_time if end_time <= signal.t_stop else None
            new_spiketrain = spiketrain.time_slice(start_time, stop_time)
            new_segment.spiketrains.append(new_spiketrain)
            spiketrain.unit.spiketrains.append(new_spiketrain)

        for epoch in old_segment.epochs:
            stop_time = end_time if end_time <= signal.t_stop else None
            new_epoch = epoch.time_slice(start_time, stop_time)
            new_segment.epochs.append(new_epoch)
            epoch.segment = new_segment

        new_segment.create_many_to_one_relationship()
        new_segments.append(new_segment)

    del block.segments[:]
    for segment in new_segments:
        block.segments.append(segment)

    block.create_many_to_one_relationship(force=True, recursive=True)
    block.create_many_to_many_relationship()
    block.create_relationship(force=True, recursive=True, append=True)

    return block

