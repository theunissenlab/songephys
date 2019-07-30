import numpy as np
import quantities as pq
from segmentation.autodetect import get_amplitude_envelope, threshold_events
from soundsig.signal import bandpass_filter

import config


def detect_syllables():
    """Takes mic data and identifies time periods where there might be syllables
    """
    pass


def shave_silences(event, loader):
    """Use amplitude envelope to shave off any silent padding"""
    event = dict(event)
    fs = loader._signals["mic"][0].sampling_rate

    if event.get("sound_type") == "silence":
        return event

    # elif event.get("vocalization_type") == "So":
    elif event.get("sound_type") == "live":
        t0 = max(0, event["start_time"].item())
        t1 = min(loader.max_time(), (event["start_time"] + event["duration"]).item())
        if t1 < t0:
            return event

        sig = loader.time_slice(t0, t1, signal="mic").data
        sig = sig - np.mean(sig, axis=0)
        sig = bandpass_filter(sig.T, fs, 1000, 8000).T
        amp_env = get_amplitude_envelope(sig, fs)
        amp_env = np.mean(amp_env, axis=1)

        threshold = 1.3 * np.median(
            np.concatenate([
                amp_env[:int(fs * config.PAD_SONG_BEFORE)],
                amp_env[-int(fs * config.PAD_SONG_AFTER):]
            ])
        )

        intervals = threshold_events(amp_env, threshold, sampling_rate=fs,
            ignore_width=0.01, min_size=0.1, fuse_duration=0.2)
        if len(intervals):
            idx0, idx1 = intervals[0][0], intervals[-1][1]
            event["start_time"] += (idx0 / fs) * pq.s
            event["duration"] = ((idx1 - idx0) / fs) * pq.s
            return event

    event["start_time"] += config.PAD_SONG_BEFORE * pq.s
    event["duration"] -= (config.PAD_SONG_AFTER + config.PAD_SONG_BEFORE) * pq.s

    return event


def get_surrounding_silence(
        all_events,
        loader,
        channel=0,
        min_duration=1.0,
        min_sound=0.002,
        max_duration=20.0,
        sampling_rate=30000
        ):
    """Return time periods corresponding to silence around given vocal events"""
    silence_events = []

    sr = loader._signals["mic"][0].sampling_rate

    for i in np.arange(len(all_events) - 1):
        e1 = all_events[i]
        e2 = all_events[i + 1]
        
        t0 = (e1["start_time"] + e1["duration"]).item() - config.PAD_SONG_AFTER
        t1 = min(t0 + max_duration, e2["start_time"].item() + config.PAD_SONG_BEFORE)

        if t1 - t0 < min_duration:
            continue
            
        time_slice = loader.time_slice(t0, t1, signal="mic")
        sig = time_slice.data[:, channel]
        t_arr = time_slice.t
        ampenv = get_amplitude_envelope(sig - np.mean(sig))
        threshold = 3.0 * np.median(ampenv)
        
        intervals = threshold_events(
            ampenv,
            threshold,
            polarity=-1,
            sampling_rate=sr,
            min_size=min_duration
        )
        
        for start_idx, stop_idx in intervals:
            silence_start = t_arr[start_idx]
            silence_stop = t_arr[-1] if stop_idx == len(t_arr) else t_arr[stop_idx]
            new_event = {
                "start_time": (silence_start) * pq.s,
                "duration": (silence_stop - silence_start) * pq.s,
                "sound_type": "silence",
            }
            silence_events.append(new_event)
    
    return silence_events
