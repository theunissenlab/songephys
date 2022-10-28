import numpy as np


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
