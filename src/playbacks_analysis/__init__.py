import numpy as np

from segment_analysis import MultiSegment


class PlaybacksManager(object):
    """Helper class to deal with sessions involving sound playback

    Uses MultiSegment to filter
   
    Example:
    >>> pbm = PlaybacksManager(block.segments)
    >>> trials_viewer = pbm.get_playback_responses(<stim_id>)
    >>> trials_viewer.plot_psth(unit=1)
    """

    def __init__(self, segments):
        self.segments = filter(
                lambda seg: seg.annotations["sound_type"] == "playback",
                segments)

        # MultiSegment with all segments
        self.full_multiseg = MultiSegment(self.segments)

        # TODO fix this to access the file correctly every time
        self.stim_ids = np.unique(list(self.get_id(seg) for seg in self.segments))

    def get_id(self, seg):
        """Return id of the playback file for the given segment"""
        # TODO: figure out if I should use the playback filename or stimulus id
        return seg.annotations["playback_file"].split("/")[-1]
        return seg["stim_id"]

    def group_playback_responses(self):
        """Return MultiSegment instances for every stim id"""
        stim_ids = [self.get_id(seg) for seg in self.segments]

        return dict(
            (
                stim_id,
                self.full_multiseg.filter(lambda seg: self.get_id(seg) == stim_id)
            )
            for stim_id in stim_ids
        )

