

class VocalObject(object):

    def __init__(self, signal, fs):
        """A template for a vocal object (call or syllable)

        This can serve as a model for detecting variable sound sources
        """
        self.signal = sig
        self.fs = fs


class Syllable(object):
    """A syllable object

    Refers to a specific site by time relative to start time of that site

    optionall can label it by its vocal object
    """
    def __init__(self, site_dir, t_start, t_stop, vocal_object=None):
        self.site_dir = site_dir
        self.t_start = t_start
        self.t_stop = t_stop
        self.vocal_object = vocal_object

