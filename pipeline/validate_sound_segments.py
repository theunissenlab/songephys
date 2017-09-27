"""Manual validation of song segments

INPUT
_-segmented.nix

OUTPUT
_-validated.nix
"""
import cPickle as pickle

from audio_segment import segment


if __name__ == "__main__":
    import sys

    # read bird, date, time

    base = "{bird}_{date}_{time}.rhd".format(**args)

    in_file = "{base}.nix".format(base)
    out_file = "{base}-segmented.pkl".format(base)

    check_exists(in_file)
    check_doesnt_exist(out_file)

    # right now we do nothing...

    # copy segmented to validated
