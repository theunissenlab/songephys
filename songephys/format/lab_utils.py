import os


def _theunissen_drive2local(filename):
    # filename is a format of '{BIRDNAME}_190616_155033.rhd'
    return filename


def _seewiesen_drive2local(filename):
    # filename is a format of '{BIRDNAME}__{x}_190616_155033.rhd'
    # the {x} is unnecessary
    return filename
    # parts = filename.split("_")
    # return "{}_{}_{}".format(parts[0], parts[-2], parts[-1])


drive2local_by_lab = {
    "theunissen": _theunissen_drive2local,
    "seewiesen": _seewiesen_drive2local,
}
