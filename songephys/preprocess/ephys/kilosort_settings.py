"""Kilosort4 settings for songephys: defaults <- project KILOSORT <- site kilosort.

Example site_config.yaml::

    kilosort:
      nblocks: 0          # drift is tiny at this site
      Th_universal: 8

Example project config::

    KILOSORT:
      dminx: 22.5

Keys are Kilosort4 setting names (see kilosort.parameters / DEFAULT_SETTINGS).
They are matched case-insensitively; unknown keys are ignored with a warning.
"""
import warnings

# songephys choices on top of Kilosort's own DEFAULT_SETTINGS.
SONGEPHYS_KILOSORT_DEFAULTS = {
    "nblocks": 1,
    "batch_size": 30000 * 6,   # 6 s of data
    "Th_universal": 7,         # lower than KS default (9) to detect more units
    "dminx": 18.5,
    "templates_from_data": True,
    "nearest_templates": 100,  # capped at the number of probe channels
    "nearest_chans": 8,        # capped at the number of probe channels
}

# Set by songephys from the data / probe; not overridable from config.
_COMPUTED = {"n_chan_bin"}
# Capped at the number of channels of the (sub)probe being sorted.
_CAPPED_BY_N_CHAN = ("nearest_templates", "nearest_chans")


def _overrides(source, known, where):
    out = {}
    lower = {k.lower(): k for k in known}
    for k, v in (source or {}).items():
        key = lower.get(str(k).lower())
        if key is None:
            warnings.warn("Unknown Kilosort setting %r in %s ignored" % (k, where))
            continue
        if key in _COMPUTED:
            warnings.warn("Kilosort setting %r in %s is set by songephys; ignored" % (k, where))
            continue
        out[key] = v
    return out


def get_kilosort_overrides(kilosort_defaults, site_config=None, project_config=None):
    """songephys defaults <- project KILOSORT <- site kilosort (only the overrides)."""
    known = set(kilosort_defaults)
    cfg = {k: v for k, v in SONGEPHYS_KILOSORT_DEFAULTS.items() if k in known}
    cfg.update(_overrides((project_config or {}).get("KILOSORT"), known, "project config KILOSORT"))
    cfg.update(_overrides((site_config or {}).get("kilosort"), known, "site_config kilosort"))
    return cfg


def build_kilosort_settings(kilosort_defaults, n_chan_bin, probe_n_chan,
                            site_config=None, project_config=None):
    """Full Kilosort4 settings dict for one run."""
    settings = dict(kilosort_defaults)
    settings.update(get_kilosort_overrides(kilosort_defaults, site_config, project_config))
    settings["n_chan_bin"] = n_chan_bin
    for k in _CAPPED_BY_N_CHAN:
        if k in settings and settings[k] is not None:
            settings[k] = min(int(probe_n_chan), int(settings[k]))
    return settings


def describe_overrides(kilosort_defaults, settings):
    """'key: value' lines for settings that differ from Kilosort's defaults."""
    return ["%s: %r" % (k, v) for k, v in sorted(settings.items())
            if k not in kilosort_defaults or kilosort_defaults[k] != v]
