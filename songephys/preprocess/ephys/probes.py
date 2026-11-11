"""Which probe a site was recorded with: site_config ``probe`` <- project EPHYS ``PROBE``.

Set ``probe`` in site_config.yaml when a site used a different probe from the rest
of the project (e.g. after a re-implant)::

    probe: H10

The value is a key of PROBE_FILES, a probe file name, or a path to a probe file.
Files are looked up in Kilosort's probe folder first, then in the repo's probes/.
"""
import os

PROBE_FILES = {
    "P1": "cambridge_P1.prb",          # 4 shank
    "H10": "cambridge_H10.prb",        # 2 shank
    "H15": "cambridge_H15.json",       # 2 shank
    "M16": "microprobes_16ch.prb",     # microprobes with 16 wires
}

REPO_PROBE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "probes"))


def get_probe_name(site_config=None, project_config=None, default="H10"):
    """The site's probe: site ``probe`` <- project EPHYS PROBE (or older PROBETYPE) <- default."""
    site_probe = (site_config or {}).get("probe")
    if site_probe:
        return str(site_probe)
    ephys = (project_config or {}).get("EPHYS") or {}
    return str(ephys.get("PROBE") or ephys.get("PROBETYPE") or default)


def find_probe_file(name, probe_dirs=()):
    """Path of the probe file for a PROBE_FILES key, a file name, or a path."""
    known = {k.upper(): v for k, v in PROBE_FILES.items()}
    filename = known.get(name.upper(), name)
    candidates = [os.path.join(d, filename) for d in list(probe_dirs) + [REPO_PROBE_DIR]]
    if os.path.dirname(filename):
        candidates.insert(0, filename)
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise ValueError("Probe %r not found. Use one of %s, or a probe file name or path. Looked for: %s"
                     % (name, sorted(PROBE_FILES), candidates))


def load_site_probe(site_config=None, project_config=None, default="H10"):
    """Load the site's probe with Kilosort. Returns (probe dict, name, file path)."""
    import kilosort
    name = get_probe_name(site_config, project_config, default)
    path = find_probe_file(name, [str(kilosort.PROBE_DIR)])
    print("Using probe %s (%s)%s" % (name, path, " from site_config" if (site_config or {}).get("probe") else ""))
    return kilosort.io.load_probe(path), name, path
