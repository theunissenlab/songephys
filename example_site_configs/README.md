## Example site_config.yaml

The first step in setting up a site is creating the `site_config.yaml`. A template can be made using `bin/song create-site SITE_DIR`.

These example site_config.yaml files would work when copied to `DATA_DIR/EXPERIMENT_DIR/sites/SITE_DIR/site_config.yaml`.

## TODO

Include a small sample rhd file with stimuli and csv files; and a site_config.yaml file with `download_type=local` that can actually run.


## Neuropixels (SpikeGLX) bird configs

`npx_bird_config.yaml` is the config for one bird recorded with Neuropixels and
SpikeGLX: paths, NI line map, probe placement and the CatGT / Kilosort / TPrime
settings (defaults in `songephys/npx/config.py`). Each entry under `sessions` is
one SpikeGLX run and becomes one NWB file:

    song npx-sort    path/to/Bird_config.yaml       # CatGT -> Kilosort 4 -> NI-clock alignment
    song npx-gen-nwb path/to/Bird_config.yaml       # NWB with NI signals, syllables, NCAF, units
    song npx-add-units path/to/Bird_config.yaml --clear-old-units   # after re-sorting / curating

See `NWB_SCHEMA_NPX.md` for what ends up in the file.
