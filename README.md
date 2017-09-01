# songephys

# Setup

Best way to do everything is to use virtualenv. Create an isolated python virtualenv with `virtualenv env`.
If you want to use conda you need to make sure you install python dependencies in requirements.txt and also set up your PYTHONPATH to include `src`, `pipeline`, and `load_intan_rhd_format` directories

Then setup by:

1. Activate virtualenv: `source bin/activate` (this script also sets up the PATH properly)

2. Install python dependencies: `pip install -r requirements.txt`

3. Maybe it works.

4. Oh right you need to clone [`load_intan_rhd_format` repository](https://github.com/theunissenlab/load_intan_rhd_format.git)
