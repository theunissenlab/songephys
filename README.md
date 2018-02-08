# songephys

## Setup

Best way to do everything is to use virtualenv. Create an isolated python virtualenv with `virtualenv env`.
If you want to use conda you need to make sure you install python dependencies in requirements.txt and also set up your PYTHONPATH to include `src`, `pipeline`, and `load_intan_rhd_format` directories

Then setup by:

1. Activate virtualenv: `source bin/activate` (this script also sets up the PATH properly)

2. Install python dependencies: `pip install -r requirements.txt`

3. Maybe it works.

4. Oh right you need to clone [`load_intan_rhd_format` repository](https://github.com/theunissenlab/load_intan_rhd_format.git)

## Dependencies

The code we are using in songephys depends on several of our internal packages, including soundsig, neosound, load_intan_rhd_format, etc.

The three options to install these are (all need to be tested)

1. Clone the repo for the dependency, add directory of the dependencies to the PYTHONPATH (add that to `bin/activate` script)

2. Add a setup.py file to the other project directory, and `pip install <directory>`

3. Install the repository directory directly `pip install -e <directory>`


## Gdrive access

settings.yaml.default

How to get the client id and secret?
