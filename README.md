# songephys

The general purpose of this repository is for preprocessing of data from our electrophysiological experiemnts in zebra finches.

The core code is in the code/ folder.

The data pipeline are defined in luigi tasks in the pipeline/ folder. Each task is defined by its inputs, its outputs, and some function that transforms the inputs to outputs.

The code in here should be for general data pre-processing and preliminary analysis. For detailed analysis for specific projects, use songephys to transform your data into package(s) of files
that can be easily read in to a separate, specialized project.

## Setup

Best way to do everything is to use virtualenv. Create an isolated python virtualenv with `virtualenv env`.
If you want to use conda you need to make sure you install python dependencies in requirements.txt and also set up your PYTHONPATH to include `src`, `pipeline`, and `load_intan_rhd_format` directories

Then setup by:

1. Activate virtualenv: `source bin/activate` (this script also sets up the PATH properly)

2. Install python dependencies: `pip install -r requirements.txt`

3. Oh right you need to clone [`load_intan_rhd_format` repository](https://github.com/theunissenlab/load_intan_rhd_format.git). Add a symlink to it or add it to the PYTHONPATH

4. Set up google drive api credentials. `mkdir tlab_gdrive_credentials`, then use `python -m gdrive_access.setup_credentials --dir tlab_gdrive_credentials` and follow the instructions carefully. See [gdrive-access](https://github.com/theunissenlab/gdrive_python) for details.

## Dependencies

The code we are using in songephys depends on several of our internal packages, including soundsig, neosound, load_intan_rhd_format, etc.

The three options to install these are (all need to be tested)

1. Clone the repo for the dependency, add directory of the dependencies to the PYTHONPATH (add that to `bin/activate` script)

2. Add a setup.py file to the other project directory, and `pip install <directory>`

3. Install the repository directory directly `pip install -e <directory>`

4. Install libgpuarray (for Theano) by cd to libgpuarray and doing `python setup.py build` and `python setup.py install`

