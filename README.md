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

## Dependencies

The code we are using in songephys depends on several of our internal packages, including soundsig, neosound, load_intan_rhd_format, etc.

The three options to install these are (all need to be tested)

1. Clone the repo for the dependency, add directory of the dependencies to the PYTHONPATH (add that to `bin/activate` script)

2. Add a setup.py file to the other project directory, and `pip install <directory>`

3. Install the repository directory directly `pip install -e <directory>`

4. Install libgpuarray (for Theano) by cd to libgpuarray and doing `python setup.py build` and `python setup.py install`


## Gdrive access

settings.yaml.default

### Instructions to get client_secrets.json

Directions found here: https://pythonhosted.org/PyDrive/quickstart.html

Drive API requires OAuth2.0 for authentication. PyDrive makes your life much easier by handling complex authentication steps for you.

    Go to APIs Console (https://console.developers.google.com/cloud-resource-manager?pli=1) and make your own project.
    Search for ‘Google Drive API’, select the entry, and click ‘Enable’.
    Select ‘Credentials’ from the left menu, click ‘Create Credentials’, select ‘OAuth client ID’.
    Now, the product name and consent screen need to be set -> click ‘Configure consent screen’ and follow the instructions. Once finished:

        Select ‘Application type’ to be Web application.
        Enter an appropriate name.
        Input http://localhost:8080 for ‘Authorized JavaScript origins’.
        Input http://localhost:8080/ for ‘Authorized redirect URIs’.
        Click ‘Save’.

    Click ‘Download JSON’ on the right side of Client ID to download client_secret_<really long ID>.json.

The downloaded file has all authentication information of your application. Rename the file to “client_secrets.json” and place it in your working directory.

Create quickstart.py file and copy and paste the following code.
