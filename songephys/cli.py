#!/usr/bin/env python
import os
import sys

# Thread limits must be set before numpy/scipy/torch/sklearn are imported (this
# module imports pandas, hence numpy, further down). See sortsinglesite.
if len(sys.argv) > 1 and sys.argv[1] == "sort-singlesite":
    os.environ.setdefault("OMP_NUM_THREADS", "7")

import pprint
import click


@click.group()
def cli():
    pass


@click.group()
def info():
    pass


@click.command("datadir")
def info_datadir():
    click.echo("data")

info.add_command(info_datadir)


@click.group()
def rhd():
    pass


@click.command("rhd-header")
@click.argument("filename", type=click.Path(exists=True))
def rhd_header(filename):
    from intanutil import read_header
    with open(filename, "rb") as fid:
        header = read_header.read_header(fid)
    click.echo(pprint.pformat(header, indent=4))


@click.command("wav-info")
@click.argument("filename", type=click.Path(exists=True))
def wav_info(filename):
    import soundfile
    with soundfile.SoundFile(filename, "r") as fid:
        info = {
            "sampling_rate": fid.samplerate,
            "frames": fid.frames,
            "channels": fid.channels
        }
    click.echo(pprint.pformat(info, indent=4))


@click.command()
@click.argument("filename", type=click.Path(exists=True))
def rhd2nix(filename):
    from intanutil import read_header
    with open(filename, "rb") as fid:
        header = read_header.read_header(fid)
    click.echo(pprint.pformat(header, indent=4))


class SiteConfigOption:
    def __init__(self, instructions, settings=None):
        self.instructions = instructions
        self.settings = settings

    def apply_to_config_dict(self, config):
        if self.settings is not None:
            config.update(self.settings)


_configs = (
    SiteConfigOption(
        instructions="None of the below",
    ),

    SiteConfigOption(
        instructions="16 channel, 1 mic",
        settings={
            "wavmap": {
                "ephys.wav": [["Amplifier", [0, 16]]],
                "mic.wav": [["Board ADC", 0]]
            },
            "filtering": {
                "ephys.wav": [[0, 16]],
                "mic.wav": [[0, 1]]
            }
        }
    ),

    SiteConfigOption(
        instructions="16 channel, 2 mic",
        settings={
            "wavmap": {
                "ephys.wav": [["Amplifier", [0, 16]]],
                "mic.wav": [["Board ADC", [0, 2]]]
            },
            "filtering": {
                "ephys.wav": [[0, 16]],
                "mic.wav": [[0, 2]]
            }
        }
    ),

    SiteConfigOption(
        instructions="32 channel, 1 mic",
        settings={
            "wavmap": {
                "ephys.wav": [["Amplifier", [0, 32]]],
                "mic.wav": [["Board ADC", 0]]
            },
            "filtering": {
                "ephys.wav": [[0, 32]],
                "mic.wav": [[0, 1]]
            }
        }
    ),

    SiteConfigOption(
        instructions="32 channel, 2 mic",
        settings={
            "wavmap": {
                "ephys.wav": [["Amplifier", [0, 32]]],
                "mic.wav": [["Board ADC", [0, 2]]]
            },
            "filtering": {
                "ephys.wav": [[0, 32]],
                "mic.wav": [[0, 1]]
            }
        }
    ),

    SiteConfigOption(
        instructions="32 channel, 3 mic (2+1)",
        settings={
            "wavmap": {
                "ephys.wav": [["Amplifier", [0, 32]]],
                "mic.wav": [["Board ADC", [0, 3]]]
            },
            "filtering": {
                "ephys.wav": [[0, 32]],
                "mic.wav": [[0, 2], [2, 3]]
            }
        }
    ),

    SiteConfigOption(
        instructions="2 subj [0-14 ephys][15 mic][16-30 ephys][31 mic], 3 mic",
        settings={
            "wavmap": {
                "ephys.wav": [["Amplifier", [0, 15]], ["Amplifier", [16, 31]]],
                "mic.wav": [["Amplifier", 15], ["Amplifier", 31], ["Board ADC", [0, 3]]],
            },
            "filtering": {
                "ephys.wav": [[0, 15], [15, 30]],
                "mic.wav": [[0, 2], [2, 5]],
            }
        }
    ),

    SiteConfigOption(
        instructions="2 subj [0-14 ephys][15 mic][16-30 ephys][31 mic], 3 mic, 6 motion",
        settings={
            "wavmap": {
                "ephys.wav": [["Amplifier", [0, 15]], ["Amplifier", [16, 31]]],
                "mic.wav": [["Amplifier", 15], ["Amplifier", 31], ["Board ADC", [0, 3]]],
                "motion.wav": [["AUX Input", [0, 6]]],
            },
            "filtering": {
                "ephys.wav": [[0, 15], [15, 30]],
                "mic.wav": [[0, 2], [2, 5]],
            }
        }
    ),

    SiteConfigOption(
        instructions="1 subj [0-14 ephys][15 mic], 3 mic",
        settings={
            "wavmap": {
                "ephys.wav": [["Amplifier", [0, 15]]],
                "mic.wav": [["Amplifier", 15], ["Board ADC", [0, 3]]],
            },
            "filtering": {
                "ephys.wav": [[0, 15]],
                "mic.wav": [[0, 1], [1, 4]],
            }
        }
    ),

    SiteConfigOption(
        instructions="1 subj [0-14 ephys][15 mic], 3 mic, 3 motion",
        settings={
            "wavmap": {
                "ephys.wav": [["Amplifier", [0, 15]]],
                "mic.wav": [["Amplifier", 15], ["Board ADC", [0, 3]]],
                "motion.wav": [["AUX Input", [0, 3]]],
            },
            "filtering": {
                "ephys.wav": [[0, 15]],
                "mic.wav": [[0, 1], [1, 4]],
            }
        }
    ),
)

@click.command("create-config")
@click.argument("experiment_name", type=click.Path())
# out_path defaults to pwd
@click.argument('data_dir', type=click.Path(), default=os.environ.get("DATA_DIR"))
@click.argument("out_path", type=click.Path(), default= os.getcwd())
def create_config(experiment_name, out_path, data_dir):
    from songephys.config import generate_config_file
    generate_config_file(experiment_name, out_path=out_path, data_dir=data_dir)

@click.command("create-remote-config")
@click.argument("remote_path", type=click.Path())
@click.argument("out_path", type=click.Path(), default="")
def create_remote_config(remote_path, out_path):
    from songephys.config import generate_remote_config
    if out_path == "":
        generate_remote_config(remote_path)
    else:
        generate_remote_config(remote_path, out_path)

@click.command("list-sites")
@click.argument("experiment_name", type=click.Path(), default=".")
def list_sites(experiment_name):
    from songephys.config import get_project_path
    import glob, os
    project_path = get_project_path(experiment_name)
    sites = glob.glob(os.path.join(project_path, "sites", "*/site_config.yaml"))
    for site in sites:
        click.echo(os.path.relpath(os.path.dirname(site), start = project_path))

@click.command("create-site")
@click.argument("site_dir", type=click.Path())
@click.option('--auto', is_flag=True)
@click.option('--ml', type=float, default=0)
@click.option('--depth', type=float,default=0)
@click.option('--bad_channels', type=str, default="")
def create_site(site_dir, auto, ml, depth, bad_channels):
    from songephys.format.download import downloaders
    from songephys.format.io import load_yaml, save_yaml
    import numpy as np

    if not os.path.exists(site_dir):
        os.makedirs(site_dir)

    project_config_path = os.path.join(site_dir.split("sites")[0], "project_config.yaml")
    if not os.path.exists(project_config_path):
        print("Project config not found at %s"%project_config_path)
        print("Please run `song create-config EXPERIMENT_NAME` for this Experiment")
        return
    project_config = load_yaml(project_config_path)


    config_path = os.path.join(site_dir, "site_config.yaml")

    if os.path.exists(config_path):
        if not click.confirm("{} already exists. Do you want to overwrite it?".format(config_path)):
            return

    valid_downloaders = list(downloaders.keys())
    if auto:
        click.echo("Auto mode, using local downloader")
        download_type = 'local'
    else:
        download_type = None
        while not download_type:
            download_type = input("Download Type [choose from: {}]: ".format(valid_downloaders))
            if download_type not in valid_downloaders:
                click.echo("Download type {} invalid. Choose from {}".format(download_type, valid_downloaders))
                download_type = None

    filelistpath = os.path.join(site_dir,'filelist.txt')
    click.echo(filelistpath)
    if not os.path.exists(filelistpath):
        if auto:
            click.echo("Auto mode, no filelist found, exiting")
            raise ValueError("Auto mode requires a filelist")
        click.echo("Type or paste all .rhd and .csv files for the site separated by line breaks\n"
        "To end recording Press Ctrl+d on Linux/Mac on Crtl+z on Windows")

        filelist = []
        try:
            while True:
                filelist.append(input())
        except EOFError:
            pass

        filelist = [line.strip() for line in filelist]
        filelist = [line for line in filelist if line.endswith("csv") or line.endswith("rhd")]
    else:
        print("Reading filelist from %s"%filelistpath)
        f = open(filelistpath,'r')
        filelist = [line.strip() for line in f.readlines()]
    click.echo("Have filelist {}".format(filelist))

    default_experiment_name = os.path.basename(site_dir).split("_")[0]
    if auto:
        experiment_name = default_experiment_name
    else:
        experiment_name = input("Experiment Name [default='{}']: ".format(default_experiment_name))
        experiment_name = experiment_name.strip() or default_experiment_name

    bcs = []
    if bad_channels != "":
        bcs = [int(x) for x in bad_channels.split(",")]
    config_data = {
        "download_type": download_type,
        "experiment_name": experiment_name,
        "filelist": filelist,
        "playback_mic_channels": None,
        "spike_sorting.seconds_before_peak": 0.0005,
        "spike_sorting.seconds_after_peak": 0.0015,
        "bad_channels": bcs
    }
    xml_check = [fn.endswith('settings.xml') for fn in filelist]
    if any(xml_check):
        xml_file = filelist[np.where(xml_check)[0][0]]
        if not os.path.isabs(xml_file):
            xml_file = os.path.join(project_config['LOCAL_RAW_FILES']['rhd_drive'],xml_file)
        print("Loading electrode config from %s" % xml_file)
        electrode_config = parse_settings_file(xml_file)

        chosen_config_option = SiteConfigOption(
                instructions="loaded from %s"%xml_file,
                settings={
                    "wavmap": {
                        "ephys.wav": [["RHD2000 amplifier channel", [0, len(electrode_config['Amplifier'])]]],
                        "mic.wav": [["USB board ADC input channel", [0, len(electrode_config['Analog'])]]],
                        "digital.wav": [["Digital Input", [0, len(electrode_config['Digital'])]]],
                        "motion.wav": [["RHD2000 auxiliary input channel", [0, len(electrode_config['AUX'])]]]
                    },
                    "filtering": {
                        "ephys.wav": [[0, len(electrode_config['Amplifier'])]],
                        "mic.wav": [[0, len(electrode_config['Analog'])]],
                    }
                }
            )
        config_data['digital_channel_names'] = electrode_config['Digital']
        config_data['digital_channel_bits'] = electrode_config['DigitalBits']
        config_data['analog_channel_names'] = electrode_config['Analog']

        # optional channels.csv in the recording folder with names/descriptions for each line
        channels_csv = os.path.join(os.path.dirname(xml_file), CHANNELS_FILENAME)
        if os.path.exists(channels_csv):
            print("Loading channel names from %s" % channels_csv)
            channel_metadata = load_channel_metadata(channels_csv)
            enabled_natives = set(electrode_config['DigitalNative'] + electrode_config['AnalogNative'])
            for native in channel_metadata:
                if native not in enabled_natives:
                    click.echo("WARNING: %s is in %s but is not enabled in %s" % (native, channels_csv, xml_file))
            config_data['digital_channel_names'] = apply_channel_metadata(
                electrode_config['DigitalNative'], electrode_config['Digital'], channel_metadata)
            config_data['analog_channel_names'] = apply_channel_metadata(
                electrode_config['AnalogNative'], electrode_config['Analog'], channel_metadata)
            config_data['channel_metadata'] = channel_metadata
            for native, name in zip(electrode_config['DigitalNative'] + electrode_config['AnalogNative'],
                                    config_data['digital_channel_names'] + config_data['analog_channel_names']):
                print("  %s -> %s" % (native, name))
        else:
            print("No %s found next to %s, using Intan channel names" % (CHANNELS_FILENAME, xml_file))
    else:
        if auto:
            click.echo("Auto mode, no settings file found, exiting")
            raise ValueError("Auto mode requires a settings file")
        options = list(enumerate(_configs))

        click.echo("Next, choose from a preset electrode configuration of ones we've used before.")
        for idx, option in options:
            click.echo("{}: {}".format(idx, option.instructions))

        valid_elec_configs = list(range(len(options)))
        elec_config = None
        while elec_config is None:
            elec_config = input("Preset electrode config [0-{}]:".format(len(options)))
            try:
                elec_config = int(elec_config)
            except:
                elec_config = None

            if elec_config not in valid_elec_configs:
                click.echo("Electrode config {} invalid. Choose from {}".format(elec_config, valid_elec_configs))
                elec_config = None

        chosen_config_option = _configs[elec_config]
    chosen_config_option.apply_to_config_dict(config_data)

    click.echo("Which microphone channel(s) would capture playback audio?")
    click.echo("Separate each with a comma or leave blank to use all.")
    click.echo("(this is only important if there are no digital codes for synchronizing playbacks to intan)")

    ch = 0
    if "wavmap" in config_data:
        for k, v in config_data["wavmap"]["mic.wav"]:
            if isinstance(v, list):
                for i in range(*v):
                    click.echo("{}: {} {}".format(ch, k, i))
                    ch += 1
            else:
                click.echo("{}: {} {}".format(ch, k, v))
                ch += 1

    if auto:
        playback_channels = [0]
    else:
        valid = False
        while valid is False:
            response = input("Mic channels with playback (separated by commas, default all): ")
            response = response.strip()

            if response == "":
                playback_channels = None
                valid = True
            else:
                try:
                    playback_channels = [int(x) for x in response.strip().split(",")]
                except:
                    click.echo("Invalid input")
                else:
                    valid = True

    config_data["playback_mic_channels"] = playback_channels

    if auto:
        if depth == 0:
            click.echo("Auto mode, you must set depth using --depth")
            raise ValueError("Auto mode requires a depth")

        config_data['depth_mm'] = depth
        click.echo("AP: %.2f mm ap from project_config"%project_config['EPHYS']['AP'])
        config_data['ap_mm'] = project_config['EPHYS']['AP']
        click.echo("ML: %.2f mm from project_config"%project_config['EPHYS']['ML'])
        config_data['ml_mm'] = project_config['EPHYS']['ML']
    else:
        response = input("What is the depth of this recording? (in mm) default %.2f"%depth)
        response = response.strip()
        if response == "":
            config_data['depth_mm'] = depth
        else:
            config_data['depth_mm'] = float(response)

        response = input("What is the medial-lateral position of this recording? (in mm)")
        response = response.strip()
        config_data['ml_mm'] = float(response)

        response = input("What is the anterior-posterior position of this recording? (in mm)")
        response = response.strip()
        config_data['ap_mm'] = float(response)


    click.echo("Okay. Saving {}".format(config_path))

    save_yaml(config_path, config_data)

    click.echo("Download files with 'song download-site {}'".format(site_dir))


@click.command("update-filelist")
@click.argument("site_dir", type=click.Path())
def update_filelist(site_dir):
    from songephys.format.download import downloaders
    from songephys.format.io import load_yaml, save_yaml

    if not os.path.exists(site_dir):
        os.makedirs(site_dir)


    config_path = os.path.join(site_dir, "site_config.yaml")

    if not os.path.exists(config_path):
        print("Site config not found at %s"%config_path)
        return
    config = load_yaml(config_path)

    filelistpath = os.path.join(site_dir,'filelist.txt')
    click.echo(filelistpath)
    if not os.path.exists(filelistpath):
        print("Filelist not found at %s"%filelistpath)
        return
    else:
        print("Reading filelist from %s"%filelistpath)
        f = open(filelistpath,'r')
        filelist = [line.strip() for line in f.readlines()]
    click.echo("Have filelist {}".format(filelist))

    config['filelist'] = filelist
    save_yaml(config_path, config)


@click.command("download-site")
@click.argument("site_dir", type=click.Path(exists=True))
def download_site(site_dir):
    import luigi
    from songephys.pipeline.file_setup import DownloadSite
    luigi.build([DownloadSite(site_dir=site_dir)], local_scheduler=True)


@click.command("gen-singlesite-files")
@click.argument("site_dir", type=click.Path(exists=True))
def gensinglesitefiles(site_dir):
    import luigi
    from songephys.pipeline.file_setup import GenerateSingleSiteFiles
    _check_tmax(site_dir)
    luigi.build([GenerateSingleSiteFiles(site_dir=site_dir)], local_scheduler=True)


def _check_tmax(site_dir):
    """Stop if the site has a tmax but outputs made before it are still around."""
    from songephys.format.io import load_yaml
    from songephys.pipeline.truncation import check_site_outputs_respect_tmax, get_site_tmax
    cfg_path = os.path.join(site_dir, "site_config.yaml")
    if not os.path.exists(cfg_path):
        return
    site_config = load_yaml(cfg_path)
    check_site_outputs_respect_tmax(site_dir, site_config)
    tmax = get_site_tmax(site_config)
    if tmax is not None:
        print("Site is truncated at tmax = %.3f s" % tmax)

@click.command("sort-singlesite")
@click.argument("site_dir", type=click.Path())
@click.option("-d", "--dir", "_dir", type=str, default="ks4_final")
def sortsinglesite(site_dir, _dir, **kwargs):
    # Kilosort's "universal templates from data" step runs scikit-learn KMeans; on
    # many-core Windows machines with the default OpenMP thread count this balloons
    # to >100 GB of memory and crashes (see MouseLand/Kilosort#866). The limit only
    # takes effect if set before numpy/torch/sklearn are imported, so it is set at
    # the top of this module when the command is sort-singlesite (this repeat only
    # covers other ways of calling the function). An existing value is respected.
    os.environ.setdefault("OMP_NUM_THREADS", "7")
    print("OMP_NUM_THREADS=%s" % os.environ["OMP_NUM_THREADS"])
    import luigi
    from songephys.pipeline.preprocess_ephys import ExportSortEvents
    if not os.path.exists(site_dir):
        os.makedirs(site_dir, exist_ok=True)
    _check_tmax(site_dir)
    # ExportSortEvents requires the sort (KiloSort4Sorting), then writes events.csv next to
    # it. On a sort that has already finished (its 'complete' file exists) Kilosort is
    # skipped and only events.csv is written.
    luigi.build([ExportSortEvents(site_dir=site_dir, kilosort_dir=_dir)], local_scheduler=True)

@click.command("gen-nwb")
@click.argument("site_dir", type=click.Path(exists=True))
@click.option("-a", "--add-spectrograms", is_flag=True)
def gennwb(site_dir, add_spectrograms):
    import luigi
    from songephys.pipeline.postprocess_stimuli import GenerateBaseNWB
    _check_tmax(site_dir)
    print("Generating NWB for site: {} with add_spectrograms: {}".format(site_dir, add_spectrograms))
    luigi.build([GenerateBaseNWB(site_dir=site_dir, add_spectrogram_info=add_spectrograms)], local_scheduler=True)


@click.command('add-ripples')
@click.argument("site_dir", type=click.Path(exists=True))
def add_ripples(site_dir):
    import luigi
    from songephys.pipeline.postprocess_ripples import AgglomerateRippleEventsNWB
    luigi.build([AgglomerateRippleEventsNWB(site_dir=site_dir)], local_scheduler=True)

@click.command('add-sorted-units')
@click.argument("site_dir", type=click.Path(exists=True))
@click.option("-d", "--dir", "_dir", type=str, default="ks4_indiv_shanks_tfd")
@click.option('--clear_old_units', is_flag=True)
def addsortedunits(site_dir,_dir,clear_old_units):
    import luigi
    from songephys.pipeline.nwb_utils import AddKilosortUnits, AddUnitValidIntervals, AddUnitWaveforms
    luigi.build([AddKilosortUnits(site_dir=site_dir, kilosort_dir=_dir,clear_old_units=clear_old_units),
                    AddUnitValidIntervals(site_dir=site_dir, kilosort_dir=_dir),
                    AddUnitWaveforms(site_dir=site_dir, kilosort_dir=_dir)], local_scheduler=True)

@click.command("run-pipeline-job")
@click.argument("module")
@click.argument("job_name")
@click.argument("site_dir", type=click.Path(exists=True))
def runpipelinejob(module, job_name, site_dir):
    import luigi, importlib, songephys
    module = importlib.import_module("songephys.pipeline.%s"%module)
    job = eval('module.%s'%job_name)
    luigi.build([job(site_dir=site_dir)], local_scheduler=True)


@click.command(help="Run unittests")
@click.option("-d", "--dir", "_dir", type=str, default="tests/")
@click.option("-v", "--verbose", type=int, default=1)
def unittest(_dir, verbose):
    import unittest

    if os.path.isdir(_dir):
        testsuite = unittest.TestLoader().discover(_dir)
    else:
        testsuite = unittest.TestLoader().loadTestsFromName(_dir)
    unittest.TextTestRunner(verbosity=verbose).run(testsuite)


import datetime
import pandas as pd
import glob

def get_start_end(rhd_file):
    from songephys.format.intan_dat import (
        is_intan_dat_path, recording_name, read_rhd_header, get_dat_num_samples)
    date_string = recording_name(rhd_file).split('__')[-1]
    start_time = datetime.datetime.strptime(date_string, "%y%m%d_%H%M%S")
    if is_intan_dat_path(rhd_file):
        # one-file-per-signal-type recording (info.rhd + *.dat)
        fs = read_rhd_header(rhd_file)['sample_rate']
        n_samples = get_dat_num_samples(rhd_file)
    else:
        from neo.rawio import IntanRawIO
        in_io = IntanRawIO(rhd_file,ignore_integrity_checks=True)
        in_io._parse_header()
        fs = 30000
        n_samples = in_io._max_sigs_length
    end = start_time + datetime.timedelta(seconds=int(n_samples/fs + 1))
    return (start_time, end)

def get_recording_start_end(rhd_folder, base_folder):
    # either a folder of .rhd files, or a "one file per signal type" folder
    # with info.rhd + .dat files (in which case info.rhd is the only .rhd)
    rhds = glob.glob(os.path.join(base_folder,rhd_folder,"*.rhd"))
    rhds_trunc = ['/'.join([rhd_folder,os.path.basename(x)]) for x in rhds]
    rec_start = None
    rec_end = None
    for rhd in rhds:
        s,e = get_start_end(rhd)
        if rec_start is None or s < rec_start:
            rec_start = s
        if rec_end is None or e > rec_end:
            rec_end = e
    return rec_start, rec_end, rhds_trunc

def get_pyop_start_end(trialdata_csv):
    date_string = trialdata_csv.split('_')[-2]
    start_time = datetime.datetime.strptime(date_string, "%Y%m%d%H%M%S")
    df = pd.read_csv(trialdata_csv)
    if len(df) == 0:
        return None
    end_time = datetime.datetime.fromisoformat(df['time'].iloc[-1])
    return start_time, end_time


@click.command("gen-filelists")
@click.argument("data_dir", type=click.Path(), default=os.getcwd())
def gen_filelists(data_dir=None):
    if data_dir is None:
        data_dir = os.getcwd()
    if not os.path.exists(data_dir):
        print("Data directory %s does not exist"%data_dir)
        return
    bird = os.path.basename(data_dir)
    rhd_start= "%s__"%bird
    rhd_folders = [os.path.basename(x) for x in glob.glob(os.path.join(data_dir,rhd_start+"*"))]

    operant_csvs = glob.glob(os.path.join(data_dir,'OperantExperiments','*_trialdata.csv'))
    playback_csvs = glob.glob(os.path.join(data_dir,'PlaybackExperiments','*_trialdata.csv'))

    experiments = dict()
    unused_csvs = []
    for rhd_folder in rhd_folders:
        st,end,rhds = get_recording_start_end(rhd_folder,data_dir)
        experiments[rhd_folder] = dict()
        experiments[rhd_folder]['start'] = st
        experiments[rhd_folder]['end'] = end
        experiments[rhd_folder]['rhds'] = rhds
        experiments[rhd_folder]['operant'] = []
        experiments[rhd_folder]['playback'] = []
        experiments[rhd_folder]['settings'] = '/'.join([rhd_folder,'settings.xml'])
        print("Recording %s"%rhd_folder)
        print("Start: %s"%st)
        print("End: %s"%end)

    for csv in operant_csvs:
        ret = get_pyop_start_end(csv)
        if ret is None:
            continue
        s,e = ret
        added = False
        for exp in experiments:
            if s > experiments[exp]['start'] and s < experiments[exp]['end']:
                experiments[exp]['operant'].append('/'.join(['OperantExperiments',os.path.basename(csv).split('_trialdata')[0]]))
                added=True
                break
        if not added:
            unused_csvs.append([csv,e-s])
    for csv in playback_csvs:
        ret = get_pyop_start_end(csv)
        if ret is None:
            continue
        s,e = ret
        added=False
        for exp in experiments:
            if s > experiments[exp]['start'] and e < experiments[exp]['end']:
                experiments[exp]['playback'].append('/'.join(['PlaybackExperiments',os.path.basename(csv).split('_trialdata')[0]]))
                added=True
                break
        if not added:
            unused_csvs.append([csv,e-s])
    if len(unused_csvs) > 0:
        print("-----------------------")
        print("There are unused csvs, this is usually fine, but you want to make sure they are not important")
        print("Unused csvs:")
        for csv, dur in unused_csvs:
            print("%s \t Duration: %s"%(csv, dur))
        print("-----------------------")

    for experiment in experiments:
        lines = []
        lines.append(experiments[experiment]['settings'])
        lines.extend(experiments[experiment]['rhds'])
        for pyop in experiments[experiment]['playback'] + experiments[experiment]['operant']:
            lines.append(pyop+'_trialdata.csv')
            lines.append(pyop+'_events.log')
            lines.append(pyop+'_config.yaml')
        os.makedirs(os.path.join(data_dir,'filelists'),exist_ok=True)
        with open(os.path.join(data_dir,'filelists','%s.txt'%experiment),'w') as f:
            for line in lines:
                f.write(line+'\n')
        print(experiment)


def parse_settings_file(settings_path):
    import xml.etree.ElementTree as ET
    tree = ET.parse(settings_path)
    enabled_channels = dict({
        "Amplifier":[],
        "Analog": [],
        "Digital": [],
        "AUX": []
    })
    # bit of each enabled digital input in the packed digital word (DIGITAL-IN-09 -> bit 8)
    digital_bits = []
    # Intan native names (e.g. ANALOG-IN-2, DIGITAL-IN-09), parallel to "Analog"/"Digital"
    native_names = {"Analog": [], "Digital": []}
    for child in tree.getroot():
        if child.tag == "SignalGroup":
            if child.attrib['Prefix'] in ['A','B','C','D']:
                for chan in child:
                    if chan.attrib['Enabled'] == 'True':
                        name = chan.attrib['CustomChannelName']
                        if 'AUX' in name:
                            enabled_channels['AUX'].append(name)
                        else:
                            enabled_channels['Amplifier'].append(name)
                continue
            elif child.attrib['Prefix'] == 'ANALOG-IN':
                for chan in child:
                    if chan.attrib['Enabled'] == 'True':
                        enabled_channels['Analog'].append(chan.attrib['CustomChannelName'])
                        native_names['Analog'].append(chan.attrib.get('NativeChannelName', chan.attrib['CustomChannelName']))
                continue
            elif child.attrib['Prefix'] == 'DIGITAL-IN':
                for chan in child:
                    if chan.attrib['Enabled'] == 'True':
                        enabled_channels['Digital'].append(chan.attrib['CustomChannelName'])
                        native_name = chan.attrib.get('NativeChannelName', chan.attrib['CustomChannelName'])
                        native_names['Digital'].append(native_name)
                        digital_bits.append(int(native_name.split('-')[-1]) - 1)
                continue
            elif child.attrib['Prefix'] == 'DIGITAL-OUT':
                continue
    for chan_type in enabled_channels.keys():
        print("Number of %s channels:\t %d"%(chan_type,len(enabled_channels[chan_type])))
    print(enabled_channels)
    enabled_channels['DigitalBits'] = digital_bits
    enabled_channels['AnalogNative'] = native_names['Analog']
    enabled_channels['DigitalNative'] = native_names['Digital']
    return enabled_channels


CHANNELS_FILENAME = "channels.csv"

def load_channel_metadata(channels_csv):
    """Read a recording's channels.csv, keyed by Intan native channel name

    channels.csv lives in the recording folder (next to settings.xml / info.rhd)
    and has the columns: intan_channel,name,type,ni_line,description
    where intan_channel is the Intan native name, e.g. ANALOG-IN-2 or DIGITAL-IN-09.
    Returns {native_name: {"name": ..., "type": ..., "ni_line": ..., "description": ...}}
    """
    import csv
    metadata = dict()
    # utf-8-sig so files saved from Excel (with a BOM) still work
    with open(channels_csv, newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            row = {(k or '').strip().lower(): (v or '').strip() for k, v in row.items()}
            native = row.get('intan_channel', '').upper()
            if not native:
                continue
            if native in metadata:
                raise ValueError("%s is listed twice in %s" % (native, channels_csv))
            metadata[native] = {k: row.get(k, '') for k in ['name', 'type', 'ni_line', 'description']}
    return metadata


def apply_channel_metadata(native_names, current_names, metadata):
    """Replace channel names with the names from channels.csv where one is given"""
    names = []
    for native, current in zip(native_names, current_names):
        name = metadata.get(native.upper(), {}).get('name', '')
        names.append(name if name else current)
    return names

@click.command("read-rhd-settings")
@click.argument("settings_path")
def read_rhd_settings(settings_path):
    parse_settings_file(settings_path)

cli.add_command(info)
cli.add_command(rhd_header)
cli.add_command(wav_info)
cli.add_command(create_site)
cli.add_command(create_config)
cli.add_command(create_remote_config)
cli.add_command(list_sites)
cli.add_command(unittest)
cli.add_command(download_site)
cli.add_command(gensinglesitefiles)
cli.add_command(sortsinglesite)
cli.add_command(addsortedunits)
cli.add_command(gennwb)
cli.add_command(add_ripples)
cli.add_command(gen_filelists)
cli.add_command(read_rhd_settings)
cli.add_command(runpipelinejob)
cli.add_command(update_filelist)
# cli.add_command(rhd)

if __name__ == "__main__":
    cli()
