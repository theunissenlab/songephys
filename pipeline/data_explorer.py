"""
Browse files for convenience
"""

from __future__ import print_function

import sys
import numpy as np
from flask import Flask, render_template, jsonify, request
from flask import Flask, send_from_directory

import config

import datetime
import os
import glob

from block_io import load_yaml


app = Flask(__name__, template_folder=".")
app.config["TEMPLATE_AUTO_RELOAD"] = True


def format_datetime(value, format="full"):
    if value:
        if format == "full":
            return value.strftime("%H:%M:%S %m/%d/%Y")
        elif format == "date":
            return value.strftime("%m/%d/%Y")
    else:
        return ""


app.jinja_env.filters["datetime"] = format_datetime


def modified_at(filename):
    return datetime.datetime.fromtimestamp(os.stat(filename).st_mtime)


def check_downloaded(session):
    matches = glob.glob(os.path.join(session, "raw", "*.rhd"))
    if not len(matches):
        return None
    else:
        return modified_at(matches[0])


def check_segmented(session):
    matches = glob.glob(os.path.join(session, "segmentation", "*-segmented.pkl"))
    if not len(matches):
        return None
    else:
        return modified_at(matches[0])
    

def check_filtered(session):
    matches = glob.glob(os.path.join(session, "segmentation", "*-filtered.pkl"))
    if not len(matches):
        return None
    else:
        return modified_at(matches[0])

'''
def check_sorted(session):
    matches = glob.glob(os.path.join(session, "spikes", "*-sorted.pkl"))
    if not len(matches):
        return None
    else:
        return modified_at(matches[0])
'''
def check_sorted(site):
    matches = glob.glob(os.path.join(site, "sorted.pkl"))
    if not len(matches):
        return None
    else:
        return modified_at(matches[0])


def check_summaries(session):
    matches = glob.glob(os.path.join(session, "unit_summaries", "*.png"))
    return len(matches)


def fetch_files():
    birddir = os.path.join(config.DATADIR, "birds", "*")

    site_files = {}

    for bird in sorted(glob.glob(birddir)):
        for site_dir in sorted(glob.glob(os.path.join(bird, "sites", "*_*_*"))):
            try:
                session_info = load_yaml(os.path.join(site_dir, "sessions.yaml"))
            except:
                print("Can't load", os.path.join(site_dir, "sessions.yaml"))
                continue
            sessions = [s["id"] for s in session_info["sessions"]]

            # sessions = glob.glob(os.path.join(birddir, "*__*_*"))

            site_files[os.path.basename(site_dir)] = []
            for session_name in sorted(sessions):
                session = os.path.join(bird, "sessions", session_name)
                file_info = {}

                file_parts = filter(None, session_name.split("_"))
                bird_name, date, time = file_parts

                date = "{}/{}/{}".format(date[2:4], date[4:], date[:2])
                time = "{}:{}".format(time[:2], time[2:4])

                file_info["str"] = "{}, {}, {}".format(bird_name, date, time)
                file_info["downloaded_date"] = check_downloaded(session)
                file_info["segmented_date"] = check_segmented(session)
                file_info["filtered_date"] = check_filtered(session)
                file_info["sorted_date"] = check_sorted(site_dir)
                file_info["path"] = session
                file_info["units"] = check_summaries(site_dir)

                try:
                    with open(os.path.join(session, "notes")) as notefile:
                        file_info["notes"] = notefile.read()
                except IOError:
                    file_info["notes"] = ""

                site_files[os.path.basename(site_dir)].append(file_info)

    return site_files


@app.route("/")
def index():
    files = fetch_files()
    return render_template("data_explorer.html", files=files, datetime=datetime)


app.run("0.0.0.0")
