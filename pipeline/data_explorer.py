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
    

def check_filtered(session):
    matches = glob.glob(os.path.join(session, "segmentation", "*-filtered.pkl"))
    if not len(matches):
        return None
    else:
        return modified_at(matches[0])

def check_sorted(session):
    matches = glob.glob(os.path.join(session, "spikes", "*-sorted.pkl"))
    if not len(matches):
        return None
    else:
        return modified_at(matches[0])


def fetch_files():
    birddir = os.path.join(config.DATADIR, "birds", "*")
    sessions = glob.glob(os.path.join(birddir, "*__*_*"))

    files = []
    for session in sorted(sessions):
        file_info = {}
        file_parts = filter(None, os.path.basename(session).split("_"))
        bird, date, time = file_parts

        date = "{}/{}/{}".format(date[2:4], date[4:], date[:2])
        time = "{}:{}".format(time[:2], time[2:4])

        file_info["str"] = "{}, {}, {}".format(bird, date, time)
        file_info["filtered_date"] = check_filtered(session)
        file_info["sorted_date"] = check_sorted(session)
        file_info["path"] = session
        try:
            with open(os.path.join(session, "notes")) as notefile:
                file_info["notes"] = notefile.read()
        except IOError:
            file_info["notes"] = ""

        files.append(file_info)
    
    return files


@app.route("/")
def index():
    files = fetch_files()
    return render_template("data_explorer.html", files=files)


app.run("0.0.0.0")
