from __future__ import print_function

import sys
import numpy as np
from flask import Flask, render_template, jsonify, request
from flask import Flask, send_from_directory

from spikes.sorting.metrics import *
from spikes.sorting.autocluster.process_block import *

import config



def format_data(time_ranges, data_blocked, labels, metric_blocks, figure_locations):
    entries = []
    for t, d, l, m, figs in zip(time_ranges, data_blocked, labels, metric_blocks, figure_locations):
        entry = {
            "t_start": t[0],
            "t_stop": t[1],
            "scatter": figs.get("scatter"),
            "clusters": [
                {
                    "label": str(c),
                    "waveforms": figs[c],
                    "selected": False,
                    "active": False
                }
                for c in sorted(np.unique(l))],
            "labels": list(map(str, sorted(np.unique(l))))
        }

        entries.append(entry)
    return entries


def create_server(data_info_file, output_file):
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config["TEMPLATE_AUTO_RELOAD"] = True
    @app.after_request
    def add_header(r):
        """
        Add headers to both force latest IE rendering engine or Chrome Frame,
        and also to cache the rendered page for 10 minutes.
        """
        r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        r.headers["Pragma"] = "no-cache"
        r.headers["Expires"] = "0"
        r.headers['Cache-Control'] = 'public, max-age=0'
        return r

    data_dir = os.path.join(os.path.dirname(config.DATADIR), os.path.dirname(data_info_file))
    print(data_dir)
    data = np.load(data_info_file)[()]
    entries = format_data(
            data["time_ranges"],
            data["data"],
            data["labels"],
            data["metrics"],
            data["figures"])

    @app.route("/images/<path:filename>")
    def img_static(filename):
        return send_from_directory(data_dir, filename)

    @app.route("/")
    def index():
        return render_template("index.html")

    def shutdown_server():
        func = request.environ.get('werkzeug.server.shutdown')
        if func is None:
            raise RuntimeError('Not running with the Werkzeug Server')
        func()

    @app.route('/shutdown')
    def shutdown():
        shutdown_server()
        return 'Server shutting down...'

    @app.route('/data')
    def give_data():
        return jsonify(entries)

    @app.route('/submit', methods=["POST"])
    def save():
        save_data = request.get_json(silent=True)

        result = []

        for unit_idx, unit in enumerate(save_data["units"]):
            unit_times = None
            unit_wfs = None
            for time_block_idx, cluster_ids in enumerate(unit):
                cluster_ids = list(map(int, cluster_ids))

                dat = data["data"][time_block_idx]
                lab = data["labels"][time_block_idx]

                # don't double count any spikes
                if time_block_idx != 0:
                    filter_times = (
                        dat["times"] >= data["time_ranges"][time_block_idx - 1][1]
                    )
                else:
                    filter_times = np.ones_like(dat["times"]).astype(bool)
                lab = lab[filter_times]

                for cluster_id in cluster_ids:
                    if unit_wfs is None:
                        if dat["waveforms"][filter_times][lab == cluster_id].shape[0] > 1:
                            unit_wfs = dat["waveforms"][filter_times][lab == cluster_id]
                            unit_times = dat["times"][filter_times][lab == cluster_id]
                    else:
                        unit_wfs = np.concatenate([unit_wfs, dat["waveforms"][filter_times][lab == cluster_id]], axis=0)
                        unit_times = np.concatenate([unit_times, dat["times"][filter_times][lab == cluster_id]], axis=0)

            sorter = np.argsort(unit_times)
            unit_times = unit_times[sorter]
            unit_wfs = unit_wfs[sorter]
            
            result.append({"times": unit_times, "waveforms": unit_wfs})

        np.save(output_file, result)

        return jsonify({"sucesss": True})

    return app


if __name__ == "__main__":
    app = create_server(sys.argv[1], sys.argv[2])
    app.run(host="0.0.0.0", port=8111, debug=True)
