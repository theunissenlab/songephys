"""Figures of stimulation artefacts: the raw data and what was written to the binary

For each stimulation bout, and for each manual blank window (stim_artifact
``extra_blank_s``), a figure compares

* the raw amplifier data, before any blanking (full scale, and zoomed in), and
* the same samples in the ephys binary given to Kilosort (blanked, tail
  subtracted, band-passed, common-average referenced).

Pure numpy/matplotlib so it can be tested without SpikeInterface or luigi; the
pipeline wrapper is ``songephys.pipeline.file_setup.PlotStimArtifacts``.
"""
import os

import numpy as np

CHANNEL_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00")
PULSE_COLOR, MEAN_COLOR, BLANK_COLOR = "0.6", "#0072B2", "0.9"


def pick_columns(n_columns, n_pick):
    """``n_pick`` binary columns spread evenly over the probe"""
    n_pick = max(1, min(int(n_pick), int(n_columns)))
    return np.unique(np.round(np.linspace(0, n_columns - 1, n_pick)).astype(int))


def pick_pulses(index, max_pulses):
    """At most ``max_pulses`` of ``index``, evenly spaced"""
    index = np.asarray(index)
    if len(index) <= max_pulses:
        return index
    return index[np.unique(np.round(np.linspace(0, len(index) - 1, max_pulses)).astype(int))]


def _new_figure(nrows, ncols, width, height):
    # Figure + Agg canvas directly: no pyplot, so the caller's backend is untouched
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(width, height))
    FigureCanvasAgg(fig)
    axes = fig.subplots(nrows, ncols, squeeze=False)
    for ax in axes.ravel():
        ax.tick_params(labelsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    return fig, axes


def _sym_limit(values, floor):
    v = np.abs(np.asarray(values, dtype=float))
    v = v[np.isfinite(v)]
    return float(max(floor, 1.3 * np.percentile(v, 99.9))) if len(v) else float(floor)


def read_pulse_snippets(read, frames, pre, post, n_samples):
    """(n_pulses, pre + post, C) float array around each frame; NaN where outside the recording"""
    out = None
    for i, f in enumerate(np.asarray(frames, dtype=np.int64)):
        a, b = max(0, f - pre), min(n_samples, f + post)
        x = np.asarray(read(a, b), dtype=np.float64)
        if out is None:
            out = np.full((len(frames), pre + post, x.shape[1]), np.nan)
        out[i, a - (f - pre):a - (f - pre) + len(x)] = x
    return out


def plot_bout_figure(path, raw, binary, t_ms, blank_ms, channel_names, title, baseline_ms=(-2.0, -0.6)):
    """One figure for a bout: rows = channels; columns = raw, raw zoomed, binary

    raw, binary : (n_pulses, T, C) in microvolts, aligned to the pulse at t_ms = 0
    blank_ms : (start, stop) of the nominal blank window, shaded
    """
    n_ch = raw.shape[2]
    base = (t_ms >= baseline_ms[0]) & (t_ms < baseline_ms[1])
    if not base.any():
        base = t_ms < 0
    raw = raw - np.nanmedian(raw[:, base], axis=1, keepdims=True)
    outside = (t_ms < blank_ms[0]) | (t_ms >= blank_ms[1])
    fig, axes = _new_figure(n_ch, 3, 11.5, 1.9 * n_ch + 1.1)
    heads = ("Raw, before blanking (full scale)", "Raw, before blanking (zoomed)",
             "Written to the binary (blanked, band-passed, referenced)")
    for r in range(n_ch):
        for c, (data, unit, scale) in enumerate(((raw, "mV", 1e-3), (raw, "uV", 1.0), (binary, "uV", 1.0))):
            ax = axes[r, c]
            ax.axvspan(blank_ms[0], blank_ms[1], color=BLANK_COLOR, lw=0, zorder=0)
            y = data[:, :, r] * scale
            ax.plot(t_ms, y.T, color=PULSE_COLOR, lw=0.4, alpha=0.5)
            ax.plot(t_ms, np.nanmean(y, axis=0), color=MEAN_COLOR, lw=1.1)
            if c == 1:
                lim = _sym_limit(y[:, base], 300.0)
                ax.set_ylim(-lim, lim)
            elif c == 2:
                lim = _sym_limit(y[:, outside], 50.0)
                ax.set_ylim(-lim, lim)
            ax.set_xlim(t_ms[0], t_ms[-1])
            ax.set_ylabel("%s (%s)" % (channel_names[r], unit), fontsize=7.5)
            if r == 0:
                ax.set_title(heads[c], fontsize=8.5)
            if r == n_ch - 1:
                ax.set_xlabel("ms from stim TTL", fontsize=8)
    fig.suptitle(title + "\ngrey: individual pulses, blue: mean, shaded: blanked window",
                 fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=110)
    return path


def plot_window_figure(path, raw, binary, t_s, window_s, channel_names, title, zoom_ms=(-5.0, 30.0)):
    """One figure for a manual blank window: rows = raw / binary; columns = wide view / zoomed on the window
    (from zoom_ms[0] before its start to zoom_ms[1] after its end)

    raw, binary : (T, C) in microvolts; t_s : absolute time of each sample (s)
    """
    pre = t_s < window_s[0]
    raw = raw - np.median(raw[pre] if pre.any() else raw, axis=0)
    length_ms = (window_s[1] - window_s[0]) * 1e3
    zoom = (t_s >= window_s[0] + zoom_ms[0] / 1e3) & (t_s < window_s[1] + zoom_ms[1] / 1e3)
    outside = (t_s < window_s[0]) | (t_s >= window_s[1])
    fig, axes = _new_figure(2, 2, 11.5, 5.6)
    for r, (data, unit, scale, name) in enumerate((
            (raw, "mV", 1e-3, "Raw, before blanking"),
            (binary, "uV", 1.0, "Written to the binary (blanked, band-passed, referenced)"))):
        for c, (m, tt, xl) in enumerate(((slice(None), t_s, "s from start of recording"),
                                          (zoom, (t_s - window_s[0]) * 1e3, "ms from start of the blank"))):
            ax = axes[r, c]
            if c == 0:
                ax.axvspan(window_s[0], window_s[1], color=BLANK_COLOR, lw=0, zorder=0)
            else:
                ax.axvspan(0, length_ms, color=BLANK_COLOR, lw=0, zorder=0)
            for k in range(data.shape[1]):
                ax.plot(tt[m], data[m, k] * scale, color=CHANNEL_COLORS[k % len(CHANNEL_COLORS)], lw=0.6,
                        label=channel_names[k] if (r == 0 and c == 0) else None)
            if r == 1:
                lim = _sym_limit(data[outside], 50.0)
                ax.set_ylim(-lim, lim)
            ax.set_ylabel(unit, fontsize=8)
            ax.set_title(name + (" - zoomed" if c == 1 else ""), fontsize=8.5)
            ax.set_xlabel(xl, fontsize=8)
            ax.ticklabel_format(useOffset=False)
    axes[0, 0].legend(fontsize=7, frameon=False, ncol=len(channel_names), loc="upper right")
    fig.suptitle(title + "\nshaded: blanked window", fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=110)
    return path


def plot_stim_artifact_figures(read_raw, read_bin, raw_cols, fs, n_samples, events, out_dir, gain_uv=0.195,
                               channel_names=None, post_blank_ms=0.0, annotations=None, n_channels=4,
                               pre_ms=2.0, post_ms=10.0, max_pulses=40, window_margin_s=0.25):
    """Write one figure per stimulation bout and per manual blank window

    Parameters
    ----------
    read_raw : callable(start, stop, cols) -> (T, len(cols)) raw amplifier samples of raw columns ``cols``
    read_bin : callable(start, stop) -> (T, C) samples of the written binary
    raw_cols : raw column of each binary column (bad channels are not in the binary)
    events : stim events dict (``stim_artifact.build_stim_events``)
    gain_uv : microvolts per count, for both raw and binary
    channel_names : name of each binary column (default "ch <raw column>")
    post_blank_ms : stim_artifact blank_after_filter_ms, for the shaded window
    annotations : site_config['stim_bouts'], for the titles

    Returns the list of files written.
    """
    os.makedirs(out_dir, exist_ok=True)
    raw_cols = np.asarray(raw_cols, dtype=int)
    cols = pick_columns(len(raw_cols), n_channels)
    names = ["ch %d" % raw_cols[c] if channel_names is None else str(channel_names[c]) for c in cols]
    read_r = lambda a, b: np.asarray(read_raw(a, b, raw_cols[cols]), dtype=np.float64) * gain_uv
    read_b = lambda a, b: np.asarray(read_bin(a, b), dtype=np.float64)[:, cols] * gain_uv
    cfg = events.get("config", {})
    e = events["events"]
    start = np.asarray(e["start_frame"], dtype=np.int64)
    bout = np.asarray(e["bout"], dtype=np.int64)
    delivered = np.asarray(e["delivered"], dtype=bool)
    offset = int(round(float(cfg.get("offset_ms", 0.0)) * fs / 1e3))
    notes = {int(a["bout"]): a for a in (annotations or []) if "bout" in a}
    pre, post = int(round(pre_ms * fs / 1e3)), int(round(post_ms * fs / 1e3))
    t_ms = np.arange(-pre, post) / fs * 1e3
    blank_ms = (-float(cfg.get("ms_before", 0.5)), max(float(cfg.get("ms_after", 1.5)), float(post_blank_ms or 0)))
    written = []
    for b in np.unique(bout):
        idx = np.nonzero((bout == b) & delivered)[0]
        label = "delivered"
        if len(idx) == 0:                                  # nothing delivered: show the TTLs anyway
            idx, label = np.nonzero(bout == b)[0], "no artefact detected"
        use = pick_pulses(idx, max_pulses)
        frames = start[use] + offset
        raw = read_pulse_snippets(read_r, frames, pre, post, n_samples)
        binary = read_pulse_snippets(read_b, frames, pre, post, n_samples)
        a = notes.get(int(b), {})
        extra = ", ".join("%s %s" % (k, a[k]) for k in ("polarity", "current_ua") if k in a)
        t0 = start[idx[0]] / fs
        title = "Stim bout %d at %d:%02d:%02d - %d pulses (%s), %d shown%s" % (
            b, t0 // 3600, t0 % 3600 // 60, t0 % 60, len(idx), label, len(use), (" - " + extra) if extra else "")
        written.append(plot_bout_figure(os.path.join(out_dir, "stim_bout_%02d.png" % b),
                                        raw, binary, t_ms, blank_ms, names, title))
    margin = int(round(window_margin_s * fs))
    for k, (w0, w1) in enumerate(events.get("extra_blank_windows") or []):
        a, b = max(0, int(w0) - margin), min(n_samples, int(w1) + margin)
        title = "Manual blank window %d: %.3f-%.3f s (%.0f ms)" % (k, w0 / fs, w1 / fs, (w1 - w0) / fs * 1e3)
        written.append(plot_window_figure(os.path.join(out_dir, "extra_blank_%02d.png" % k),
                                          read_r(a, b), read_b(a, b), np.arange(a, b) / fs,
                                          (w0 / fs, w1 / fs), names, title))
    ea = events.get("extra_artifacts") or {}
    for k, (on, w0, w1) in enumerate(zip(ea.get("onset_frame", []), ea.get("blank_start_frame", []),
                                         ea.get("blank_stop_frame", []))):
        a, b = max(0, int(w0) - margin), min(n_samples, int(w1) + margin)
        title = ("Artefact without a TTL %d: jump at %.4f s - nominal blank %.1f ms before to %.1f ms after "
                 "(longer on channels that stay on the rail)" % (k, on / fs, (on - w0) / fs * 1e3, (w1 - on) / fs * 1e3))
        written.append(plot_window_figure(os.path.join(out_dir, "extra_artifact_%02d.png" % k),
                                          read_r(a, b), read_b(a, b), np.arange(a, b) / fs,
                                          (w0 / fs, w1 / fs), names, title))
    return written
