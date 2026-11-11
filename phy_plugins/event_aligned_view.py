"""phy plugin: raster + PSTH of the selected clusters aligned to event times.

Install
-------
In ``~/.phy/phy_config.py``::

    c.Plugins.dirs = [r'G:\\Code\\songephys\\phy_plugins']
    c.TemplateGUI.plugins = ['EventAlignedPlugin']

Then in phy: View > New view > EventAlignedView. phy remembers the layout.

Event sources (all are optional, found automatically next to params.py)
-------------------------------------------------------------------------
* ``events.csv`` in the sort folder, written by ``song sort-singlesite`` after
  sorting (ExportSortEvents; rerun it on a finished sort to write just this file).
  When present it replaces the two
  site sources below: stim sets (delivered, no artefact, per polarity, per bout),
  target (+ no feedback / with feedback), feedback, song, syllable (+ per code), ...
* songephys stim events: ``<site>/raw/stim_events_*.json`` (the phy folder is
  ``<site>/<sort dir>``). Gives "stim: delivered", "stim: TTL, no artefact",
  one set per polarity if ``site_config.yaml`` has ``stim_bouts`` with
  ``polarity``, and one set per bout. The blanked window is shaded.
* Digital TTL lines from ``<site>/raw/digital_*.bin``, named in ``site_config.yaml``
  (``digital_channel_names`` / ``digital_channel_bits``): "Feedback" -> "feedback",
  "Match" (or "Target") -> "target", plus "target: no feedback" / "target: with
  feedback" (feedback onset within 100 ms after the target). Rising edges are
  cached in ``<phy dir>/.phy/event_aligned_digital.npz``.
* Any ``events*.npy`` / ``events*.txt`` / ``events*.csv`` / ``events*.tsv`` in
  the phy folder. One event set per file. Float values are seconds, integer
  values are sample indices (divided by the sample rate). For text files the
  first numeric column of each line is used; header lines are skipped.

Snippets (type in the phy GUI, positive numbers only: phy reads ``-`` as a range)
----------------------------------------------------------------------------------
* ``:ew 50 100``  window = 50 ms before to 100 ms after the event
* ``:eb 1``       PSTH bin = 1 ms
* ``:es 2``       use event set #2 (the list is printed in the phy console and
  is in the view menu under "Align to")
* ``:er``         reload event files
* ``:eg 12``      centre the TraceView on event #12 of the current set

TraceView link
--------------
* Click a raster tick: the TraceView centres on that trial at the clicked time.
* ``alt+]`` / ``alt+[``: TraceView to the next / previous event of the current set.
* The TraceView shows the current event set as yellow lines, the blanked window
  edges as grey lines, and TTLs without a stim artefact as red lines. Toggle with
  the TraceView menu item "Toggle event markers".
"""

import csv
import glob
import json
import logging
import os

import numpy as np

logger = logging.getLogger('phy')

EVENT_FILE_PATTERNS = ('events*.npy', 'events*.txt', 'events*.csv', 'events*.tsv')


# -----------------------------------------------------------------------------
# Pure functions (no Qt), testable on their own
# -----------------------------------------------------------------------------

class EventSet:
    """A named set of event times in seconds, with an optional shaded window (ms)."""

    def __init__(self, name, times, shade_ms=None):
        self.name = name
        self.times = np.sort(np.asarray(times, dtype=np.float64).ravel())
        self.shade_ms = shade_ms

    def __repr__(self):
        return 'EventSet(%r, n=%d)' % (self.name, len(self.times))


def _times_from_array(arr, sample_rate):
    arr = np.asarray(arr).ravel()
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.float64) / float(sample_rate)
    return arr.astype(np.float64)


def read_text_events(path):
    """First numeric value of each line of a txt/csv/tsv file; non-numeric lines skipped."""
    times = []
    is_int = True
    with open(path, newline='') as f:
        sample = f.read(4096)
        f.seek(0)
        delim = ',' if ',' in sample else ('\t' if '\t' in sample else None)
        rows = csv.reader(f, delimiter=delim) if delim else (line.split() for line in f)
        for row in rows:
            for tok in row:
                tok = tok.strip()
                if not tok:
                    continue
                try:
                    v = float(tok)
                except ValueError:
                    continue
                is_int &= ('.' not in tok and 'e' not in tok.lower())
                times.append(v)
                break
    arr = np.array(times, dtype=np.int64 if (is_int and times) else np.float64)
    return arr


def load_event_files(phy_dir, sample_rate):
    """Event sets from events*.npy/txt/csv/tsv files in the phy folder."""
    sets = []
    paths = sorted({p for pat in EVENT_FILE_PATTERNS for p in glob.glob(os.path.join(phy_dir, pat))})
    for path in paths:
        if path.endswith('.csv') and read_event_table(path) is not None:
            continue                     # an events table, handled by event_table_sets
        try:
            if path.endswith('.npy'):
                arr = np.load(path)
            else:
                arr = read_text_events(path)
            times = _times_from_array(arr, sample_rate)
            if len(times):
                sets.append(EventSet(os.path.splitext(os.path.basename(path))[0], times))
        except Exception as e:  # keep phy running if one file is bad
            logger.warning('EventAlignedView: could not read %s: %s', path, e)
    return sets


def _read_stim_bout_annotations(site_dir):
    """{bout: {'polarity': ..., 'current_ua': ...}} from site_config.yaml, or {}."""
    path = os.path.join(site_dir, 'site_config.yaml')
    if not os.path.exists(path):
        return {}
    try:
        import yaml
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning('EventAlignedView: could not read %s: %s', path, e)
        return {}
    out = {}
    for row in cfg.get('stim_bouts') or []:
        if isinstance(row, dict) and 'bout' in row:
            out[int(row['bout'])] = row
    return out


def stim_event_sets(events_json, bout_annotations=None):
    """Event sets from a songephys stim_events JSON dict (see stim_artifact.build_stim_events)."""
    if not events_json or not events_json.get('enabled', True):
        return []
    ev = events_json.get('events') or {}
    start = np.asarray(ev.get('start_frame', []), dtype=np.int64)
    if not len(start):
        return []
    fs = float(events_json['sampling_rate'])
    n = len(start)
    delivered = np.asarray(ev.get('delivered', [True] * n), dtype=bool)
    bout = np.asarray(ev.get('bout', [-1] * n), dtype=np.int64)
    t = start / fs

    shade = None
    if 'blank_start_frame' in ev and 'blank_stop_frame' in ev:
        blanked = np.asarray(ev.get('blanked', [True] * n), dtype=bool)
        bs = np.asarray(ev['blank_start_frame'], dtype=np.float64)
        be = np.asarray(ev['blank_stop_frame'], dtype=np.float64)
        ok = blanked & np.isfinite(bs) & np.isfinite(be)
        if ok.any():
            shade = (float(np.median(bs[ok] - start[ok])) / fs * 1e3,
                     float(np.median(be[ok] - start[ok])) / fs * 1e3)

    sets = [EventSet('stim: delivered', t[delivered], shade)]
    if (~delivered).any():
        sets.append(EventSet('stim: TTL, no artefact', t[~delivered]))

    ann = bout_annotations or {}
    polarities = {}
    for b, row in ann.items():
        pol = row.get('polarity')
        if pol is not None:
            polarities.setdefault(str(pol), []).append(b)
    for pol, bouts in sorted(polarities.items()):
        m = delivered & np.isin(bout, bouts)
        if m.any():
            sets.append(EventSet('stim: polarity %s' % pol, t[m], shade))

    for b in np.unique(bout[bout >= 0]):
        m = delivered & (bout == b)
        if not m.any():
            continue
        row = ann.get(int(b), {})
        label = 'stim: bout %d' % b
        extra = ' '.join(str(row[k]) + u for k, u in (('polarity', ''), ('current_ua', ' uA')) if k in row)
        if extra:
            label += ' (%s)' % extra
        sets.append(EventSet(label, t[m], shade))
    return sets


def load_stim_event_sets(phy_dir):
    """Find <site>/raw/stim_events_*.json for a phy folder <site>/<sort dir>."""
    site_dir = os.path.dirname(os.path.abspath(phy_dir))
    paths = sorted(glob.glob(os.path.join(site_dir, 'raw', 'stim_events_*.json')))
    sets = []
    for path in paths:
        try:
            with open(path) as f:
                sets += stim_event_sets(json.load(f), _read_stim_bout_annotations(site_dir))
        except Exception as e:
            logger.warning('EventAlignedView: could not read %s: %s', path, e)
    return sets


# Digital TTL lines turned into event sets: (event-set prefix, accepted channel names).
DIGITAL_EVENT_LINES = (
    ('feedback', ('feedback',)),
    ('target', ('match', 'target')),
)
# A target counts as "given feedback" if a feedback pulse starts within this time after it.
FEEDBACK_WINDOW_S = 0.1


def scan_digital_rising(path, bits, chunk=30_000_000):
    """Rising-edge sample indices of each bit in a uint16 digital memmap. {bit: array}."""
    d = np.memmap(path, dtype=np.uint16, mode='r')
    out = {b: [] for b in bits}
    if not len(d):
        return {b: np.zeros(0, np.int64) for b in bits}
    prev = {b: (int(d[0]) >> b) & 1 for b in bits}  # a line already high at 0 is not an edge
    for s in range(0, len(d), chunk):
        x = np.asarray(d[s:s + chunk])
        for b in bits:
            v = ((x >> b) & 1).astype(np.int8)
            dv = np.diff(np.concatenate(([prev[b]], v)))
            out[b].append(np.nonzero(dv == 1)[0].astype(np.int64) + s)
            prev[b] = int(v[-1])
    return {b: np.concatenate(v) for b, v in out.items()}


def _digital_bits_by_name(site_config):
    """{lowercase channel name: bit} from site_config digital_channel_names/bits."""
    names = site_config.get('digital_channel_names') or []
    bits = site_config.get('digital_channel_bits') or list(range(len(names)))
    return {str(n).strip().lower(): int(b) for n, b in zip(names, bits) if n}


def target_feedback_sets(feedback_s, target_s, window_s=FEEDBACK_WINDOW_S):
    """Event sets from feedback and target onset times (seconds)."""
    sets = []
    fb = np.sort(np.asarray(feedback_s, np.float64)) if feedback_s is not None else None
    tg = np.sort(np.asarray(target_s, np.float64)) if target_s is not None else None
    if fb is not None and len(fb):
        sets.append(EventSet('feedback', fb))
    if tg is not None and len(tg):
        sets.append(EventSet('target', tg))
        if fb is not None:
            j = np.searchsorted(fb, tg)
            nxt = np.where(j < len(fb), fb[np.clip(j, None, max(len(fb) - 1, 0))] - tg, np.inf) \
                if len(fb) else np.full(len(tg), np.inf)
            given = nxt <= window_s
            sets.append(EventSet('target: no feedback', tg[~given]))
            sets.append(EventSet('target: with feedback', tg[given]))
    return [s for s in sets if len(s.times)]


def load_digital_event_sets(phy_dir):
    """feedback / target sets from <site>/raw/digital_*.bin, cached in <phy_dir>/.phy."""
    site_dir = os.path.dirname(os.path.abspath(phy_dir))
    bins = sorted(glob.glob(os.path.join(site_dir, 'raw', 'digital_*.bin')))
    cfg_path = os.path.join(site_dir, 'site_config.yaml')
    if not bins or not os.path.exists(cfg_path):
        return []
    try:
        import yaml
        with open(cfg_path) as f:
            by_name = _digital_bits_by_name(yaml.safe_load(f) or {})
        lines = {}
        for label, aliases in DIGITAL_EVENT_LINES:
            for a in aliases:
                if a in by_name:
                    lines[label] = by_name[a]
                    break
        if not lines:
            return []
        path = bins[0]
        info_path = path[:-4] + '_info.json'
        fs = 30000.0
        if os.path.exists(info_path):
            with open(info_path) as f:
                fs = float(json.load(f).get('sampling_rate', fs))
        st = os.stat(path)
        key = '%s|%d|%d|%s' % (os.path.basename(path), st.st_size, int(st.st_mtime),
                               sorted(lines.items()))
        cache = os.path.join(phy_dir, '.phy', 'event_aligned_digital.npz')
        edges = None
        if os.path.exists(cache):
            try:
                z = np.load(cache, allow_pickle=False)
                if str(z['key']) == key:
                    edges = {lab: z[lab] for lab in lines}
            except Exception:
                edges = None
        if edges is None:
            logger.info('EventAlignedView: scanning %s for %s (once, then cached)...',
                        os.path.basename(path), ', '.join(lines))
            by_bit = scan_digital_rising(path, sorted(set(lines.values())))
            edges = {lab: by_bit[b] for lab, b in lines.items()}
            try:
                os.makedirs(os.path.dirname(cache), exist_ok=True)
                np.savez(cache, key=key, **edges)
            except Exception as e:
                logger.warning('EventAlignedView: could not write cache %s: %s', cache, e)
        to_s = {lab: e / fs for lab, e in edges.items()}
        return target_feedback_sets(to_s.get('feedback'), to_s.get('target'))
    except Exception as e:
        logger.warning('EventAlignedView: could not read digital events: %s', e)
        return []


EVENT_TABLE = 'events.csv'


def read_event_table(path):
    """Rows of a songephys events table (time_s, event, ... columns), or None if the
    file is not one (no 'time_s' and 'event' header)."""
    try:
        with open(path, newline='') as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return None
    if not rows or 'time_s' not in rows[0] or 'event' not in rows[0]:
        return None
    return rows


def _f(v, default=np.nan):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _b(v):
    return str(v).strip().lower() in ('true', '1', 'yes')


def event_table_sets(rows):
    """Event sets from an events table (see songephys.postprocess.events_table)."""
    by = {}
    for r in rows:
        by.setdefault(r['event'], []).append(r)
    sets = []
    stim = by.pop('stim', [])
    if stim:
        t = np.array([_f(r['time_s']) for r in stim])
        delivered = np.array([_b(r.get('delivered', 'True')) for r in stim])
        pol = np.array([str(r.get('polarity', '')) for r in stim])
        bout = np.array([int(_f(r.get('bout'), -1)) for r in stim])
        cur = np.array([_f(r.get('current_ua')) for r in stim])
        bs = np.array([_f(r.get('blank_start_ms')) for r in stim])
        be = np.array([_f(r.get('blank_stop_ms')) for r in stim])
        ok = delivered & np.isfinite(bs) & np.isfinite(be)
        shade = (float(np.median(bs[ok])), float(np.median(be[ok]))) if ok.any() else None
        sets.append(EventSet('stim: delivered', t[delivered], shade))
        if (~delivered).any():
            sets.append(EventSet('stim: TTL, no artefact', t[~delivered]))
        for p in sorted(set(pol[delivered]) - {'', 'unknown', 'nan'}):
            sets.append(EventSet('stim: polarity %s' % p, t[delivered & (pol == p)], shade))
        for b in sorted(set(bout[delivered & (bout >= 0)])):
            m = delivered & (bout == b)
            extra = ' '.join(x for x in (pol[m][0] if pol[m][0] not in ('unknown', 'nan') else '',
                                         '%g uA' % cur[m][0] if np.isfinite(cur[m][0]) else '') if x)
            sets.append(EventSet('stim: bout %d' % b + (' (%s)' % extra if extra else ''), t[m], shade))
    target = by.pop('target', [])
    if target:
        t = np.array([_f(r['time_s']) for r in target])
        sets.append(EventSet('target', t))
        if 'with_feedback' in target[0]:
            wf = np.array([_b(r.get('with_feedback')) for r in target])
            sets.append(EventSet('target: no feedback', t[~wf]))
            sets.append(EventSet('target: with feedback', t[wf]))
    for name in sorted(by):
        rs = by[name]
        t = np.array([_f(r['time_s']) for r in rs])
        sets.append(EventSet(name, t))
        if name == 'syllable' and 'code' in rs[0]:
            code = np.array([int(_f(r.get('code'), -1)) for r in rs])
            for c in sorted(set(code) - {-1, 0}):
                sets.append(EventSet('syllable: code %d' % c, t[code == c]))
    return [s for s in sets if len(s.times)]


def load_all_event_sets(phy_dir, sample_rate):
    """events.csv in the sort folder (ExportSortEvents) if present; otherwise the
    site's raw stim events / digital lines. Plus any other events* files."""
    table = os.path.join(phy_dir, EVENT_TABLE)
    rows = read_event_table(table) if os.path.exists(table) else None
    if rows is not None:
        primary = event_table_sets(rows)
    else:
        primary = load_stim_event_sets(phy_dir) + load_digital_event_sets(phy_dir)
    return primary + load_event_files(phy_dir, sample_rate)


def align_spikes(spike_times, event_times, pre_s, post_s):
    """Spike times relative to each event, within [-pre_s, post_s].

    spike_times must be sorted (seconds). Returns (rel_times_s, event_index).
    """
    spike_times = np.asarray(spike_times, dtype=np.float64)
    event_times = np.asarray(event_times, dtype=np.float64)
    lo = np.searchsorted(spike_times, event_times - pre_s, side='left')
    hi = np.searchsorted(spike_times, event_times + post_s, side='right')
    counts = hi - lo
    if counts.sum() == 0:
        return np.zeros(0), np.zeros(0, dtype=np.int64)
    trial = np.repeat(np.arange(len(event_times)), counts)
    # index of each spike: lo[trial] + position within its trial
    offs = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    idx = np.repeat(lo, counts) + offs
    rel = spike_times[idx] - event_times[trial]
    return rel, trial


def psth(rel_times_s, n_events, pre_s, post_s, bin_s):
    """Firing rate (Hz) histogram. Returns (bin_edges_s, rate_hz)."""
    n_bins = max(1, int(round((pre_s + post_s) / bin_s)))
    edges = np.linspace(-pre_s, post_s, n_bins + 1)
    counts, _ = np.histogram(rel_times_s, bins=edges)
    width = edges[1] - edges[0]
    rate = counts / (max(n_events, 1) * width)
    return edges, rate


# -----------------------------------------------------------------------------
# phy view and plugin
# -----------------------------------------------------------------------------

def events_in_interval(times, t0, t1):
    """Event times within [t0, t1] (times sorted)."""
    times = np.asarray(times, dtype=np.float64)
    return times[np.searchsorted(times, t0, 'left'):np.searchsorted(times, t1, 'right')]


def marker_lines(markers, t0, t1):
    """Vertical line segments for the trace view.

    markers: list of (times_s, offset_ms, rgba). Returns (pos (n, 4), colors (n, 4)) with
    pos rows (t, -1, t, 1), or (None, None) if nothing falls in [t0, t1].
    """
    pos, col = [], []
    for times, offset_ms, rgba in markers:
        t = np.asarray(times, dtype=np.float64) + offset_ms / 1e3
        t = events_in_interval(t, t0, t1)
        for x in t:
            pos.append((x, -1.0, x, 1.0))
            col.append(rgba)
    if not pos:
        return None, None
    return np.array(pos, dtype=np.float64), np.array(col, dtype=np.float64)


def neighbour_event(times, t, direction):
    """Index of the next (direction=+1) or previous (-1) event strictly after/before t."""
    times = np.asarray(times, dtype=np.float64)
    eps = 1e-4
    if direction > 0:
        i = np.searchsorted(times, t + eps, 'left')
        return int(i) if i < len(times) else None
    i = np.searchsorted(times, t - eps, 'right') - 1
    return int(i) if i >= 0 else None


ONSET_COLOR = (1.0, 0.85, 0.2, 0.9)
BLANK_EDGE_COLOR = (0.6, 0.6, 0.6, 0.6)
UNDELIVERED_COLOR = (1.0, 0.3, 0.3, 0.9)

try:
    from phy import IPlugin
    from phy.cluster.views import ManualClusteringView
    from phy.plot.plot import PlotCanvasMpl
    from phy.plot.visuals import LineVisual
    from phy.utils.color import selected_cluster_color
    from phylib.utils import connect, emit, unconnect
except ImportError:  # allows the pure functions above to be imported without phy
    IPlugin = object
    ManualClusteringView = object
    PlotCanvasMpl = None
    LineVisual = None
    selected_cluster_color = None


class EventAlignedView(ManualClusteringView):
    """Raster (top) and PSTH (bottom) of the selected clusters aligned to event times."""

    plot_canvas_class = PlotCanvasMpl
    _default_position = 'right'
    max_n_clusters = 8
    default_snippets = {
        'set_window': 'ew',
        'set_bin': 'eb',
        'select_event_set': 'es',
        'reload_events': 'er',
        'trace_go_to_event': 'eg',
    }
    default_shortcuts = {
        'trace_next_event': 'alt+]',
        'trace_previous_event': 'alt+[',
    }

    def __init__(self, get_spike_times=None, load_event_sets=None, **kwargs):
        super().__init__(**kwargs)
        self.state_attrs += ('pre_ms', 'post_ms', 'bin_ms', 'event_set_name')
        self.get_spike_times = get_spike_times  # cluster_id -> sorted spike times (s)
        self._load_event_sets = load_event_sets
        self.pre_ms = 20.0
        self.post_ms = 50.0
        self.bin_ms = 0.5
        self.event_set_name = None
        self.event_sets = []
        self._plotted = None  # (event_set, cluster_ids) of the last plot, for clicks
        self._click_marker = None
        self.reload_events(replot=False)

        ax_r, ax_p = self.canvas.subplots(
            2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.05}).flat
        self.ax_raster, self.ax_psth = ax_r, ax_p
        self.canvas.canvas.mpl_connect('button_press_event', self._on_mpl_click)

    # Event sets
    # -------------------------------------------------------------------------

    @property
    def event_set(self):
        for s in self.event_sets:
            if s.name == self.event_set_name:
                return s
        return self.event_sets[0] if self.event_sets else None

    @property
    def event_set_index(self):
        s = self.event_set
        return self.event_sets.index(s) if s is not None else 0

    def reload_events(self, replot=True):
        """Reload event sets from disk (stim_events json, events* files)."""
        self.event_sets = self._load_event_sets() if self._load_event_sets else []
        if self.event_sets:
            logger.info('EventAlignedView event sets (use :es <n>):\n%s', '\n'.join(
                '  %d: %s (%d events)' % (i, s.name, len(s.times))
                for i, s in enumerate(self.event_sets)))
        else:
            logger.warning('EventAlignedView: no event times found (see plugin docstring).')
        if replot:
            self._replot()
            emit('event_set_changed', self)

    def select_event_set(self, index):
        """Choose the event set to align to, by number (see the "Align to" menu).

        Example: `0`

        """
        index = int(index)
        if not 0 <= index < len(self.event_sets):
            logger.warning('Event set %d does not exist (0-%d).', index, len(self.event_sets) - 1)
            return
        self.event_set_name = self.event_sets[index].name
        self._replot()
        emit('event_set_changed', self)

    # Trace view navigation
    # -------------------------------------------------------------------------

    def _trace_view(self):
        gui = getattr(self, 'gui', None)
        tv = gui.get_view('TraceView') if gui is not None else None
        if tv is None:
            logger.warning('EventAlignedView: open a TraceView to jump to events.')
        return tv

    def trace_go_to_event(self, index, offset_ms=0):
        """Centre the TraceView on event #index of the current set (+ offset in ms).

        Example: `0`

        """
        es = self.event_set
        tv = self._trace_view()
        if es is None or tv is None:
            return
        index = int(index)
        if not 0 <= index < len(es.times):
            logger.warning('Event %d does not exist (0-%d).', index, len(es.times) - 1)
            return
        t = float(es.times[index]) + float(offset_ms) / 1e3
        logger.info('TraceView -> %s event #%d at %.4f s', es.name, index, es.times[index])
        tv.go_to(t)

    def _trace_step_event(self, direction):
        es = self.event_set
        tv = self._trace_view()
        if es is None or tv is None or not len(es.times):
            return
        i = neighbour_event(es.times, tv.time, direction)
        if i is not None:
            self.trace_go_to_event(i)

    def trace_next_event(self):
        """Centre the TraceView on the next event of the current set."""
        self._trace_step_event(+1)

    def trace_previous_event(self):
        """Centre the TraceView on the previous event of the current set."""
        self._trace_step_event(-1)

    def _click_to_event(self, xdata, ydata):
        """(event index, cluster id) under a raster click, or None."""
        if self._plotted is None or xdata is None or ydata is None:
            return None
        es, cluster_ids = self._plotted
        n_ev = len(es.times)
        row = int(round(ydata))
        if n_ev == 0 or not 0 <= row < n_ev * len(cluster_ids):
            return None
        return row % n_ev, cluster_ids[row // n_ev]

    def _on_mpl_click(self, e):
        if e.inaxes is not self.ax_raster or e.button != 1:
            return
        hit = self._click_to_event(e.xdata, e.ydata)
        if hit is None:
            return
        i, cid = hit
        if self._click_marker is not None:
            try:
                self._click_marker.remove()
            except Exception:
                pass
        self._click_marker = self.ax_raster.axhline(
            round(e.ydata), color='w', alpha=0.5, lw=3, zorder=0)
        self.canvas.update()
        self.trace_go_to_event(i, offset_ms=e.xdata)

    def set_window(self, pre_ms, post_ms):
        """Window in ms: before and after the event, both positive.

        Example: `20 50`

        """
        self.pre_ms, self.post_ms = abs(float(pre_ms)), abs(float(post_ms))
        self._replot()

    def set_bin(self, bin_ms):
        """PSTH bin width in ms.

        Example: `0.5`

        """
        if float(bin_ms) > 0:
            self.bin_ms = float(bin_ms)
            self._replot()

    # Plotting
    # -------------------------------------------------------------------------

    def _replot(self):
        if self.cluster_ids:
            self.plot()
            self.canvas.update()
        self.update_status()

    @property
    def status(self):
        s = self.event_set
        if s is None:
            return 'no events'
        return '%d: %s, -%g..%g ms, bin %g ms' % (
            self.event_set_index, s.name, self.pre_ms, self.post_ms, self.bin_ms)

    def plot(self, **kwargs):
        ax_r, ax_p = self.ax_raster, self.ax_psth
        ax_r.clear()
        ax_p.clear()
        self._click_marker = None
        self._plotted = None
        es = self.event_set
        if es is None or not len(es.times):
            ax_r.set_title('No event times found', fontsize=9)
            return
        pre, post = self.pre_ms / 1e3, self.post_ms / 1e3
        n_ev = len(es.times)
        cluster_ids = list(self.cluster_ids)
        self._plotted = (es, cluster_ids)

        for ax in (ax_r, ax_p):
            if es.shade_ms is not None:
                ax.axvspan(es.shade_ms[0], es.shade_ms[1], color='0.5', alpha=0.35, lw=0)
            ax.axvline(0, color='w', alpha=0.5, lw=0.8)
            ax.grid(color='w', alpha=0.15)

        msize = max(1.5, min(6.0, 300.0 / max(n_ev * len(cluster_ids), 1)))
        for k, cid in enumerate(cluster_ids):
            color = selected_cluster_color(k)
            st = self.get_spike_times(cid)
            rel, trial = align_spikes(st, es.times, pre, post)
            ax_r.plot(rel * 1e3, trial + k * n_ev, '|', color=color, ms=msize, mew=0.8)
            if k:
                ax_r.axhline(k * n_ev - 0.5, color='w', alpha=0.4, lw=0.5)
            edges, rate = psth(rel, n_ev, pre, post, self.bin_ms / 1e3)
            ax_p.step(edges[:-1] * 1e3, rate, where='post', color=color, lw=1.0,
                      label=str(cid))

        ax_r.set_xlim(-self.pre_ms, self.post_ms)
        ax_r.set_ylim(-0.5, n_ev * len(cluster_ids) - 0.5)
        ax_r.invert_yaxis()
        ax_r.tick_params(labelbottom=False)
        ax_r.set_ylabel('event #')
        ax_r.set_title('%s (%d events)' % (es.name, n_ev), fontsize=9)
        ax_p.set_xlabel('time from event (ms)')
        ax_p.set_ylabel('rate (Hz)')
        if len(cluster_ids) > 1:
            ax_p.legend(fontsize=7, frameon=False, loc='upper right')

    # GUI
    # -------------------------------------------------------------------------

    def attach(self, gui):
        super().attach(gui)
        self.actions.add(self.set_window, prompt=True,
                         prompt_default=lambda: '%g %g' % (self.pre_ms, self.post_ms))
        self.actions.add(self.set_bin, prompt=True, prompt_default=lambda: '%g' % self.bin_ms)
        self.actions.add(self.select_event_set, prompt=True,
                         prompt_default=lambda: str(self.event_set_index))
        self.actions.add(self.reload_events)
        self.actions.separator()
        self.actions.add(self.trace_next_event)
        self.actions.add(self.trace_previous_event)
        self.actions.add(self.trace_go_to_event, prompt=True, n_args=1)
        self.actions.separator()
        for i, s in enumerate(self.event_sets):
            def _choose(i=i):
                self.select_event_set(i)
            self.actions.add(_choose, name='%d: %s' % (i, s.name), alias='_es%d' % i,
                             view_submenu='Align to', show_shortcut=False)
        self.update_status()


class TraceEventOverlay:
    """Vertical lines on a TraceView at event times (and blank-window edges)."""

    def __init__(self, trace_view, get_markers):
        self.view = trace_view
        self.get_markers = get_markers  # () -> list of (times_s, offset_ms, rgba)
        self.visible = True
        self.visual = LineVisual()
        canvas = trace_view.canvas
        # Exclude the per-channel stacking so each line spans the full height.
        canvas.add_visual(self.visual, exclude_origins=(canvas.stacked,), clearable=False)

    def update(self):
        interval = getattr(self.view, 'interval', None)
        if interval is None:
            return
        t0, t1 = interval
        pos, col = marker_lines(self.get_markers(), t0, t1) if self.visible else (None, None)
        if pos is None:
            self.visual.hide()
        else:
            self.view.canvas.update_visual(
                self.visual, pos=pos, color=col, data_bounds=(t0, -1.0, t1, 1.0))
            self.visual.show()
        self.view.canvas.update()


class EventAlignedPlugin(IPlugin):
    def attach_to_controller(self, controller):
        phy_dir = controller.dir_path
        sample_rate = controller.model.sample_rate
        cache = {}

        def load_sets():
            cache['sets'] = load_all_event_sets(phy_dir, sample_rate)
            return cache['sets']

        def get_spike_times(cluster_id):
            spike_ids = controller.supervisor.clustering.spikes_per_cluster.get(cluster_id, [])
            return np.sort(controller.model.spike_times[spike_ids])

        def create_view():
            return EventAlignedView(get_spike_times=get_spike_times, load_event_sets=load_sets)

        controller.view_creator['EventAlignedView'] = create_view

        overlays = []

        def get_markers(gui):
            """Current set of the EventAlignedView (set 0 if none is open), its blank-window
            edges, and TTLs without a stim artefact (always, in red)."""
            eav = gui.get_view('EventAlignedView')
            sets = eav.event_sets if eav is not None else cache.get('sets') or load_sets()
            es = eav.event_set if eav is not None else (sets[0] if sets else None)
            markers = []
            if es is not None:
                markers.append((es.times, 0.0, ONSET_COLOR))
                if es.shade_ms is not None:
                    markers.append((es.times, es.shade_ms[0], BLANK_EDGE_COLOR))
                    markers.append((es.times, es.shade_ms[1], BLANK_EDGE_COLOR))
            for s in sets:
                if s.name == 'stim: TTL, no artefact' and s is not es:
                    markers.append((s.times, 0.0, UNDELIVERED_COLOR))
            return markers

        @connect
        def on_view_attached(view, gui):
            if view.__class__.__name__ != 'TraceView':
                return
            try:
                overlay = TraceEventOverlay(view, lambda: get_markers(gui))
            except Exception as e:
                logger.warning('EventAlignedView: could not add event markers to TraceView: %s', e)
                return
            overlays.append(overlay)

            def toggle_event_markers(checked):
                """Show stim/event markers in the trace view."""
                overlay.visible = checked
                overlay.update()

            view.actions.separator()
            view.actions.add(toggle_event_markers, checkable=True, checked=True)

            @connect(sender=view)
            def on_time_range_selected(sender, interval):
                overlay.update()

            overlay.update()

        @connect
        def on_event_set_changed(sender):
            for overlay in overlays:
                if not overlay.view._closed:
                    overlay.update()
