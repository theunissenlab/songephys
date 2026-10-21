"""NCAF closed-loop (live spike detection + feedback) tables, linked to the recorded data.

NCAF runs on the live SpikeGLX stream, so its logs give absolute (acquisition)
sample counts. These map onto the recording through the per-trigger raw file
table; events that fell between saved trigger files are kept with NaN times
and recorded=False.

Links are DynamicTableRegions, one direction only (HDMF cannot read cycles):

    online_spike_events.trial         -> intervals/ncaf_trials
    online_spike_events.matched_unit  -> units
    ncaf_trials.syllable              -> intervals/syllables
    ncaf_trials.feedback_event        -> processing/ncaf/feedback_events
    feedback_events.ttl_event         -> acquisition/ttl_events (feedback line)
    syllable_times.syllable           -> intervals/syllables
    target_unit_candidates.unit       -> units

Optional links are ragged: each row holds 0 or 1 target rows. Read them with
table.to_dataframe(index=True), where an empty list means no link.
"""
import numpy as np
from pynwb import NWBHDF5IO
from pynwb.epoch import TimeIntervals
from hdmf.common import DynamicTable
from scipy.ndimage import uniform_filter1d

from ..pipeline.targets import NWBIntervalsTarget, NWBProcessingTarget
from ..utils import pulse_edges
from . import ncaf
from .nwb import (NPXSessionTask, AddNIAcquisitionNWB, AddSyllablesNWB, AddNPXUnits,
                  make_table, optional_link, nearest)
from .spikeglx import memmap_bin, ni_volts_per_bit, sample_rate


def detect_feedback_bursts(x_volts, fs, threshold_v=0.02, smooth_samples=100, min_ms=5.0):
    """Noise bursts on the feedback-copy channel: onset, offset (samples) and mean dB.

    Bursts are found on a moving-average envelope of |x| (as snakeSort's
    get_feedbacks); onset/offset are then refined to the first/last raw sample
    above threshold so the envelope's smoothing does not shift them."""
    ax = np.abs(x_volts)
    env = uniform_filter1d(ax, size=smooth_samples, mode='nearest')
    r, f = pulse_edges(env >= threshold_v, invert=False)
    r, f = r + 1, f + 1
    keep = (f - r) >= min_ms * fs / 1000
    r, f = r[keep], f[keep]
    half = smooth_samples
    on, off, vol = [], [], []
    for a, b in zip(r, f):
        lo, hi = max(a - half, 0), min(b + half, len(ax))
        above = np.flatnonzero(ax[lo:hi] >= threshold_v)
        a2, b2 = (lo + above[0], lo + above[-1] + 1) if len(above) else (a, b)
        on.append(a2); off.append(b2)
        vol.append(np.mean(10 * np.log10(np.maximum(ax[a2:b2], np.finfo(np.float32).eps))))
    return np.array(on, dtype=np.int64), np.array(off, dtype=np.int64), np.array(vol)


def score_target_units(units_df, electrodes_df, online_t, filter_channels, window_s, max_dist_um, n):
    """Rank offline units by how well they explain the online-detected spikes.

    score = (frac - chance) / (1 - chance), where frac is the fraction of online
    spikes with a spike of the unit within +-window_s and chance = 1 - exp(-rate * 2 * window_s).
    Returns the top n rows and the row index (into units) of the selected unit:
    the best-scoring unit within max_dist_um of the filter channels."""
    fx = electrodes_df.loc[filter_channels, 'rel_x'].mean()
    fy = electrodes_df.loc[filter_channels, 'rel_y'].mean()
    rows = []
    for row_i, (uid, u) in enumerate(units_df.iterrows()):
        x = u.spike_times
        if len(x) == 0 or len(online_t) == 0:
            continue
        j = nearest(x, online_t, window_s)
        frac = np.mean(j >= 0)
        chance = 1 - np.exp(-u.firing_rate * 2 * window_s)
        lag = (x[j[j >= 0]] - online_t[j >= 0]) * 1e3
        rows.append(dict(unit_row=row_i, kilosort_id=int(u.kilosort_id), group=str(u.group),
                         score=(frac - chance) / (1 - chance), match_frac=frac, chance=chance,
                         dist_um=float(np.hypot(u.ks_x - fx, u.ks_y - fy)),
                         median_lag_ms=float(np.median(lag)) if len(lag) else np.nan))
    rows.sort(key=lambda r: -r['score'])
    near = [r for r in rows if r['dist_um'] <= max_dist_um]
    selected = near[0]['unit_row'] if near else -1
    return rows[:n], selected


class AddNCAFNWB(NPXSessionTask):
    """NCAF trials, feedback events, online spikes and target-unit matching."""

    def requires(self):
        kw = dict(config=self.config, session=self.session)
        return [AddNIAcquisitionNWB(**kw), AddSyllablesNWB(**kw), AddNPXUnits(**kw)]

    def output(self):
        return {'intervals': NWBIntervalsTarget(self.nwb_path, 'ncaf_trials'),
                'processing': NWBProcessingTarget(self.nwb_path, 'ncaf', 'online_spike_events',
                                                  'feedback_events', 'target_unit_candidates')}

    def run(self):
        cfg = self.cfg
        p = cfg['ncaf']
        d = cfg.ncaf_dir(self.session)
        clock = self.clock()
        ifs = clock.imec_fs
        tol = p['link_tol_ms'] / 1000

        # ---- NCAF logs ------------------------------------------------------------
        dec = ncaf.parse_decisions_log(d / 'decisions.log').sort_values('imec_onset_sample').reset_index(drop=True)
        for c in ('window_start_ms', 'window_end_ms', 'decision_imec_sample', 'pulse_imec_sample', 'pulse_ms'):
            if c not in dec:
                dec[c] = np.nan
        # the counting window is only logged on TRIGGER lines; it is constant within a session
        for c in ('window_start_ms', 'window_end_ms'):
            logged = dec[c].dropna()
            dec[c] = dec[c].fillna(logged.mode().iloc[0] if len(logged) else np.nan)
        onset_imec = dec.imec_onset_sample.values.astype(np.int64)
        onset_t = clock.ni_abs_sample_to_time(dec.ni_onset_sample.values)
        recorded = np.isfinite(onset_t)

        def imec_t(col):
            s = dec[col].values
            ok = np.isfinite(s)
            out = np.full(len(s), np.nan)
            out[ok] = clock.imec_abs_sample_to_time(s[ok].astype(np.int64))
            return out
        decision_t, pulse_t = imec_t('decision_imec_sample'), imec_t('pulse_imec_sample')

        st = np.sort(ncaf.read_spike_times(d / 'spikeTimes.txt'))
        st_t = clock.imec_abs_sample_to_time(st)
        syl_log = ncaf.read_syllable_times(d / 'syllableTimes.txt')
        filters = ncaf.read_filters(d / 'filters')

        # ---- per-trial online spikes ---------------------------------------------
        pre, post = [x * ifs / 1000 for x in p['trial_window_ms']]
        trial_spikes, window_count = [], []
        for o, ws, we in zip(onset_imec, dec.window_start_ms.values, dec.window_end_ms.values):
            sel = st[(st >= o + pre) & (st <= o + post)]
            rel = (sel - o) / ifs * 1000
            trial_spikes.append(rel.astype(float))
            window_count.append(int(np.sum((rel >= ws) & (rel < we))))
        window_count = np.array(window_count)
        # spikeTimes.txt can stop before the session ends; an empty trial then means "not logged"
        spikes_logged = (len(st) > 0) & (onset_imec + post >= (st.min() if len(st) else 0)) & \
                        (onset_imec + pre <= (st.max() if len(st) else -1))

        with NWBHDF5IO(self.nwb_path, 'r+') as io:
            nwbfile = io.read()
            syllables = nwbfile.intervals['syllables']
            ttl = nwbfile.acquisition['ttl_events']
            units = nwbfile.units
            sdf = syllables.to_dataframe()
            ttl_df = ttl.to_dataframe()
            udf = units.to_dataframe(exclude={'spike_times_imec'})
            edf = nwbfile.electrodes.to_dataframe()

            # ---- feedback events: TTL line + XA1 bursts + NCAF triggers ------------
            fb_rows = np.flatnonzero(ttl_df.ttl_name.values == 'feedback')
            fb_on = ttl_df.timestamp.values[fb_rows]
            fb_dur = ttl_df.duration.values[fb_rows]
            fb_name = [k for k, v in cfg['ni']['analog'].items() if v == 'feedback_copy']
            xa_on = xa_off = xa_vol = np.array([])
            if fb_name:
                run = self.run_dir
                meta = run.ni_meta
                fs = sample_rate(meta)
                ch = int(fb_name[0])
                x = memmap_bin(run.ni_bin, meta)[:, ch].astype(np.float32) * np.float32(ni_volts_per_bit(meta, ch))
                a, b, xa_vol = detect_feedback_bursts(x, fs, p['xa1_threshold_v'], p['xa1_smooth_samples'], p['xa1_min_ms'])
                xa_on, xa_off = a / fs, b / fs
                del x

            # rows: every TTL feedback pulse, then XA1 bursts with no TTL pulse
            ev_ttl = list(range(len(fb_on)))
            ev_xa = list(nearest(xa_on, fb_on, tol)) if len(fb_on) else []
            used = set(i for i in ev_xa if i >= 0)
            for i in range(len(xa_on)):
                if i not in used:
                    ev_ttl.append(-1); ev_xa.append(i)
            ev_ttl, ev_xa = np.array(ev_ttl, dtype=np.int64), np.array(ev_xa, dtype=np.int64)
            ev_t = np.where(ev_ttl >= 0, fb_on[np.clip(ev_ttl, 0, None)] if len(fb_on) else np.nan,
                            xa_on[np.clip(ev_xa, 0, None)] if len(xa_on) else np.nan)
            # NCAF triggers -> nearest measured event; unmatched triggers become their own rows
            trig = np.flatnonzero(dec.pulse_imec_sample.notna().values)
            order = np.argsort(ev_t)
            ev_ttl, ev_xa, ev_t = ev_ttl[order], ev_xa[order], ev_t[order]
            k = nearest(ev_t, pulse_t[trig], tol)
            ev_trig = np.full(len(ev_t), -1, dtype=np.int64)
            for ti, ki in zip(trig, k):
                if ki >= 0 and ev_trig[ki] < 0:
                    ev_trig[ki] = ti
            extra = [ti for ti, ki in zip(trig, k) if ki < 0 or ev_trig[ki] != ti]
            ev_ttl = np.r_[ev_ttl, np.full(len(extra), -1)]
            ev_xa = np.r_[ev_xa, np.full(len(extra), -1)]
            ev_trig = np.r_[ev_trig, extra].astype(np.int64)
            ev_t = np.r_[ev_t, pulse_t[extra]]
            order = np.argsort(ev_t, kind='stable')  # NaN (unrecorded) last
            ev_ttl, ev_xa, ev_trig, ev_t = ev_ttl[order], ev_xa[order], ev_trig[order], ev_t[order]
            pick = lambda arr, idx: np.where(idx >= 0, arr[np.clip(idx, 0, None)] if len(arr) else np.nan, np.nan)
            trial_feedback = np.full(len(dec), -1, dtype=np.int64)
            trial_feedback[ev_trig[ev_trig >= 0]] = np.flatnonzero(ev_trig >= 0)
            feedback_events = make_table(
                DynamicTable, 'feedback_events',
                'Feedback deliveries from three sources: the NI feedback TTL line, noise bursts on the '
                'feedback copy (XA1), and NCAF TRIGGER log lines. Rows are matched within '
                f"{p['link_tol_ms']} ms; a row seen by only some sources has the others NaN/False.", {
                    'timestamp': (np.where(ev_ttl >= 0, pick(fb_on, ev_ttl), np.where(ev_xa >= 0, pick(xa_on, ev_xa), ev_t)),
                                  'best measured onset (s): TTL onset, else XA1 onset, else logged pulse time'),
                    'recorded': (np.isfinite(ev_t), 'inside a recorded segment'),
                    'on_ttl': (ev_ttl >= 0, 'seen on the feedback TTL line'),
                    'on_xa1': (ev_xa >= 0, 'seen as a noise burst on the feedback copy (XA1)'),
                    'in_ncaf_log': (ev_trig >= 0, 'logged as a TRIGGER by NCAF'),
                    'ttl_event': (optional_link(np.where(ev_ttl >= 0, fb_rows[np.clip(ev_ttl, 0, None)], -1)),
                                  'row in acquisition/ttl_events (feedback line)', ttl),
                    'ttl_onset': (pick(fb_on, ev_ttl), 'feedback TTL onset (s)'),
                    'ttl_duration': (pick(fb_dur, ev_ttl), 'feedback TTL duration (s)'),
                    'xa1_onset': (pick(xa_on, ev_xa), 'XA1 burst onset (s)'),
                    'xa1_offset': (pick(xa_off, ev_xa), 'XA1 burst offset (s)'),
                    'xa1_vol_db': (pick(xa_vol, ev_xa), 'mean 10*log10(|V|) over the XA1 burst'),
                    'logged_pulse_time': (pick(pulse_t, ev_trig), 'NCAF pulse_imec_sample on the NI clock (s)'),
                    'logged_pulse_ms': (pick(dec.pulse_ms.values, ev_trig), 'NCAF pulse latency from syllable onset (ms)'),
                    'pulse_imec_sample': (np.where(ev_trig >= 0, dec.pulse_imec_sample.fillna(-1).values[np.clip(ev_trig, 0, None)], -1).astype(np.int64),
                                          'absolute imec sample from the NCAF log (-1 if not logged)'),
                })

            # ---- trials ----------------------------------------------------------
            syl_link = nearest(sdf.start_time.values, onset_t, 1e-3)
            syl_link = np.where((syl_link >= 0) & (sdf.code.values[np.clip(syl_link, 0, None)] == dec.code.values), syl_link, -1)
            ncaf_trials = make_table(
                TimeIntervals, 'ncaf_trials',
                f'NCAF decisions from {d}/decisions.log, one per detected target syllable. Trials that fell '
                'between saved trigger files have recorded=False and NaN times.', {
                    'start_time': (onset_t, 'targeted syllable onset (NI clock, from ni_onset_sample; NaN if not recorded)'),
                    'stop_time': (np.where(np.isfinite(decision_t), decision_t, onset_t), 'decision time if logged, else onset'),
                    'recorded': (recorded, 'inside a recorded segment'),
                    'code': (dec.code.values, 'syllable code'),
                    'mode': (dec['mode'].astype(str).values, 'push_up / push_down'),
                    'count': (dec['count'].values, 'online spike count in the counting window (as logged)'),
                    'window_count': (window_count, 'count recomputed from spikeTimes.txt in the counting window '
                                     '(only meaningful where spikes_logged)'),
                    'threshold': (dec.threshold.values, 'count threshold'),
                    'triggered': (dec.triggered.values, 'feedback delivered'),
                    'closed': (dec.closed.astype(str).values, 'how the trial closed'),
                    'window_start_ms': (dec.window_start_ms.values, 'counting window start (ms after onset)'),
                    'window_end_ms': (dec.window_end_ms.values, 'counting window end (ms after onset)'),
                    'spike_times': (trial_spikes, f"online-detected spikes (spikeTimes.txt) from {p['trial_window_ms'][0]} "
                                    f"to {p['trial_window_ms'][1]} ms around onset, in ms relative to onset"),
                    'spikes_logged': (spikes_logged, 'trial lies within the span covered by spikeTimes.txt '
                                      '(False -> spike_times is empty because nothing was logged)'),
                    'spikes_ms': (list(dec.spikes_ms.values), 'online spike times relative to onset (ms), from decisions.log'),
                    'sub_counts': (list(dec.sub_counts.values), 'spike counts in sub-windows (decisions.log)'),
                    'pair_counts': (list(dec.pair_counts.values), 'spike counts in paired sub-windows (decisions.log)'),
                    'decision_time': (decision_t, 'decision time (NI clock)'),
                    'pulse_time': (pulse_t, 'logged feedback pulse time (NI clock; NaN if not triggered)'),
                    'pulse_ms': (dec.pulse_ms.values, 'feedback pulse latency from onset (ms)'),
                    'syllable': (optional_link(syl_link), 'row in intervals/syllables (same onset and code)', syllables),
                    'feedback_event': (optional_link(trial_feedback), 'row in processing/ncaf/feedback_events', feedback_events),
                    'ni_onset_sample': (dec.ni_onset_sample.values, 'absolute NI sample from the log'),
                    'imec_onset_sample': (onset_imec, 'absolute imec sample from the log'),
                })

            # ---- target unit ---------------------------------------------------
            filt_ch = filters[0]['channels'] if filters else np.array([], dtype=int)
            cands, selected = score_target_units(udf, edf, st_t[np.isfinite(st_t)], filt_ch,
                                                 p['match_window_ms'] / 1000, p['match_max_dist_um'], p['n_candidates'])
            target = make_table(
                DynamicTable, 'target_unit_candidates',
                f"Offline units scored against the online-detected spikes (template {filters[0]['template'] if filters else '?'} "
                f"of the baseline sort used to build the filter). score = (frac - chance)/(1 - chance) within "
                f"+-{p['match_window_ms']} ms; selected = best score within {p['match_max_dist_um']} um of the filter channels.", {
                    'rank': (np.arange(len(cands)) + 1, 'rank by score'),
                    'unit': (np.array([c['unit_row'] for c in cands], dtype=np.int64), 'row in units', units),
                    'kilosort_id': (np.array([c['kilosort_id'] for c in cands]), 'Kilosort cluster id in this session'),
                    'group': (np.array([c['group'] for c in cands]), 'Kilosort group'),
                    'score': (np.array([c['score'] for c in cands]), 'chance-corrected match fraction'),
                    'match_frac': (np.array([c['match_frac'] for c in cands]), 'fraction of online spikes with a unit spike in the window'),
                    'chance': (np.array([c['chance'] for c in cands]), 'expected match fraction from the unit rate alone'),
                    'dist_um': (np.array([c['dist_um'] for c in cands]), 'distance from unit position to the filter channels'),
                    'median_lag_ms': (np.array([c['median_lag_ms'] for c in cands]), 'median (offline - online) spike time of matches'),
                    'selected': (np.array([c['unit_row'] == selected for c in cands]), 'chosen as the online target'),
                })

            # ---- online spikes ---------------------------------------------------
            k = np.searchsorted(onset_imec, st, side='right') - 1
            kc = np.clip(k, 0, None)
            rel = (st - onset_imec[kc]) / ifs * 1000
            in_trial = (k >= 0) & (rel <= p['trial_window_ms'][1])
            # spikes before an onset (within the pre window) belong to the following trial
            k2 = np.clip(k + 1, 0, len(onset_imec) - 1)
            rel2 = (st - onset_imec[k2]) / ifs * 1000
            pre_next = ~in_trial & (k + 1 < len(onset_imec)) & (rel2 >= p['trial_window_ms'][0])
            trial_idx = np.where(in_trial, kc, np.where(pre_next, k2, -1))
            trel = np.where(trial_idx >= 0, (st - onset_imec[np.clip(trial_idx, 0, None)]) / ifs * 1000, np.nan)
            in_win = (trial_idx >= 0) & (trel >= dec.window_start_ms.values[np.clip(trial_idx, 0, None)]) & \
                     (trel < dec.window_end_ms.values[np.clip(trial_idx, 0, None)])
            dec_spk = np.sort(np.concatenate([o + np.asarray(m) * ifs / 1000 for o, m in zip(onset_imec, dec.spikes_ms)]
                                             + [np.array([])]))
            in_spikes_ms = nearest(dec_spk, st, 2) >= 0
            if selected >= 0:
                usp = udf.spike_times.iloc[selected]
                m = nearest(usp, st_t, p['match_window_ms'] / 1000)
                lag = np.where(m >= 0, (usp[np.clip(m, 0, None)] - st_t) * 1e3, np.nan)
                matched = np.where(m >= 0, selected, -1)
            else:
                lag, matched = np.full(len(st), np.nan), np.full(len(st), -1)
            online = make_table(
                DynamicTable, 'online_spike_events',
                f'Target spikes detected live by NCAF (spikeTimes.txt), all of them. spikeTimes.txt can stop '
                f'before the session ends (last spike here at imec sample {st.max() if len(st) else -1}).', {
                    'timestamp': (st_t, 'NI clock (s); NaN if between saved trigger files'),
                    'recorded': (np.isfinite(st_t), 'inside a recorded segment'),
                    'imec_sample': (st, 'absolute imec sample from the live stream'),
                    'trial': (optional_link(trial_idx), f"row in intervals/ncaf_trials whose {p['trial_window_ms']} ms window "
                              'contains this spike', ncaf_trials),
                    'trial_latency_ms': (trel, 'time from that trial\'s syllable onset (ms)'),
                    'in_count_window': (in_win, 'inside that trial\'s counting window'),
                    'in_spikes_ms': (in_spikes_ms, 'listed in a decisions.log spikes_ms entry'),
                    'matched_unit': (optional_link(matched), 'selected target unit, if it has a spike within '
                                     f"+-{p['match_window_ms']} ms", units),
                    'matched_lag_ms': (lag, 'offline (Kilosort) minus online spike time (ms)'),
                })

            syl_t = clock.ni_abs_sample_to_time(syl_log.ni_sample.values)
            sl = nearest(sdf.start_time.values, syl_t, 1e-3)
            sl = np.where((sl >= 0) & (sdf.code.values[np.clip(sl, 0, None)] == syl_log.code.values), sl, -1)
            syllable_times = make_table(DynamicTable, 'syllable_times', 'syllableTimes.txt: all online syllable detections', {
                'code': (syl_log.code.values, 'syllable code'),
                'timestamp': (syl_t, 'NI time (NaN if not recorded)'),
                'recorded': (np.isfinite(syl_t), 'inside a recorded segment'),
                'syllable': (optional_link(sl), 'row in intervals/syllables (same onset and code)', syllables),
                'ni_sample': (syl_log.ni_sample.values, 'absolute NI sample'),
                'imec_sample': (syl_log.imec_sample.values, 'absolute imec sample')})

            mod = nwbfile.create_processing_module('ncaf', f'NCAF closed-loop outputs from {d}')
            mod.add(feedback_events)
            nwbfile.add_time_intervals(ncaf_trials)
            mod.add(online)
            mod.add(syllable_times)
            mod.add(target)
            mod.add(make_table(DynamicTable, 'params', 'NCAF template-matching filters (filters/*.bin)', {
                'template': ([x['template'] for x in filters], 'template id in the baseline sort used to build the filter '
                             '(not a Kilosort id of this session; see target_unit_candidates)'),
                'channels': ([x['channels'].astype(np.int64) for x in filters], 'channels used by the filter'),
                'threshold': ([x['threshold'] for x in filters], 'detection threshold'),
                'car_shank': ([x['car_shank'] for x in filters], 'shank used for CAR'),
                'filter': ([x['filter'].ravel() for x in filters], 'filter weights, flattened (n_channels x n_samples)'),
                'filter_n_samples': ([x['filter'].shape[-1] for x in filters], 'samples per channel in filter')}))
            io.write(nwbfile)
