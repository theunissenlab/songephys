# Neuropixels NWB schema (songephys.npx)

One `{session}.nwb` per SpikeGLX run (e.g. `Choc57_d1.1.nwb`), written by
`song npx-gen-nwb CONFIG` to the config's `nwb_dir`. Names follow the Intan
songephys files (`audio`, `ttl_events`, `units`, `unit_waveforms`,
`unit_intervals`) so analysis code can read both.

## Time base

**Every time is seconds on the NI (nidq) clock, along the CatGT-concatenated NI
file.** t = 0 is the first sample of the first trigger file;
`session_start_time` is that file's SpikeGLX `fileCreateTime`.

The recordings are TTL-triggered and CatGT concatenates the trigger files, so
NWB time skips the gaps between triggers. `intervals/recording_segments` lists
the contiguous stretches with their real acquisition time (`acq_time`), so you
can still get the real time between segments.

Spike times are converted from the imec clock to NI per trigger file
(`songephys/npx/sync.py`). TPrime pairs sync edges across the whole
concatenated file and mis-pairs them at joins where time was skipped. TPrime's
NI-master output is still written to the KS folder as a cross-check
(`spike_times_ni_tprime.npy`).

## Top level

| what | where | notes |
|---|---|---|
| session | `session_id`, `session_description`, `session_start_time` | start time is in the rig `timezone` from the config |
| subject | `nwbfile.subject` | `subject_id`, `sex`, `date_of_birth`, `species` |
| probe | `devices['imec0']` | part and serial numbers in `description` / `serial_number` |
| electrode groups | `electrode_groups['imec0_shank{1..4}']` | one per shank, `location` from config |

## Electrodes (`nwbfile.electrodes`)

`id` = SpikeGLX/Kilosort channel number.

| column | meaning |
|---|---|
| `x`, `y`, `z` | stereotaxic estimate in mm, songephys convention: +x posterior, +y inferior (depth below surface), +z right. Computed from `probe.ap_mm` (posterior shank), `probe.depth_mm` (tip) and `probe.ml_mm` |
| `rel_x`, `rel_y` | site position on the probe in µm (Kilosort `channel_positions`; `rel_y` is height above the tip) |
| `shank` | Kilosort shank label (1-4); `probe.posterior_shank` is the posterior one |
| `channel` | channel number (same as `id`) |
| `reference`, `filtering` | `imChan0Ref`, CatGT command line |

## Acquisition — NI signals

| name | contents |
|---|---|
| `audio` | NI XA0 (mic), int16 counts, `conversion` = V/count, gzip compressed |
| `feedback_copy` | NI XA1 (copy of the feedback noise), same format |
| `ni_digital` | raw NI digital word XD0 as uint8; bit n = line n. `comments` maps lines to names |
| `ttl_events` | one row per pulse on each named line except `sync` and the syllable code lines: `timestamp`, `ttl_type` (line number), `ttl_name`, `duration` |

Default line map (config `ni.digital_lines`, bits counted from 0): 0 `sync`,
1 `trigger` (15 s recording gate), 3 `feedback`, 4 `sound_threshold`; 5-6
syllable code bits.

## Intervals

| name | contents |
|---|---|
| `syllables` | online syllable detections decoded from lines 5-6. `start_time` = code pulse onset, `code` = bit5 + 2·bit6 (1-3). The code is emitted at syllable onset, just before the matching `sound_threshold` pulse rises: `sound_onset`/`sound_offset` give that pulse and `stop_time` = `sound_offset` |
| `ncaf_trials` | the NCAF hub table, one row per `decisions.log` decision (all of them; `recorded=False` and NaN times if the syllable fell between saved trigger files). Has `code`, `mode`, `count` (logged), `window_count` (recounted from `spikeTimes.txt`), `threshold`, `triggered`, `closed`, counting window, ragged `spike_times` (online spikes −50…+250 ms, ms relative to onset), `spikes_logged`, ragged `spikes_ms`/`sub_counts`/`pair_counts`, `decision_time`, `pulse_time`/`pulse_ms`, and links `syllable` → `intervals/syllables` and `feedback_event` → `processing/ncaf/feedback_events`. `stop_time` = decision time |
| `recording_segments` | contiguous recorded stretches: tcat start/stop, trigger file range, `acq_time` (s since acquisition start), absolute first samples, `file_create_time` |
| `unit_intervals` | valid (active) periods per unit, same method as the Intan files |

## Units (`nwbfile.units`)

One row per Kilosort cluster that isn't labelled `noise`. `group` comes from
`cluster_group.tsv` (Phy curation) if present, else `cluster_KSLabel.tsv`.

| column | meaning |
|---|---|
| `spike_times` | **NI clock** seconds (from `spike_times_ni_sec.npy`) |
| `spike_times_imec` | same spikes on the imec clock (tcat seconds), before alignment |
| `kilosort_dir`, `kilosort_id` | Kilosort output folder and cluster id |
| `group` | `good` / `mua` |
| `ks_x`, `ks_y` | mean spike position (µm, probe coordinates) |
| `electrodes`, `peak_channel` | channel with the largest template peak-to-peak |
| `amplitude`, `contam_pct` | Kilosort `cluster_Amplitude.tsv`, `cluster_ContamPct.tsv` |
| `n_spikes`, `firing_rate` | spike count, mean rate over the recorded time |

`acquisition/unit_waveforms`: `unit_id` (row id in `units`), `electrodes` (top 4
channels), `waveform_mean`/`waveform_sd` (channel × time, ±2 ms, up to 1000
spikes, int16 counts of the CatGT-filtered AP band; µV/count in the table
description), `template` (Kilosort template on the same channels).

## Processing modules

| module / table | contents |
|---|---|
| `metadata/catgt_offsets` | one row per trigger file: absolute first sample and length (NI, imec), `file_create_time`, and where it starts in each tcat stream (`smp_*`, `sec_*` from `ct_offsets.txt`) |
| `metadata/ni_clock_pieces` | per trigger file: sync edges paired and the imec→NI offset |
| `metadata/sync_edges` | raw sync edge times in each tcat stream |
| `metadata/pipeline_params` | CatGT command line, sample rates, KS settings, clock, full config |
| `ncaf/online_spike_events` | every live-detected target spike (`spikeTimes.txt`): NI `timestamp`, `recorded`, `imec_sample`, link `trial` → `ncaf_trials` (trial whose −50…+250 ms window contains it), `trial_latency_ms`, `in_count_window`, `in_spikes_ms`, link `matched_unit` → `units` (selected target unit has a spike within ±0.5 ms), `matched_lag_ms` |
| `ncaf/feedback_events` | one row per feedback delivery, merged from the NI feedback TTL line, XA1 noise bursts and NCAF TRIGGER lines (matched within 5 ms): `timestamp`, `recorded`, `on_ttl`/`on_xa1`/`in_ncaf_log`, link `ttl_event` → `ttl_events`, `ttl_onset`/`ttl_duration`, `xa1_onset`/`xa1_offset`/`xa1_vol_db`, `logged_pulse_time`/`logged_pulse_ms`, `pulse_imec_sample` |
| `ncaf/target_unit_candidates` | top 5 offline units scored against the online spikes: link `unit` → `units`, `kilosort_id`, `score` = (match − chance)/(1 − chance), `match_frac`, `chance`, `dist_um` from the filter channels, `median_lag_ms`, `selected` |
| `ncaf/syllable_times` | `syllableTimes.txt` (all online syllable detections) with NI `timestamp`, `recorded`, link `syllable` → `intervals/syllables` |
| `ncaf/params` | `filters/*.bin`: template id **in the baseline sort the filter was built from** (not this session's Kilosort id), channels, threshold, CAR shank, filter weights |

NCAF tables are read from `{raw_dir}/{run}_g{gate}/ncaf_outs` and are built with
the units (they link to them), so `npx-gen-nwb --no-units` skips them and
`npx-add-units --clear-old-units` rebuilds them.

**Links** are `DynamicTableRegion`s and only point one way (HDMF can't read
cycles). Optional links are ragged, with 0 or 1 target rows per row. Read them
with `table.to_dataframe(index=True)`, where `[]` means no link. Indexing an
empty link directly, e.g. `table['col'][i]`, raises in HDMF.

## Quick lookup

- mic → `acquisition['audio'].data[:] * acquisition['audio'].conversion`
- syllable onsets/codes → `intervals['syllables']`
- feedback pulses → `processing['ncaf']['feedback_events']` (TTL + XA1 + NCAF log, merged)
- live target spikes per trial → `intervals['ncaf_trials']['spike_times']`; all of them → `processing['ncaf']['online_spike_events']`
- which offline unit NCAF was targeting → `processing['ncaf']['target_unit_candidates']` where `selected`
- spikes → `units['spike_times'][i]` (same clock as the audio)
- real time between triggers → `intervals['recording_segments'].acq_time`
