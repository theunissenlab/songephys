import unittest
import warnings

import numpy as np
from scipy.signal import sosfiltfilt

from songephys.preprocess.ephys.stim_artifact import (
    STIM_ARTIFACT_DEFAULTS,
    annotate_bouts,
    blank_and_filter,
    blank_and_filter_chunk,
    blank_windows,
    build_stim_events,
    extend_windows,
    extra_blank_windows,
    find_extra_artifacts,
    find_jump_onset,
    parse_extra_artifacts_s,
    parse_extra_blank_s,
    stim_events_summary,
    find_ttl_edges,
    get_stim_artifact_config,
    group_bouts,
    make_bandpass_sos,
    merge_windows,
    pairs_outside_windows,
    resolve_digital_bit,
    score_artifacts,
    stim_nwb_rows,
    ttl_on_off,
)

FS = 30000.0
ms = lambda x: int(round(x * FS / 1e3))


def synthetic_stim_recording(n_ch=16, dur_s=3.0, pulse_s=(0.5, 1.5, 2.5), seed=0,
                             delivered=(True, True, True)):
    """Noise + clipped biphasic pulses with mV-scale, channel-specific exponential tails"""
    rng = np.random.default_rng(seed)
    n = int(dur_s * FS)
    x = rng.normal(0, 50, size=(n, n_ch))  # ~10 uV at 0.195 uV/bit
    t = np.arange(ms(200)) / FS
    for p, d in zip(pulse_s, delivered):
        if not d:
            continue
        f = int(p * FS)
        amp = rng.uniform(0.3, 1.0, n_ch) * 30000
        x[f:f + ms(0.1)] += amp
        x[f + ms(0.1):f + ms(0.2)] -= amp
        tail = rng.uniform(-1, 1, n_ch) * 20000
        seg = x[f + ms(0.2):f + ms(0.2) + len(t)]
        seg += np.exp(-t / 0.03)[:len(seg), None] * tail
    return np.clip(x, -32764, 32764), [int(p * FS) for p in pulse_s]


class TestConfigAndDigital(unittest.TestCase):
    def test_config_merge(self):
        cfg = get_stim_artifact_config(
            {"stim_artifact": {"enabled": True, "ms_after": 3.0}},
            {"STIM_ARTIFACT": {"MS_AFTER": 2.0, "MS_BEFORE": 1.0}})
        self.assertTrue(cfg["enabled"])
        self.assertEqual(cfg["ms_after"], 3.0)   # site wins
        self.assertEqual(cfg["ms_before"], 1.0)  # project over default
        self.assertEqual(cfg["freq_min"], STIM_ARTIFACT_DEFAULTS["freq_min"])
        self.assertFalse(get_stim_artifact_config(None, None)["enabled"])
        with self.assertWarns(UserWarning):
            get_stim_artifact_config({"stim_artifact": {"bogus": 1}}, None)

    def test_resolve_digital_bit(self):
        self.assertEqual(resolve_digital_bit("DIGITAL-IN-01"), 0)
        self.assertEqual(resolve_digital_bit("digital-in-10"), 9)
        self.assertEqual(resolve_digital_bit(3), 3)
        site = {"digital_channel_names": ["DIGITAL-IN-01", "DIGITAL-IN-09"]}
        self.assertEqual(resolve_digital_bit("DIGITAL-IN-09", site), 8)
        site = {"digital_channel_names": ["Stim pulse", "Feedback"], "digital_channel_bits": [0, 8]}
        self.assertEqual(resolve_digital_bit("Feedback", site), 8)
        with self.assertRaises(ValueError):
            resolve_digital_bit("nope", site)

    def test_find_ttl_edges_matches_ttl_on_off_across_chunks(self):
        rng = np.random.default_rng(1)
        n = 100_003
        word = rng.integers(0, 2, size=n).astype(np.uint16) << 4  # noise on bit 4
        stim = np.zeros(n, np.uint16)
        starts = np.sort(rng.choice(np.arange(10, n - 20, 50), 300, replace=False))
        for s in starts:
            stim[s:s + 6] = 1
        word |= stim  # stim on bit 0
        r, f = find_ttl_edges(lambda a, b: word[a:b], n, bit=0, chunk_size=997)
        np.testing.assert_array_equal(r, starts)
        np.testing.assert_array_equal(f, starts + 6)
        r2, f2 = ttl_on_off(stim)  # ttl_on_off reports the last low sample
        np.testing.assert_array_equal(r, r2 + 1)
        np.testing.assert_array_equal(f, f2 + 1)

    def test_find_ttl_edges_pulse_at_end_and_start(self):
        d = np.array([1, 1, 0, 0, 1, 1, 0, 1, 1], np.uint16)
        r, f = find_ttl_edges(lambda a, b: d[a:b], len(d), bit=0, chunk_size=4)
        # a pulse already high at sample 0 has no rising edge
        np.testing.assert_array_equal(r, [4, 7])
        np.testing.assert_array_equal(f, [6, 9])


class TestWindows(unittest.TestCase):
    def test_blank_and_merge(self):
        w = blank_windows([100, 105, 1000], FS, 0.5, 1.5, n_samples=1020)
        np.testing.assert_array_equal(w, [[85, 145], [90, 150], [985, 1020]])
        np.testing.assert_array_equal(merge_windows(w), [[85, 150], [985, 1020]])
        np.testing.assert_array_equal(extend_windows([[10, 20], [25, 30]], after=5), [[10, 35]])
        np.testing.assert_array_equal(extend_windows([[10, 20], [40, 50]], after=5), [[10, 25], [40, 55]])
        self.assertEqual(merge_windows(np.zeros((0, 2))).shape, (0, 2))

    def test_pairs_outside_windows(self):
        starts, stops = [5, 50, 200], [8, 52, 210]
        s, e = pairs_outside_windows(starts, stops, 1000, np.array([[1048, 1060]]))
        np.testing.assert_array_equal(s, [5, 200])
        np.testing.assert_array_equal(e, [8, 210])
        s, e = pairs_outside_windows(starts, stops, 1000, None)
        np.testing.assert_array_equal(s, starts)


class TestDelivered(unittest.TestCase):
    def test_score_separates_delivered(self):
        x, frames = synthetic_stim_recording(delivered=(True, False, True))
        score, peak = score_artifacts(lambda a, b: x[a:b], frames, FS, len(x), gain_uv=0.195)
        self.assertGreater(score[0], 100)
        self.assertGreater(score[2], 100)
        self.assertLess(score[1], 10)
        self.assertGreater(peak[0], 1000)
        ev = build_stim_events(frames, np.array(frames) + 6, FS, len(x),
                               get_stim_artifact_config({"stim_artifact": {"enabled": True}}), score, peak, 0)
        self.assertEqual(ev["events"]["delivered"], [True, False, True])
        self.assertEqual(ev["events"]["blanked"], [True, False, True])
        self.assertEqual(len(ev["blank_windows"]), 2)
        self.assertIsNone(ev["events"]["blank_start_frame"][1])


class TestExtraBlank(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_extra_blank_s(None), [])
        self.assertEqual(parse_extra_blank_s([]), [])
        self.assertEqual(parse_extra_blank_s([1.0, 1.5]), [[1.0, 1.5]])
        self.assertEqual(parse_extra_blank_s([[1, 2], [3.5, 4]]), [[1.0, 2.0], [3.5, 4.0]])
        for bad in ([[2, 1]], [[1, 1]], [[-1, 1]], [1, 2, 3], "soon", [[1, 2, 3]]):
            with self.assertRaises(ValueError):
                parse_extra_blank_s(bad)
        self.assertEqual(get_stim_artifact_config(None, None)["extra_blank_s"], [])

    def test_windows_merged_and_clipped(self):
        cfg = get_stim_artifact_config(
            {"stim_artifact": {"enabled": True, "extra_blank_s": [[0.5, 0.6], [0.55, 0.7], [2.9, 5.0]]}})
        w = extra_blank_windows(cfg, FS, int(3 * FS))
        np.testing.assert_array_equal(w, [[15000, 21000], [87000, 90000]])
        with self.assertWarns(UserWarning):
            w = extra_blank_windows(dict(cfg, extra_blank_s=[[10, 11]]), FS, int(3 * FS))
        self.assertEqual(w.shape, (0, 2))

    def test_in_stim_events_and_blanked(self):
        x, frames = synthetic_stim_recording()
        x[ms(1000):ms(1200)] += 30000                       # a long transient with no TTL
        x = np.clip(x, -32764, 32764)
        site = {"stim_artifact": {"enabled": True, "extra_blank_s": [[0.99, 1.25]]}}
        cfg = get_stim_artifact_config(site)
        score, peak = score_artifacts(lambda a, b: x[a:b], frames, FS, len(x))
        ev = build_stim_events(frames, np.array(frames) + 6, FS, len(x), cfg, score, peak, 0)
        self.assertEqual(ev["extra_blank_windows"], [[ms(990), ms(1250)]])
        self.assertEqual(len(ev["blank_windows"]), 4)        # 3 pulses + the manual window
        self.assertIn([ms(990), ms(1250)], ev["blank_windows"])
        self.assertIn("manual blank", stim_events_summary(ev))
        # pulses are untouched by the extra window
        self.assertEqual(ev["events"]["blanked"], [True, True, True])
        sos = make_bandpass_sos(FS)
        out = blank_and_filter(x, np.asarray(ev["blank_windows"]), sos)
        self.assertTrue((out[ms(990):ms(1250)] == 0).all())
        # without the window the step rings in the filtered data; with it the edges stay small
        ringing = blank_and_filter(x, np.asarray([w for w in ev["blank_windows"] if w != [ms(990), ms(1250)]]), sos)
        self.assertGreater(np.abs(ringing[ms(1195):ms(1205)]).max(), 2000)
        self.assertLess(np.abs(out[ms(1250):ms(1260)]).max(), 500)
        # nwb blank intervals include it
        self.assertIn((0.99, 1.25), stim_nwb_rows(ev)[2])

    def test_extra_artifacts_located_at_the_jump(self):
        x, frames = synthetic_stim_recording()
        jump = ms(1000) + 7                                   # a step with no TTL, common to all channels
        x[jump:jump + ms(150)] -= 25000
        x = np.clip(x, -32764, 32764)
        read = lambda a, b: x[a:b]
        on, score = find_jump_onset(read, ms(1010), FS, len(x), search_ms=50)
        self.assertEqual(on, jump)
        self.assertGreater(score, 100)
        self.assertEqual(parse_extra_artifacts_s(1.0), [1.0])
        self.assertEqual(parse_extra_artifacts_s([2, 1.0]), [1.0, 2.0])
        with self.assertRaises(ValueError):
            parse_extra_artifacts_s([-1])
        cfg = get_stim_artifact_config({"stim_artifact": {"enabled": True, "extra_artifacts_s": [1.01, 2.0, 99]}})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            found = find_extra_artifacts(read, cfg, FS, len(x))
        np.testing.assert_array_equal(found[0], [jump])       # nothing at 2.0 s; 99 s is outside
        self.assertEqual(found[2], [1.01])
        self.assertEqual(len(w), 2)
        score, peak = score_artifacts(read, frames, FS, len(x))
        ev = build_stim_events(frames, np.array(frames) + 6, FS, len(x), cfg, score, peak, 0, extra_artifacts=found)
        ea = ev["extra_artifacts"]
        self.assertEqual(ea["onset_frame"], [jump])
        self.assertEqual((ea["blank_start_frame"], ea["blank_stop_frame"]), ([jump - ms(0.5)], [jump + ms(1.0)]))
        self.assertIn([jump - ms(0.5), jump + ms(1.0)], ev["blank_windows"])
        self.assertEqual(len(ev["events"]["start_frame"]), 3)  # not a stim pulse
        self.assertIn("without a TTL", stim_events_summary(ev))
        # no extra artefacts given: empty lists
        self.assertEqual(build_stim_events(frames, np.array(frames) + 6, FS, len(x), cfg, score, peak, 0)
                         ["extra_artifacts"]["onset_frame"], [])

    def test_disabled_ignores_extra(self):
        cfg = get_stim_artifact_config({"stim_artifact": {"enabled": False, "extra_blank_s": [[1, 2]]}})
        ev = build_stim_events([], [], FS, int(3 * FS), cfg)
        self.assertEqual(ev["extra_blank_windows"], [])
        self.assertEqual(ev["blank_windows"], [])


class TestBouts(unittest.TestCase):
    def test_group_and_annotate(self):
        frames = (np.array([0, 1, 2, 3, 20, 21, 22, 60]) * FS).astype(int)
        b = group_bouts(frames, FS, 5.0)
        np.testing.assert_array_equal(b, [0, 0, 0, 0, 1, 1, 1, 2])
        cfg = get_stim_artifact_config({"stim_artifact": {"enabled": True}})
        score = np.array([500, 500, 3, 500, 500, 500, 500, 500.])
        ev = build_stim_events(frames, frames + 6, FS, int(70 * FS), cfg, score, score)
        self.assertEqual([x["n_pulses"] for x in ev["bouts"]], [4, 3, 1])
        self.assertEqual([x["n_delivered"] for x in ev["bouts"]], [3, 3, 1])
        self.assertAlmostEqual(ev["bouts"][0]["rate_hz"], 1.0, places=3)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ann = annotate_bouts(ev["bouts"], [{"bout": 0, "polarity": "-", "current_ua": 500},
                                               {"bout": 1, "polarity": "+"}])
            self.assertTrue(any("annotations" in str(x.message) for x in w))  # 2 annotations, 3 bouts
        self.assertEqual([a["polarity"] for a in ann], ["-", "+", "unknown"])
        self.assertEqual(ann[0]["current_ua"], 500.0)
        self.assertTrue(np.isnan(ann[2]["current_ua"]))

    def test_nwb_rows(self):
        frames = (np.array([1, 2, 30]) * FS).astype(int)
        cfg = get_stim_artifact_config({"stim_artifact": {"enabled": True}})
        ev = build_stim_events(frames, frames + 6, FS, int(40 * FS), cfg, np.array([500, 2, 500.]),
                               np.array([6000, 10, 6000.]), 0)
        pulses, bouts, blanks = stim_nwb_rows(ev, [{"bout": 0, "polarity": "-", "current_ua": 500}], "DIGITAL-IN-01")
        self.assertEqual(len(pulses), 3)
        self.assertAlmostEqual(pulses[0]["start_time"], 1.0)
        self.assertAlmostEqual(pulses[0]["stop_time"], 1.0 + 6 / FS)
        self.assertEqual([p["delivered"] for p in pulses], [True, False, True])
        self.assertAlmostEqual(pulses[0]["blank_start"], 1.0 - 0.0005)
        self.assertTrue(np.isnan(pulses[1]["blank_start"]))
        self.assertEqual(pulses[0]["polarity"], "-")
        self.assertEqual(pulses[2]["polarity"], "unknown")
        self.assertEqual([b["n_delivered"] for b in bouts], [1, 1])
        self.assertEqual(len(blanks), 2)
        # empty event list still gives (empty) rows
        empty = build_stim_events([], [], FS, 100, cfg)
        self.assertEqual(stim_nwb_rows(empty), ([], [], []))


class TestBlankAndFilter(unittest.TestCase):
    def setUp(self):
        self.sos = make_bandpass_sos(FS, 300, 8000, 5)
        self.x, self.frames = synthetic_stim_recording()
        self.windows = merge_windows(blank_windows(self.frames, FS, 0.5, 1.5, len(self.x)))

    def test_far_from_stims_matches_plain_bandpass(self):
        out = blank_and_filter(self.x, self.windows, self.sos)
        ref = sosfiltfilt(self.sos, self.x, axis=0)
        # 0.75-1.25 s lies between pulses at 0.5 and 1.5 s, >250 ms from each:
        # the plain filter's own edge effects from the artefacts have died out
        sl = slice(int(0.75 * FS), int(1.25 * FS))
        noise = np.std(ref[sl])
        self.assertLess(np.max(np.abs(out[sl] - ref[sl])), 0.01 * noise)

    def test_blank_windows_are_zero_and_no_ringing_before_pulse(self):
        out = blank_and_filter(self.x, self.windows, self.sos)
        noise = np.median(np.abs(out[int(0.1 * FS):int(0.4 * FS)])) / 0.6745
        ref = sosfiltfilt(self.sos, self.x, axis=0)
        for f, (a, b) in zip(self.frames, self.windows):
            self.assertTrue(np.all(out[a:b] == 0))
            pre = out[f - ms(10):a]
            self.assertLess(np.max(np.abs(pre)), 6 * noise)          # clean before the pulse
            self.assertGreater(np.max(np.abs(ref[f - ms(10):a])), 20 * noise)  # plain bandpass rings
            post = out[f + ms(5):f + ms(20)]
            self.assertLess(np.median(np.abs(post)) / 0.6745, 1.5 * noise)

    def test_spike_after_blank_survives(self):
        # a spike at 4 ms after the pulse goes through the same linear filtering
        spike = np.zeros_like(self.x)
        f = self.frames[1]
        wf = -np.exp(-0.5 * ((np.arange(ms(1.5)) - ms(0.3)) / 4.0) ** 2) * 400
        spike[f + ms(4) - ms(0.3):f + ms(4) - ms(0.3) + len(wf), :] += wf[:, None]
        out = blank_and_filter(self.x + spike, self.windows, self.sos) - blank_and_filter(self.x, self.windows, self.sos)
        ref = sosfiltfilt(self.sos, spike, axis=0)
        sl = slice(f + ms(3), f + ms(6))
        rec = (out[sl] * ref[sl]).sum() / (ref[sl] ** 2).sum()
        self.assertGreater(rec, 0.9)

    def test_chunking_does_not_change_output(self):
        whole = blank_and_filter(self.x, self.windows, self.sos)
        read = lambda a, b: self.x[a:b]
        margin = ms(50)
        for chunk in (ms(100), 29999, ms(1000)):
            parts = [blank_and_filter_chunk(read, len(self.x), s, min(s + chunk, len(self.x)),
                                            self.windows, self.sos, margin)
                     for s in range(0, len(self.x), chunk)]
            out = np.concatenate(parts)
            noise = np.std(whole[int(0.1 * FS):int(0.4 * FS)])
            self.assertEqual(out.shape, whole.shape)
            self.assertLess(np.max(np.abs(out - whole)), 0.01 * noise)

    def test_short_pieces_are_blanked(self):
        w = np.array([[1000, 1060], [1070, 1130]])  # 10-sample gap: too short to filter
        out = blank_and_filter(self.x, w, self.sos)
        self.assertTrue(np.all(out[1000:1130] == 0))


if __name__ == "__main__":
    unittest.main()
