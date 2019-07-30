import unittest
from unittest import mock

import numpy as np

from preprocess.ephys.detect import (
    detect_spike_times,
    extract_spikes,
    get_spike_snippets
)
from preprocess.ephys.filter import (
    _consolidate_intervals,
    basic_filter, 
    common_average_reference,
    create_noise_intervals,
    filter_snippets,
    find_correlated_events,
)


class TestFilter(unittest.TestCase):
    """Test filter functions

    General approach will be to take some real data, manually validate the data,
    then put it into the test
    """

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_common_average_reference(self):
        common_noise = np.random.random(size=100)

        # Ground truth array has mean zero across channels
        ground_truth = np.array([
            np.arange(100),
            -np.arange(100),
            np.zeros(100)
        ]).T
        
        test_signal = ground_truth + common_noise[:, None]

        np.testing.assert_array_almost_equal(
            common_average_reference(test_signal),
            ground_truth
        )

    def test_find_correlated_events(self):
        """Test that correlated events are located with correct correlation matrices
        """
        sig = np.array([
            [3, 1, 0, 0, 0, 1, 3, 1, 0, 0, 0, 1, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [3, 1, 0, 0, 0, 1, 3, 1, 0, 0, 0, -1, -3, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]).T
        indices, snippets, corrs = find_correlated_events(sig, 2)

        np.testing.assert_array_equal(indices, [6, 12])
        np.testing.assert_array_equal(sig[4:8], snippets[0])
        np.testing.assert_array_equal(sig[10:14], snippets[1])
        np.testing.assert_array_equal(corrs[0], [[0, 1], [1, 0]])
        np.testing.assert_array_equal(corrs[1], [[0, -1], [-1, 0]])

    def test_consolidate_intervals(self):
        intervals = [(0, 4), (2, 7), (9, 11)]
        self.assertSequenceEqual(_consolidate_intervals(intervals), [(0, 7), (9, 11)])

        intervals = [(3, 5), (5, 7), (9, 11)]
        self.assertSequenceEqual(_consolidate_intervals(intervals), [(3, 7), (9, 11)])

    def test_create_noise_intervals(self):
        windows = create_noise_intervals([4, 7, 18], 4)
        np.testing.assert_array_equal(windows, [[0, 11], [14, 22]])

    def test_filter_snippets(self):
        sig = np.array([
            np.arange(20),
            np.arange(20)
        ]).T

        noise_snippets = [
            np.random.random(size=(20, 16)),
            np.random.random(size=(30, 16)),
            np.random.random(size=(40, 16)),
            np.random.random(size=(50, 16)),
        ]

        # Goal: test that signal filtering works by making sure we get the piece
        # that we expect back out
        with mock.patch("config.FILTER_BLOCK", 1):
            result = filter_snippets(noise_snippets, lambda x: x)
            self.assertEqual(len(result), len(noise_snippets))
            for x, y in zip(result, noise_snippets):
                np.testing.assert_array_equal(x, y)

        with mock.patch("config.FILTER_BLOCK", 100000000):
            result = filter_snippets(noise_snippets, lambda x: x)
            self.assertIsNone(result)


class TestSpikeDetect(unittest.TestCase):
    """Test spike detection"""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_detect_spike_times(self):
        # First spike should be ignored (too close to start)
        sig = np.array([
            3, 1, 0, 0, 0, 1, 3, 1, 0, 0, 
            1, 3, 1, 0, 0, 0, 0, 0, 0, 1,
            3, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        ])
        times = detect_spike_times(sig, threshold_std=1, spike_radius=2)
        np.testing.assert_array_equal(times, [6, 11, 20])

        times = detect_spike_times(sig,
            threshold_std=1,
            spike_radius=2,
            min_spacing=5)
        np.testing.assert_array_equal(times, [6, 20])

    def test_get_spike_snippets(self):
        sig = np.array([
            3, 1, 0, 0, 0, 1, 3, 1, 0, 0, 
            1, 3, 1, 0, 0, 0, 0, 0, 0, 1,
            3, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        ])
        snippets = get_spike_snippets(sig, [6, 11, 12, 15], spike_radius=2)
        self.assertEqual(len(snippets), 4)
        np.testing.assert_array_equal(snippets[0], [0, 1, 3, 1])
        np.testing.assert_array_equal(snippets[1], [0, 1, 3, 1])
        np.testing.assert_array_equal(snippets[2], [1, 3, 1, 0])
        np.testing.assert_array_equal(snippets[3], [0, 0, 0, 0])

    def test_extract_spikes(self):
        # Mock the lazy loading signal loader
        # Two batches, 2 channels
        mock_lazy_loader = mock.MagicMock()
        mock_lazy_loader.batches.return_value = iter([
            (
                0, 
                np.array([
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                    [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1],
                    [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]).T
            ),
            (
                15, 
                np.array([
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]).T
            )
        ])

        # Channel 0: Spikes at 3, 10, 18, 25
        # Channel 1: Spikes at 7, 10
        # Channel 2: No spikes should be detected with threshold = 2std
        # Channel 3: Spike at 5

        result = extract_spikes(
            mock_lazy_loader,
            batch_size=15,
            threshold_std=2,
            spike_radius=2,
            channel_ids=[8, 9, 10, 11]
        )

        np.testing.assert_array_equal(result[8]["indices"], [3, 10, 18, 25])
        np.testing.assert_array_equal(result[9]["indices"], [7, 10])
        np.testing.assert_array_equal(result[10]["indices"], [])
        np.testing.assert_array_equal(result[11]["indices"], [5])

        np.testing.assert_array_equal(result[8]["waveforms"], [[0, 0, 1, 0]] * 4)
        np.testing.assert_array_equal(result[9]["waveforms"], [[0, 0, 1, 0]] * 2)
        np.testing.assert_array_equal(result[10]["waveforms"], np.zeros((0, 4)))
        np.testing.assert_array_equal(result[11]["waveforms"], [[0, 0, 2, 0]])