import os
import shutil
import tempfile
import unittest

import numpy as np

from format.lazy import (
    LazySignal,
    LazySite,
    LoadedTimeSlice,
    LoadedTimeSliceDict,
    time_to_index,
    create_lazy_signals_from_nix,
)


class TestLazy(unittest.TestCase):

    def setUp(self):
        self.t = np.array([0, 1, 2, 3])
        self.arr = np.array([
            [0, 0],
            [1, 2],
            [2, 3],
            [4, 4]
        ])

        self.test_dir = tempfile.mkdtemp()
        self.sig_path = os.path.join(self.test_dir, "temp_test.memmap")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_time_to_index(self):
        self.assertEqual(time_to_index(1.0, 30000), 30000)
        self.assertEqual(time_to_index(1.19, 10), 12)
        self.assertEqual(time_to_index(1.12, 10), 11)
        self.assertEqual(time_to_index(1.15, 10), 12)

    def test_loaded_time_slice_dict(self):
        x = LoadedTimeSliceDict(1.0, 2.0)
        self.assertEqual(x._t0, 1.0)
        self.assertEqual(x._t1, 2.0)

    def test_loaded_time_slice(self):
        x = LoadedTimeSlice(self.t, self.arr)
        np.testing.assert_array_equal(self.t, x.t)
        np.testing.assert_array_equal(self.arr, x.data)

    def test_lazy_signal_write(self):
        self.assertFalse(os.path.exists(self.sig_path))
        x = LazySignal(
            N=4,
            M=2,
            file=self.sig_path,
            sampling_rate=1,
            dtype="float32"
        )
        self.assertTrue(os.path.exists(self.sig_path))
        # np.testing.assert_array_almost_equal(x.read_samples(4, 0), np.zeros((4, 2)))

        x.write(self.arr, n=4, offset=0)
        np.testing.assert_array_equal(x.read_samples(4, 0), self.arr)

        x.write(np.array([[0, 0]]), n=1, offset=2)
        np.testing.assert_array_equal(x.read_samples(4, 0), [
            [0, 0],
            [1, 2],
            [0, 0],
            [4, 4]
        ])


class TestLazySignal(unittest.TestCase):
    """
    Test reading from LazySignal object
    """

    def setUp(self):
        self.t = np.array([0, 1, 2, 3])
        self.arr = np.array([
            [0, 0],
            [1, 2],
            [2, 3],
            [4, 4]
        ])

        self.test_dir = tempfile.mkdtemp()
        self.sig_path = os.path.join(self.test_dir, "temp_test.memmap")
        self.sig = LazySignal(
            N=4,
            M=2,
            file=self.sig_path,
            sampling_rate=1,
            dtype="float32"
        )
        self.sig.write(self.arr, n=4, offset=0)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_lazy_signal_time_slice(self):
        res = self.sig.time_slice(-1, 2)
        np.testing.assert_array_equal(res.t, [-1, 0, 1])
        np.testing.assert_array_equal(res.data, [
            [0, 0],
            [0, 0],
            [1, 2]
        ])

        res = self.sig.time_slice(0, 4)
        np.testing.assert_array_equal(res.t, [0, 1, 2, 3])
        np.testing.assert_array_equal(res.data, self.arr)

        res = self.sig.time_slice(0, 6)
        np.testing.assert_array_equal(res.t, [0, 1, 2, 3, 4, 5])
        np.testing.assert_array_equal(res.data, [
            [0, 0],
            [1, 2],
            [2, 3],
            [4, 4],
            [0, 0],
            [0, 0],
        ])

    def test_lazy_signal_batches(self):
        batches = self.sig.batches(1)
        for i in range(4):
            idx, data = next(batches)
            self.assertEqual(idx, i)
            self.assertEqual(data.shape, (1, 2))
        with self.assertRaises(StopIteration):
            next(batches)

        batches = self.sig.batches(2)
        for i in range(2):
            idx, data = next(batches)
            self.assertEqual(idx, i * 2)
            self.assertEqual(data.shape, (2, 2))
        with self.assertRaises(StopIteration):
            next(batches)

        batches = self.sig.batches(3)
        idx, data = next(batches)
        self.assertEqual(idx, 0)
        self.assertEqual(data.shape, (3, 2))
        idx, data = next(batches)
        self.assertEqual(idx, 3)
        self.assertEqual(data.shape, (1, 2))
        with self.assertRaises(StopIteration):
            next(batches)


class TestLazySite(unittest.TestCase):
    """
    Test reading from LazySite object
    """

    def setUp(self):
        self.t = np.array([0, 1, 2, 3])
        self.arr = np.array([
            [0, 0],
            [1, 2],
            [2, 3],
            [4, 4]
        ])

        self.test_dir = tempfile.mkdtemp()
        self.sig_path = os.path.join(self.test_dir, "temp_test.memmap")

        self.i = 0
        def _dummy_signal(size):
            self.i += 1
            return LazySignal(
                N=size[0],
                M=size[1],
                file=os.path.join(self.test_dir, "temp_{}".format(self.i)),
                sampling_rate=1,
                dtype="float32"
            )

        self.sig11 = _dummy_signal((4, 16))
        self.sig12 = _dummy_signal((5, 16))
        self.sig21 = _dummy_signal((4, 2))
        self.sig22 = _dummy_signal((5, 2))

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_init(self):
        times = [0, 6]
        site = LazySite(
            times,
            raw=[self.sig11, self.sig12],
            mic=[self.sig21, self.sig22]
        )

        np.testing.assert_array_equal(site.signal_times, times)
        np.testing.assert_array_equal(
            site._signals["raw"],
            [self.sig11, self.sig12]
        )
        np.testing.assert_array_equal(
            site._signals["mic"],
            [self.sig21, self.sig22]
        )

    def test_max_time(self):
        times = [0, 6]
        site = LazySite(
            times,
            raw=[self.sig11, self.sig12],
            mic=[self.sig21, self.sig22]
        )
        self.assertEqual(site.max_time(), 11)

    def test_get_lazy_signals(self):
        """Test that LazySite._get_lazy_signals fetches the right signal"""
        times = [0, 6]
        site = LazySite(
            times,
            raw=[self.sig11, self.sig12],
            mic=[self.sig21, self.sig22]
        )

        # # Grab a time before the first
        # t, sigs = site._get_lazy_signals(-1)
        # self.assertEqual(t, times[0])
        # self.assertListEqual(list(sigs.keys()), ["raw", "mic"])
        # self.assertEqual(sig["raw"], self.sig11)
        # self.assertEqual(sig["mic"], self.sig21)

        # Grab a time during first
        t, sigs = site._get_lazy_signals(1.2)
        self.assertEqual(t, times[0])
        self.assertSetEqual(set(sigs.keys()), set(["raw", "mic"]))
        self.assertEqual(sigs["raw"], self.sig11)
        self.assertEqual(sigs["mic"], self.sig21)

        # Grab a time between first and second
        t, sigs = site._get_lazy_signals(5.5)
        self.assertEqual(t, times[0])
        self.assertSetEqual(set(sigs.keys()), set(["raw", "mic"]))
        self.assertEqual(sigs["raw"], self.sig11)
        self.assertEqual(sigs["mic"], self.sig21)
        self.assertEqual(sigs["mic"], self.sig21)

        # Grab a time during second
        t, sigs = site._get_lazy_signals(10)
        self.assertEqual(t, times[1])
        self.assertSetEqual(set(sigs.keys()), set(["raw", "mic"]))
        self.assertEqual(sigs["raw"], self.sig12)
        self.assertEqual(sigs["mic"], self.sig22)

        # Grab a time after second
        t, sigs = site._get_lazy_signals(20)
        self.assertEqual(t, times[1])
        self.assertSetEqual(set(sigs.keys()), set(["raw", "mic"]))
        self.assertEqual(sigs["raw"], self.sig12)
        self.assertEqual(sigs["mic"], self.sig22)

    def test_time_slice(self):
        """Test that time_slice fetches the right signal"""
        times = [0, 6]
        site = LazySite(
            times,
            raw=[self.sig11, self.sig12],
            mic=[self.sig21, self.sig22]
        )

        # Grab a time during first
        sigs = site.time_slice(0.0, 3.0)
        np.testing.assert_array_equal(sigs["raw"].t, [0, 1, 2])
        np.testing.assert_array_equal(sigs["raw"].data,
                self.sig11.read_samples(3, 0))
        np.testing.assert_array_equal(sigs["mic"].t, [0, 1, 2])
        np.testing.assert_array_equal(sigs["mic"].data,
                self.sig21.read_samples(3, 0))

        # Time slice that extends past first
        sigs = site.time_slice(0.0, 5.5)
        np.testing.assert_array_equal(sigs["raw"].t, [0, 1, 2, 3, 4, 5])
        np.testing.assert_array_equal(
            sigs["raw"].data,
            np.concatenate([
                self.sig11.read_samples(4, 0),
                np.zeros((2, self.sig11.M))
            ])
        )

        # Time slice that extends into second
        sigs = site.time_slice(2.0, 8.0)
        np.testing.assert_array_equal(sigs["raw"].t, [2, 3, 4, 5, 6, 7])
        np.testing.assert_array_equal(
            sigs["raw"].data,
            np.concatenate([
                self.sig11.read_samples(2, 2),
                np.zeros((2, self.sig11.M)),
                self.sig12.read_samples(2, 0),
            ])
        )

        # Time slice that extends past second
        sigs = site.time_slice(9.0, 14.0)
        np.testing.assert_array_equal(sigs["raw"].t, [9, 10, 11, 12, 13])
        np.testing.assert_array_equal(
            sigs["raw"].data,
            np.concatenate([
                self.sig12.read_samples(2, 3),
                np.zeros((3, self.sig11.M)),
            ])
        )
