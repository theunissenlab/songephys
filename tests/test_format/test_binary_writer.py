import os
import tempfile
import unittest

import numpy as np

from songephys.format.binary_writer import write_recording_binary


class FakeRecording:
    """Minimal stand-in for a SpikeInterface recording"""

    def __init__(self, segments, fail_at=None, fs=30000.0):
        self.segments = segments
        self.fail_at = fail_at

    def get_num_segments(self):
        return len(self.segments)

    def get_num_samples(self, segment_index=0):
        return self.segments[segment_index].shape[0]

    def get_num_channels(self):
        return self.segments[0].shape[1]

    def get_sampling_frequency(self):
        return 30000.0

    def get_traces(self, start_frame=None, end_frame=None, segment_index=0):
        if self.fail_at is not None and start_frame <= self.fail_at < end_frame:
            raise RuntimeError("boom")
        return self.segments[segment_index][start_frame:end_frame]


class TestWriteRecordingBinary(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "out", "x.bin")
        rng = np.random.default_rng(0)
        self.a = rng.normal(0, 1000, (10_007, 5)).astype(np.float32)
        self.b = rng.normal(0, 1000, (3_001, 5)).astype(np.float32)

    def tearDown(self):
        self.tmp.cleanup()

    def test_matches_concatenated_cast(self):
        n, c = write_recording_binary(FakeRecording([self.a, self.b]), self.path,
                                      dtype=np.int16, chunksize=997, max_workers=4)
        self.assertEqual((n, c), (13_008, 5))
        out = np.fromfile(self.path, dtype=np.int16).reshape(-1, 5)
        expected = np.concatenate([self.a, self.b]).astype(np.int16)
        np.testing.assert_array_equal(out, expected)
        self.assertFalse(os.path.exists(self.path + ".partial"))

    def test_error_in_chunk_raises_and_leaves_nothing(self):
        with self.assertRaises(RuntimeError):
            write_recording_binary(FakeRecording([self.a], fail_at=5_000), self.path,
                                   chunksize=1000, max_workers=3)
        self.assertFalse(os.path.exists(self.path))
        self.assertFalse(os.path.exists(self.path + ".partial"))

    def test_single_chunk_and_single_worker(self):
        write_recording_binary(FakeRecording([self.a[:50]]), self.path, chunksize=30000, max_workers=1)
        out = np.fromfile(self.path, dtype=np.int16).reshape(-1, 5)
        np.testing.assert_array_equal(out, self.a[:50].astype(np.int16))


if __name__ == "__main__":
    unittest.main()
