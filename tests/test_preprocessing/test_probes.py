import os
import tempfile
import unittest

from songephys.preprocess.ephys.probes import (
    PROBE_FILES, REPO_PROBE_DIR, find_probe_file, get_probe_name, load_site_probe)


class TestProbeName(unittest.TestCase):
    def test_site_overrides_project(self):
        self.assertEqual(get_probe_name({'probe': 'H10'}, {'EPHYS': {'PROBE': 'H15'}}), 'H10')

    def test_project_when_site_has_none(self):
        self.assertEqual(get_probe_name({}, {'EPHYS': {'PROBE': 'H15'}}), 'H15')
        self.assertEqual(get_probe_name({'probe': None}, {'EPHYS': {'PROBE': 'H15'}}), 'H15')

    def test_older_key_and_default(self):
        self.assertEqual(get_probe_name({}, {'EPHYS': {'PROBETYPE': 'P1'}}), 'P1')
        self.assertEqual(get_probe_name({}, {'EPHYS': {'PROBE': 'H15', 'PROBETYPE': 'P1'}}), 'H15')
        self.assertEqual(get_probe_name({}, {'EPHYS': {}}, default='H15'), 'H15')


class TestFindProbeFile(unittest.TestCase):
    def test_every_known_probe_is_in_the_repo(self):
        for key, filename in PROBE_FILES.items():
            self.assertEqual(find_probe_file(key), os.path.join(REPO_PROBE_DIR, filename))

    def test_kilosort_folder_wins_over_repo(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, PROBE_FILES['H15'])
            open(p, 'w').close()
            self.assertEqual(find_probe_file('H15', [d]), p)
            self.assertEqual(find_probe_file('h15', [d]), p)       # case doesn't matter

    def test_file_name_and_path(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 'new_probe.json')
            open(p, 'w').close()
            self.assertEqual(find_probe_file('new_probe.json', [d]), p)
            self.assertEqual(find_probe_file(p), p)

    def test_unknown_probe_is_an_error(self):
        with self.assertRaisesRegex(ValueError, 'H99'):
            find_probe_file('H99')


class TestLoad(unittest.TestCase):
    def test_known_probes_load_with_kilosort(self):
        for key in ('P1', 'H10', 'H15'):    # microprobes_16ch.prb isn't in Kilosort's .prb format
            prb, name, _ = load_site_probe({'probe': key}, {'EPHYS': {'PROBE': 'H15'}})
            self.assertEqual(name, key)
            self.assertEqual(len(prb['chanMap']), prb['n_chan'])


if __name__ == '__main__':
    unittest.main()
