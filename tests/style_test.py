import unittest
import pycodestyle
import os


class TestCodeFormat(unittest.TestCase):
    def test_conformance(self):
        style = pycodestyle.StyleGuide(quiet=False, config_file='tox.ini')
        files = os.listdir('road_detector/')
        files = list(filter(lambda x: 'py' in x and 'pyc' not in x, files))
        for i in range(len(files)):
            files[i] = 'road_detector/' + files[i]
        result = style.check_files(files)
        self.assertEqual(result.total_errors, 0,
        "Found code style errors (and warnings).")

if __name__ == '__main__':
    unittest.main()