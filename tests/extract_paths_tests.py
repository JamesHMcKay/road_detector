from road_detector import prob_road
import unittest
import pycodestyle
import numpy as np

class Test_extract_paths(unittest.TestCase):
    def test_prob_road_zeros(self):
        image = np.zeros([10,10])
        result = prob_road(image, 1, 1)
        print 'results = ' + str(result)
        self.assertEqual(result, 1.0)

    def test_prob_road_ones(self):
            prob = 1.0 - 0.11111111 * (9.0)
            image = np.ones([10,10])
            image = image * 255.0
            result = prob_road(image, 1, 1)
            print 'results = ' + str(result)
            self.assertEqual(result, prob)

if __name__ == '__main__':
    unittest.main()