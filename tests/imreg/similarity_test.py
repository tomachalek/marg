
import unittest
import numpy as np
import mahotas
import os

from imreg import similarity

class SimilarityTest(unittest.TestCase):
    """
    """

    def test_ncc_negative_corr(self):
        """
        """
        a = np.array([1, 2, 3, 0, 4, -1, 2])
        b = np.array([-1, -2, -3, 0, -4, 1, -2])
        ans = similarity.ncc(a, b)
        self.assertEquals(ans, -1)

    def test_ncc_positive_corr(self):
        """
        """
        a = np.array([1, 2, 3, 0, 4, -1, 2])
        b = np.array([1, 2, 3, 0, 4, -1, 2])
        ans = similarity.ncc(a, b)
        self.assertEquals(ans, 1)


    def test_ncc_positive_corr2(self):
        """
        """
        a = np.array([1, 2, 3, 0, 4, -1, 2])
        b = np.array([2, 4, 6, 0, 8, -2, 4])
        ans = similarity.ncc(a, b)
        self.assertEquals(ans, 1)


    def test_ncc_small_corr(self):
        """
        """
        a = np.array([1, 2, 13, 30, 124, 347, 22])
        b = np.array([882, -900, -902, -237, -337, -17, 180])
        ans = similarity.ncc(a, b)
        self.assertAlmostEquals(ans, -0.0822, 3)


    def test_ncc_images(self):
        img1 = mahotas.imread('%s/../../test-data/lena-64.jpg' % os.path.dirname(__file__), as_grey=True)
        img2 = mahotas.imread('%s/../../test-data/lena-64.jpg' % os.path.dirname(__file__), as_grey=True)
        ans = similarity.ncc(img1, img2)
        self.assertEquals(ans, 1)