'''
Created on 5.2.2012

@author: Tomas Machalek <tomas.machalek@gmail.com>
'''

import unittest

from imreg.inform.entropy_map import *

class TestEntropyMap(unittest.TestCase):
    """
    """
    
    def setUp(self):
        """
        """
        self.image = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 0]]) 
    
    def test_calc_point(self):
        """
        """
        emap = EntropyMap(self.image, 1)
        entropy = emap.calc_point(1, 1)
        self.assertAlmostEqual(entropy, 0.3488, 4)
        
        
        
if __name__ == "__main__":
    unittest.main()