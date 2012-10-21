# Copyright (C) 2012 Tomas Machalek
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import numpy as np

from imreg.inform.mi import mutual_information


class TestMutualInformation(unittest.TestCase):
    """
    """

    def test_identical_data_mi(self):
        a1 = np.array([1, 0, 1, 0])
        a2 = np.array([1, 0, 1, 0])
        mi = mutual_information(a1, a2, domain=2, smoothing=0)
        self.assertAlmostEqual(mi, 1, 4)

    def test_zero_mi(self):
        a1 = np.array([0, 1, 0, 1])
        a2 = np.array([1, 1, 1, 1])
        mi = mutual_information(a1, a2, domain=2, smoothing=0)
        self.assertEqual(mi, 0)


    def test_sample_data_with_smoothing(self):
        a = [0, 1, 1, 2, 0, 3, 3, 2, 0, 1, 2, 3, 1, 1, 1, 2, 3, 3, 2]
        b = [0, 1, 1, 2, 0, 3, 3, 2, 0, 1, 2, 3, 1, 1, 1, 2, 3, 1, 3]
        ans = mutual_information(a, b, 4, smoothing=1)
        self.assertAlmostEqual(0.4158, ans, 4)

    def test_sample_data_without_smoothing(self):
        a = [0, 1, 1, 2, 0, 3, 3, 2, 0, 1, 2, 3, 1, 1, 1, 2, 3, 3, 2]
        b = [0, 1, 1, 2, 0, 3, 3, 2, 0, 1, 2, 3, 1, 1, 1, 2, 3, 1, 3]
        ans = mutual_information(a, b, 4, smoothing=0)
        self.assertAlmostEqual(1.5513, ans, 4)