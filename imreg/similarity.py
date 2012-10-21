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

"""
@author: Tomas Machalek <tomas.machalek@uhk.cz>
"""

import numpy as np
import image

def sad(image1, image2):
    """
    Sum of absolute differences

    Parameters
    ----------
    image1 : numpy.ndarray
    image2 : numpy.ndarray
    """
    pixels1 = np.ravel(image1)
    pixels2 = np.ravel(image2)
    return np.sum(abs(pixels1 - pixels2)) / (np.size(pixels1, 0))


def ssd(image1, image2):
    """
    Parameters
    ----------
    image1 : numpy.ndarray
    image2 : numpy.ndarray
    """
    pixels1 = np.ravel(image1)
    pixels2 = np.ravel(image2)
    return np.sum((pixels1 - pixels2) * (pixels1 - pixels2)) / (np.size(pixels1, 0))


def ncc(image1, image2):
    """
    Parameters
    ----------
    image1 : numpy.ndarray
    image2 : numpy.ndarray
    """
    return np.corrcoef(np.ravel(image1), np.ravel(image2))[1][0]


def ncc_min(image1, image2):
    """
    Modified cross correlation so it can be used along with the
    minimizing optimizer. It returns -abs(ncc(x,y))

    Parameters
    ----------
    image1 : numpy.ndarray
    image2 : numpy.ndarray
    """
    return -abs(np.corrcoef(np.ravel(image1), np.ravel(image2))[1][0])


class CostFunction(object):
    """
    This class represents abstract meta-cost-function with attached tiles
    which are transformed according to the "position" parameter.
    """

    def __init__(self, cost_function, tile1, tile2):
        """
        """
        self.cost_function = cost_function
        self.tile1 = tile1
        self.tile2 = tile2

    def __call__(self, position):
        self.tile2.x = int(round(position[0]))
        self.tile2.y = int(round(position[1]))
        coord_z = int(round(position[2])) if len(position) == 3 else 0
        self.tile1.z = coord_z
        self.tile2.z = coord_z
        img1, img2 = image.get_overlapping_regions(self.tile1, self.tile2)
        return self.cost_function(img1, img2)