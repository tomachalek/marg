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
Image manipulation and access and a "tile" abstraction. Tile means
an image with a position in some 2D coordinate system.
Image functions are based on mahotas and numpy.

"""

import re
import math
import os

import mahotas
from scipy import ndimage
import numpy as np

class ImageTile(object):
    """
    Image segment which is able to keep scale-space like representation of an image.
    It has three dimensions
      * x position
      * y position
      * z position = level of gaussian blur applied to the image
    """
    
    def __init__(self, path, scale_depth, scale_from=0, step_coeff=3):
        """
        Parameters
        ----------
        path : str
            path to an image
        scale_depth : int
            how many levels of blur to create
        scale_from : int
            from which level to start (0 = no blur at all)
        step_coeff : float

        """
        self.x, self.y, self.z = (0, 0, 0)
        self.filename = os.path.basename(path)
        self.image = []
        orig_image = mahotas.imread(path, as_grey=True)
        self.width = np.size(orig_image, 1)
        self.height = np.size(orig_image, 0)
        
        if scale_from == 0:
            self.image.append(orig_image)
        else:
            self.image.append(None)
        for i in range(1, scale_depth + 1):
            if i < scale_from:
                self.image.append(None)
            else:
                sigma = (step_coeff * i, step_coeff * i)
                print("sigma: %s" % (sigma,))
                self.image.append(ndimage.gaussian_filter(orig_image, sigma))
        
    def __repr__(self):
        return "tile %dx%d [%d, %d, %d]" % (self.width, self.height, self.x, self.y, self.z)
        
    def export(self, path, z = None):
        """
        Exports selected blur level of the segment

        Parameters
        ----------
        z : int
            level to export (0 = original image)
        """
        if z is None:
            z = [img for img in self.image if self.image][0]
        save_greyscale(path, self.image[z])
        
    def export_all(self, dir_path):
        path = "%s/%s" % (dir_path, self.filename)
        path_mask = re.sub(r'^(.+)\.(\w+)$', r'\1-level-%0*d.\2', path)
        for i in range(len(self.image)):
            if self.image[i] is not None:
                self.export(path_mask % (int(math.ceil(math.log(len(self.image)))), i), i)
                
    
    def get_area(self, area):
        relative_area = list(area)
        relative_area[0] = relative_area[0] - self.x
        relative_area[1] = relative_area[1] - self.y
        
    def get_image(self, z = None):
        if z is None:
            return self.image[int(round(self.z))]
        else:
            return self.image[z]  
        

def save_greyscale(path, image):
    out = []
    for i in range(len(image)):
        row = []
        for j in range(len(image[i])):
            row.append((np.uint8(image[i][j]), np.uint8(image[i][j]), np.uint8(image[i][j])))
        out.append(row)
    out = np.array(out)
    mahotas.imsave(path, out)


    

def tile_overlapping(tile1, tile2):
    """
    expects two squares in form of tuples containing
    (x, y, width, height)
    and returns a region with the same format containing
    an intersection of the two images
    """
    rect1 = (tile1.x, tile1.y, tile1.width, tile1.height)
    rect2 = (tile2.x, tile2.y, tile2.width, tile2.height)

    x1 = int(round(max(rect1[0], rect2[0])))
    x2 = int(round(min(rect1[0] + rect1[2], rect2[0] + rect2[2])))
    y1 = int(round(max(rect1[1], rect2[1])))
    y2 = int(round(min(rect1[1] + rect1[3], rect2[1] + rect2[3])))
    return (x1, y1, abs(x1 - x2), abs(y1 - y2))


def get_overlapping_regions(tile1, tile2):
    """
    Parameters
    ----------
    tile1 : ImageTile
        filex segment
    tile2 : ImageTile
        transformed segment

    Returns
    -------
    numpy.ndarray
        array containing data of overlapping region
    """

    over = tile_overlapping(tile1, tile2)

    t1x1 = over[0] - tile1.x
    t1x2 = over[0] - tile1.x + over[2]
    t1y1 = over[1] - tile1.y
    t1y2 = over[1] - tile1.y + over[3]
    subimg1 = tile1.get_image()[t1y1:t1y2, t1x1:t1x2]
    t2x1 = over[0] - tile2.x
    t2x2 = over[0] - tile2.x + over[2]
    t2y1 = over[1] - tile2.y
    t2y2 = over[1] - tile2.y + over[3]
    subimg2 = tile2.get_image()[t2y1:t2y2, t2x1:t2x2]

    return (subimg1, subimg2)