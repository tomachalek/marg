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
This is currently an experimental module...
"""

import matplotlib.pyplot as plt
from scipy.stats import entropy
from scipy.stats import histogram
import numpy as np
import math
import random

def stringify_num(v):
    """
    Returns string representation of a number with 6-digit precision.
    It's used here mostly to generate keys to dictionaries.
    """
    return "%01.6f" % round(v, 6)

class EntropyTiles(object):
    """
    """
    def __init__(self, image):
        """
        """
        self.image = image
        
    def calc_tile_weights(self, num_tiles = (1, 1)):
        """
        """
        img_w = np.size(self.image, 0)
        img_h = np.size(self.image, 1)
        w = img_w / num_tiles[0]
        h = img_h / num_tiles[1]
        print("w = %d, h = %d" % (w, h))
        tile_map = {}
        i = 0
        while i < img_w:
            if i + w > img_w:
                i += w
                continue
            j = 0
            while j < img_h:            
                if j + h > img_h:
                    j += h
                    continue
                ent = tile_entropy(self.image, (i, j), (w, h)) ** 2
                ent_s = "%01.6f" % round(ent, 6)
                if not ent_s in tile_map:
                    tile_map[ent_s] = [(i, j, w, h)]
                else:
                    tile_map[ent_s].append((i, j, w, h))
                #print("tile pos: %d x %d, entropy: %01.6f" % (i, j, ent))
                j += h
            i += w
        return tile_map
        
    
    def generate_attractors(self, matrix_size):
        weights = self.calc_tile_weights(self.image, matrix_size)
        coords = []
        ents_sq = []
        for k, v in weights.items():
            for frame in v:
                coords.append(create_random_coords(frame, 1))
                ents_sq.append(float(k) * float(k))
        
        max_ent_sq = max(ents_sq)
        for i in range(len(ents_sq)):
            ents_sq[i] = ents_sq[i] / max_ent_sq
        
        ans = []
        for i in range(len(ents_sq)):
            ans.append(Attractor(coords[i], ents_sq[i], 0.1))
        return ans


    def distribute_coords(self, weights, num_points):
        """
        """
        e_list = weights.keys()
        tmp = [float(i) for i in e_list]
        tmp.sort()
        
        distrib_map = {}
        distrib_f = []
        distrib = 0
        histogram = {}
        
        for v in tmp:
            distrib += v
            distrib_map[stringify_num(distrib)] = weights[stringify_num(v)]
            distrib_f.append(distrib)
            histogram[stringify_num(distrib)] = 0
        
        tmp = np.array(tmp)
        ans = []
        
        for i in range(0, num_points):
            r = random.random() * sum
            m = find_matching_level(distrib_f, r)
            histogram[stringify_num(m)] = histogram[stringify_num(m)] + 1
            ranges = distrib_map[stringify_num(m)]
            for r in ranges:
                ans.append(create_random_coords(r, 1))
        return ans


class EntropyMap(object):
    """
    """
    def __init__(self, image, radius):
        self.image = image
        self.map = np.zeros((np.size(self.image, 0), np.size(self.image, 1)))
        self.radius = radius
    
    def calc_point(self, x, y):
        x1 = max(0, x - self.radius)
        x2 = min(np.size(self.image, 0) - 1, x + self.radius)
        y1 = max(0, y - self.radius)
        y2 = min(np.size(self.image, 1) - 1, y + self.radius)
        sub = np.ravel(self.image[x1:x2+1, y1:y2+1])
        #print(histogram(sub, numbins=256, defaultlimits=(0, 255)))
        ent = entropy(histogram(sub, numbins=256, defaultlimits=(0, 255))[0])
        self.map[x1:x2+1,y1:y2+1] = (self.map[x1:x2+1,y1:y2+1] + ent) / 2
        return ent 

    def generate_randomized(self, num_hypo):
        for i in range (num_hypo):
            x = random.randint(0, np.size(self.image, 0))
            y = random.randint(0, np.size(self.image, 1))
            self.calc_point(x, y)
        return self.map
    
    def generate_full(self):
        for i in range(np.size(self.image, 0)):
            for j in range(np.size(self.image, 1)):
                ent = self.calc_point(i, j)
        save_greyscale("lena-entropy-x.png", self.map / np.max(self.map) * 255)

            

    def generate_entropy_map(self, path):
        img2 = np.zeros((np.size(self.image, 0), np.size(self.image, 1)))
        #f = np.vectorize(self.calc_point)
        #f()
        print(np.size(self.image, 0))
        for i in range(np.size(self.image, 0)):
            for j in range(np.size(self.image, 1)):
                img2[i][j] = self.calc_point(i, j)
        self.save_entropy_map(img2, path)


    def save_entropy_map(self, image, path):
        values = np.ravel(image)
        print(values)
        tmp = [(max(0, k - np.array(values).mean()) / np.array(values).std(ddof=1) * 20 + 128) for k in values]
        #print(tmp)
        save_greyscale(path, np.resize(tmp, (np.size(image, 0), np.size(image, 1))))


    
def tile_entropy(image, coords, size):
    tmp = []
    for i in range(coords[0], coords[0] + size[0]):
        for j in range(coords[1], coords[1] + size[1]):
            #print("(%d, %d" % (i, j))
            tmp.append(image[i][j])
    h = histogram(tmp, numbins=256, defaultlimits=(0, 255))[0]
    return entropy(h)








def calc_full_tile_weights(image):
    """
    """
    

    
def find_matching_level(distrib, v):
    i = 0
    while i < len(distrib) and v > distrib[i]:
        i += 1
    return distrib[i] if i < len(distrib) else None

def create_random_coords(frame, num_items):
    x1, y1 = [int(round(i)) for i in frame[:2]]
    w1, h1 = [int(round(i)) for i in frame[2:]]
    #print("generating in range [%d, %d] size [%d, %d]" % (x1, y1, w1, h1))
    x = random.randint(x1, x1 + w1)
    y = random.randint(y1, y1 + h1)
    return (x, y)

    
    


def plot_coords(coord_list):
    """
    plots coordinates ((x1,y1), (x2, y2),...)
    using the matplotlib
    """
    x_coords = []
    y_coords = []
    for coords in coord_list:
        x_coords.append(coords[0])
        y_coords.append(coords[1])
    plt.plot(x_coords, y_coords, '.')
    plt.show()


def generate_coords(image, size, num_particles):
    """
    This function provides kind of a shortcut... TODO
    """
    et = EntropyTiles(image)
    weights = et.calc_tile_weights(image, size)
    return et.distribute_coords(weights, num_particles)
