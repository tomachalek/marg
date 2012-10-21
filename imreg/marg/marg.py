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
This module contains core Swarming Tiles algorithm implementation.
"""

import numpy as np
import random
import math
import mahotas
from collections import namedtuple
from scipy.cluster import hierarchy
from scipy import stats
from scipy.stats import entropy
from scipy.stats import histogram

from imreg import image

def generate_segment(image, position, radius):
    """
    Creates a sub-image of provided image on a position taken as a centre of the 'segment'.
    If the position and radius cause the segment to be (partially) outside of the source image
    then the size is reduced accordingly.

    Parameters
    ----------
    image : numpy.ndarray
        input image
    position : 2-element tuple of int
        centre of the segment in image's coordinate system
    radius : int
        width and height of the segment will be (2 * radius + 1)

    Returns
    -------
    ImgSegment instance with provided parameters
    """
    x, y = position
    x1 = max(0, x - radius)
    x2 = min(np.size(image, 0) - 1, x + radius)
    y1 = max(0, y - radius)
    y2 = min(np.size(image, 1) - 1, y + radius)
    return ImgSegment(image[x1:x2+1, y1:y2+1], (x, y))

def calculate_domain(image, radius):
    """
    Calculates ranges for X and Y axes with respect to a segment size (= radius).
    This means that we can always select any value from this range, make a segment
    with center at this value and with specified radius without worrying about
    segment being (partially) out of fixed image.

    Parameters
    ----------
    image : numpy.array
        image to be considered a source of segments
    radius : int
        This is not in fact the radius because we talk about rectangles here. It means
        number of pixels to the edge from the central pixel (width and height are
        always odd here) when the azimuth of the 'walk' is multiple of Pi/4

    Returns
    tuple containing tuples of floats: ((x_min, y_min), (x_max, y_max))
    -------
    """
    return ((radius, radius), (np.size(image, 0) - radius - 1,
                               np.size(image, 1) - radius - 1))

def generate_random_segments(image, number, radius, entropy_hint=False, sample_size=1.5):
    """
    Generates selected number of random segments with selected radius. The distribution
    of number values is uniform.

    Parameters
    ----------
    image : numpy.ndarray
        2-dimensional array representing a grayscale image
    number : int
        number of segments
    radius : int
        "radius" of the segment (= number of pixels to the edge when walking from the central pixel
        with azimuth i*Pi/4)

    Returns
    -------

    """
    segments = []
    score_map = {}
    num_samples = number if entropy_hint is False else int(number * sample_size)
    domain = calculate_domain(image, radius)
    for i in range(num_samples):
        x = random.randint(domain[0][0], domain[1][0])
        y = random.randint(domain[0][1], domain[1][1])
        segment = generate_segment(image, (x, y), radius)
        if entropy_hint is True:
            ent = entropy(histogram(segment.image, numbins=256, defaultlimits=(0, 255))[0])
            score_map[ent] = segment
        else:
            segments.append(segment)

    if entropy_hint is True:
        values = score_map.keys()
        values.sort()
        for v in values[:number]:
            segments.append(score_map[v])

    return segments



def calc_best_result(coords, threshold=0.01):
    """
    Calculates most possible result based on clustering of provided coordinates.
    We assume that the bigger cluster represents the value most of the best agent
    have agreed on. Method uses SciPy's hierarchy.fclusterdata function.

    Parameters
    ----------
    coords : list of two-element tuples
        coordinates to guess result from
    threshold : float
        see documentation for scipy.hierarchy.fclusterdata

    Returns
    -------
    x : float
        x coordinate of the result
    y : float
        y coordinate of the result

    """
    coords = np.array(coords)
    t = coords[:,0].std()
    idx = hierarchy.fclusterdata(coords, threshold * t)
    best = int(stats.mode(idx)[0][0])
    ans = np.array([coords[i] for i in range(len(coords)) if idx[i] == best])
    return namedtuple('Ans', 'x, y')(ans[:,0].mean(), ans[:,1].mean())


def add_border_to_image(image, w, h):
    """
    In development
    TODO
    """
    new_img = np.zeros((np.size(image, 0) + 2 * h, np.size(image, 1) + 2 * h))
    for i in range(0, h):
        pass
    for line in image:
        pass
    print("new image: %s x %s" % (np.size(new_img, 1), np.size(new_img, 0)))


def create_agent_pool(image1, image2, num_agents, num_incom_segments, segment_radius,
                      cost_func, entropy_hint, sample_size):
    """
    Factory function which loads input images and creates an AgentPool with requested parameters

    Parameters
    ----------
    image1 : str
        path to the first image
    image2 : str
        path to the second image
    num_agents : int
        number of agents to be created
    num_incom_segments : int
        number of segments agent borrows from the repository each iteration
    segment_radius : int
        "radius" of segments
    cost_func : any function able to evaluate two 2-dimensional arrays of the same size as a float
        error function representing similarity of segment and part of the image1 it covers
    """
    images = (mahotas.imread(image1, as_grey=True), mahotas.imread(image2, as_grey=True))
    return AgentPool(images, num_agents, num_incom_segments, segment_radius, cost_func,
            entropy_hint, sample_size)


class ImgSegment(object):
    """
    This class represents image segment, i.e. the image (with his width and height) plus
    a position in a 2D space. Image is expected to be a numpy array.
    """

    def __init__(self, image, position):
        """
        Parameters
        ----------
        image : numpy.array
            greyscale image data (i.e. 2D array, not 3D like in case of RGB image)
        position : tuple containing two real numbers
            initial position of the segment
        """
        if type(image) == list:
            image = np.array(image)
        elif type(image) <> np.ndarray:
            raise TypeError('Segment accepts only numpy.ndarray data')
        self.image = image
        self.x, self.y = position
        self.width = np.size(self.image, 1)
        self.height = np.size(self.image, 0)

    def __repr__(self):
        """
        String representation of the segment displays its hash, x, y coordinates and
        width and height
        """
        return "Segment #%s on (%01.2f, %01.2f), width: %01.2f, height: %01.2f" %\
               (self.__hash__(), self.y, self.x, self.width, self.height)

    def get_image(self):
        """
        This is defined to make the class compatible with image's ImageSegment
        """
        return self.image


class Agent(object):
    """
    Agent represents individual search for best cost value with some segment of
    image 2. Each iteration he borrows some additional segments from the segment
    repository and along with his own segment he tries to reach better solution.
    """
    
    def __init__(self, agent_pool, segment, num_incom_segments, position=None, id=None):
        """
        Parameters
        ----------

        agent_pool : AgentPool object

        segment : ImgSegment
            initial image segment

        num_incom_segments :
        """
        self.agent_pool = agent_pool
        self.segment = segment
        self.num_incom_segments = num_incom_segments
        self.current_value = None
        self.x, self.y = position if position is not None else (0, 0)
        self.id = id
        self.iteration = 0
        
    def __repr__(self):
        return "Agent %s (%01.2f, %01.2f), value: %s" % (self.id, self.y, self.x, "%01.4f" % self.current_value
                                                            if self.current_value is not None else "?")

    def generate_new_positions(self, segment):
        """
        """
        ans = []
        for i in range(self.num_incom_segments):
            # prev was 200
            v1 = int(np.size(self.agent_pool.images[1], 1) / 2.0 / (1 + self.iteration))
            v2 = int(np.size(self.agent_pool.images[1], 0) / 2.0 / (1 + self.iteration))
            new_x = self.x + random.randint(-v1, v1)
            new_y = self.y + random.randint(-v2, v2)

            if new_x < self.agent_pool.domain[0][0]:
                new_x = self.agent_pool.domain[1][0] - (self.agent_pool.domain[0][0] - new_x)
            elif new_x > self.agent_pool.domain[1][0]:
                new_x = self.agent_pool.domain[0][0] + new_x - self.agent_pool.domain[1][0]
                

            if new_y < self.agent_pool.domain[0][1]:
                new_y = self.agent_pool.domain[1][1] - (self.agent_pool.domain[0][1] - new_y)
            elif new_y > self.agent_pool.domain[1][1]:
                new_y = self.agent_pool.domain[0][1] + new_y - self.agent_pool.domain[1][1]

            ans.append((new_x, new_y))
        return ans

    def iterate(self):
        score_map = {}
        out_segments = []
        test_segments = self.agent_pool.fetch_random_stock_segments(self.num_incom_segments) + [self.segment]

        for segment in test_segments:
            for test_pos in self.generate_new_positions(segment):
                mi = self.agent_pool.cost_function(test_pos, segment)
                if not mi in score_map:
                    score_map[mi] = [(segment, test_pos)]
                else:
                    score_map[mi].append((segment, test_pos))
        scores = score_map.keys()
        scores.sort()

        best = score_map[scores[0]][0]
        self.current_value = scores[0]
        self.x, self.y = score_map[scores[0]][0][1]
        self.segment = score_map[scores[0]][0][0]

        for i in range(1, len(scores)):
            for item in score_map[scores[i]]:
                self.agent_pool.segments_stock.append(item[0])

        self.iteration += 1


    def get_image_position(self):
        return (self.y - self.segment.y, self.x - self.segment.x)

    def get_status(self):
        return "Result: %s, %s" % (self.y - self.segment.y, self.x - self.segment.x)

class AgentPool(object):
    """
    """

    def __init__(self, images, num_agents, num_incom_segments, segment_radius, cost_func,
            entropy_hint, sample_size):
        """
        Parameters
        ---------
        images : 2-item tuple
            processed images - 1-st one is considered to be the 'fixed' one, 2-nd is the transformed one
        num_agents : int
            number of agents to be created
        num_incom_segments : int
            number of segments agent borrows from the repository each iteration
        segment_radius : int
            "radius" of segments
        cost_func : any function able to evaluate two 2-dimensional arrays of the same size as a float
            error function representing similarity of segment and part of the image1 it covers
        entropy_hint :
        """
        self.images = images
        self.segment_radius = segment_radius
        self.cost_func = cost_func
        self.domain = calculate_domain(images[0], segment_radius)
        #add_border_to_image(images[0], 10, 10) # experimental stuff
        self.entropy_hint = entropy_hint
        self.sample_size = sample_size
        self.segments_stock = generate_random_segments(self.images[1],
            num_agents * num_incom_segments, segment_radius, self.entropy_hint, self.sample_size)
        self.agents = self._create_agents(num_agents, num_incom_segments, segment_radius)


        
    def cost_function(self, position, segment1):
        """
        This method wraps used cost function. Before the cost function itself is called
        it prepares the data needed to make the call.

        Parameters
        ----------
        position : 2-float tuple
            position to be applied to provided segment
        segment1 : ImgSegment
            segment to be examined
        """
        segment2 = generate_segment(self.images[0], position, self.segment_radius)
        return self.cost_func(segment1.image, segment2.image)
        
    def _create_agents(self, num_agents, num_incom_segments, segment_radius):
        """
        Generates group of agents with random positions and random segments attached

        Parameters
        ----------
        num_agents : int
            number of agents to be created
        num_incom_segments
        """
        ans = []
        for i in range(num_agents):
            x = random.randint(self.domain[0][0], self.domain[1][0])
            y = random.randint(self.domain[0][1], self.domain[1][1])
            si = random.randint(0, len(self.segments_stock) - 1)
            a = Agent(self, segment=self.segments_stock[si],
                      num_incom_segments=num_incom_segments, position=(x, y), id=i)
            ans.append(a)
            del(self.segments_stock[si])
        return ans

    def fetch_random_stock_segments(self, num_items):
        ans = []
        if num_items == len(self.segments_stock):
            ans += self.segments_stock
            self.segments_stock = []
        else:
            for i in range(num_items):
                j = random.randint(0, len(self.segments_stock) - 1)
                ans.append(self.segments_stock[j])
                del(self.segments_stock[j])
        return ans
    
    def export_all_segments(self):
        """
        TODO
        """
        for a in self.agents:
            image.save_greyscale("d:/work/swarmspace/minisegment-%d.png" % a.id, a.best_segment.image)

    def evaluate(self, agent_percentile=90):
        """
        Evaluates current state of the system.

        Returns
        -------
        best_coords : list of 2-float tuples
            best coordinates
        """
        best_agents = self.get_best_agents(agent_percentile)
        print("Best coordinates: %s" % (best_agents))
        best_coords = calc_best_result(best_agents)
        return namedtuple('Ans', 'best_coords, best_agents')(best_coords, best_agents)

    def get_best_agents(self, percentile):
        """
        Returns best agents according to their current cost function value (where less is better).

        Parameters
        ----------
        percentile : int
            portion of best agents to use (e.g. the 90th percentile selects best 10% of agents)
        """
        scores = {}
        for a in self.agents:
            scores[a.current_value] = a
        vals = scores.keys()
        vals.sort()
        ans = []
        for v in vals[:(int(math.ceil((1 - percentile / 100.0) * len(self.agents))))]:
            ans.append(scores[v].get_image_position())
        return ans

    def iterate(self):
        """
        Generates single iteration for all included agents.
        """
        for agent in self.agents:
            agent.iterate()
