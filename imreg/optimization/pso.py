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
Simple implementation of the Particle Swarm Optimization algorithm
"""

import random
import sys
import math
import numpy as np


class Particle(object):
    """
    """
    
    def __init__(self, ident, params, cost_func, collector=None, num_dimensions=3):
        self.id = ident
        self.iteration = 0
        self.params = params
        self.cost_func = cost_func
        self.pos = [0.0 for i in range(num_dimensions)]
        self.v = [0.0 for i in range(num_dimensions)]
        self.best_value = float(sys.maxint)
        self.best_pos = [0.0 for i in range(num_dimensions)]
        self.connections = []
        self.collector = collector
        
    def find_best_connection(self):
        best_val = float(sys.maxint)
        best_p = None
        for p in self.connections:
            if p.best_value < best_val or best_p is None:
                best_val = p.best_value
                best_p = p
        return best_p        
    
    def __repr__(self):
        conn_ids = []
        for p in self.connections:
            conn_ids.append(p.id)
        return "particle(%d): %s; value: %f, cons: %s" % (self.id, str(self.pos), self.best_value, ", ".join([str(x) for x in conn_ids]))
    
    
    def iterate(self):
        curr_pos = [self.pos[i] for i in range(len(self.pos))]
        if len(self.pos) == 2:
            curr_pos.append(self.params['domain'][2][0])
        
        curr_pos = tuple(curr_pos)
        cost = self.cost_func(curr_pos)
        if cost < self.best_value:
            self.best_value = cost
            self.best_pos = curr_pos
        
        if self.collector is not None:
            self.collector.best_vals.append(self.best_value)
            self.collector.vals.append(cost)
            self.collector.coords.append(curr_pos)
            
        best_other = self.find_best_connection()

        for i in range(len(self.pos)):
            c1 = random.random() * self.params['c1']
            c2 = random.random() * self.params['c_max']
            c3 = random.random() * self.params['c_max']
            self.v[i] = c1 * self.v[i] + c2 * (self.best_pos[i] - self.pos[i]) \
                    + c3 * (best_other.pos[i] - self.pos[i])
            
            if self.pos[i] + self.v[i] > self.params['domain'][i][1]:
                self.pos[i] = self.params['domain'][i][1]
            elif self.pos[i] + self.v[i] < self.params['domain'][i][0]:
                self.pos[i] = self.params['domain'][i][0]
            else:
                self.pos[i] += self.v[i]
        self.iteration += 1
        
    def get_2d_distance(self, (x, y)):
        return math.sqrt((self.pos[0] - x) * (self.pos[0] - x) + (self.pos[1] - y) * (self.pos[1] - y))


class PSO(object):
    """
    Implementation of the Particle Swarm Optimization Algorithm for n-dimensional search space.
    """
    
    def __init__(self, params, particle_collector_type=None):
        """
        Parameters
        ----------
        params : dict
            configuration of the algorithm containing keys
              * constraints
              * c1
              * c_max
        """
        self.params = params
        # TODO following line violates declared n-dimensional support
        self.num_dimensions = 2 if params['domain'][2][0] == params['domain'][2][1] else 3
        self.particles = []
        self.iteration = 0
        self.particle_collector_type = particle_collector_type

    def create_swarm(self, num_particles, cost_func, topology_builder, init_coords = None):
        """
        Parameters
        ----------
        num_particles : int
            number of particles in the swarm
        cost_func : any function accepting tuple/list of values and returning single float/int value
            optimization cost function
        topology_builder : any function accepting list of particles
            function which interconnects particles in some way
        init_coords : list of n-value (float) tuples default None
            initial coordinates for particles

        """
        for i in range(num_particles):
            pc = self.particle_collector_type() if self.particle_collector_type is not None else None
            self.particles.append(Particle(i, self.params, cost_func, pc, num_dimensions=self.num_dimensions))

            if init_coords is not None:
                for i in range(min(len(init_coords), len(self.particles))):
                    for j in range(len(self.particles[i].pos)):
                        self.particles[i].pos[j] = init_coords[i][j]

            else:
                for p in self.particles:
                    for j in range(len(p.pos)):
                        p.pos[j] = random.random() * \
                                (self.params['domain'][j][1] - self.params['domain'][j][0]) \
                                + self.params['domain'][j][0]
        topology_builder(self.particles)

    def get_best_particle(self):
        """
        Returns particle with best (= lowest) cost function score

        Returns
        -------
        Particle instance
        """
        best_val = None
        best_part = None
        for p in self.particles:
            if best_val is None or p.best_value < best_val:
                best_part = p
                best_val = p.best_value
        return best_part

    def iterate(self):
        """
        Iterates all particles by one iteration
        """
        for p in self.particles:
            p.iterate()
        self.iteration += 1
