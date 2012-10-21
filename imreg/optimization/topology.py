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
PSO topology builders

TODO: implement more topologies
"""

def create_ring_topology(particles):
    """
    Connects particles into a ring topology.
    I.e. 0 <-> 1 <-> ... <-> n-1 <-> n <-> 0
    """
    for i in range(1, len(particles) - 1):
        particles[i].connections.append(particles[i-1])
        particles[i].connections.append(particles[i+1])
    particles[0].connections.append(particles[-1])
    particles[0].connections.append(particles[1])
    particles[-1].connections.append(particles[0])
    particles[-1].connections.append(particles[-2])


def create_wheel_topology(particles):
    """
    Connects particles into a wheel topology.
    I.e. 0 <-> 1, 0 <-> 2,..., 0 <-> n-1, 0 <-> n
    """
    for p in particles[1:]:
        p[0].connections.append(p)
        p.connections.append(p[0])
