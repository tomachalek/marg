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
This module generates experiments according to provided ini files.
Its purpose is to demonstrate prototype of our multi-agent image registration
algorithm and compare obtained results with some other methods.
"""

import sys
import os
import ConfigParser
import re
import datetime
import math
import time

from imreg.reporting import collector
from imreg import image
from imreg import similarity
from imreg.optimization import pso
from imreg.optimization import topology

def save_totals(config_file, dist_collector):
    """
    Saves results of the test to the CSV file stored inside a directory which name
    derived from current date and time.

    Parameters
    ----------
    config_file : str
        path to a file with an experiment configuration
    dist_collector : TotalsDistCollector
        object collecting data generated during experiments
    """
    out_dir = "output/%s" % datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    out_file = re.sub(r'^.+\%s(.+)\.ini$' % os.sep, r'\1-totals.csv', config_file)
    out_path = "%s/%s" % (out_dir, out_file)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    dist_collector.export(out_path)

def print_results(aggregator, prefix = None):
    """
    """
    # TODO fix this function
    if prefix is not None:
        prefix = "%s-" % prefix

    bvals = aggregator.get_average_best_result()
    plt.grid(True)
    plt.plot(bvals)
    plt.savefig('%sbest_vals.png' % prefix)

    plt.rcdefaults()
    plt.close()

    vals = aggregator.get_average_result()
    plt.grid(True)
    plt.plot(vals)
    plt.savefig('%scurr_vals.png' % prefix)

    plt.rcdefaults()
    plt.close()

    z_vals = aggregator.get_z_histogram()
    plt.grid(True)
    plt.hist(z_vals)
    plt.savefig('%sz_histogram.png' % prefix)

    plt.rcdefaults()
    plt.close()

    z_avgs = aggregator.get_z_average()
    plt.grid(True)
    plt.plot(z_avgs)
    plt.savefig('%sz_avgs.png' % prefix)


def pso_optimize(config, test, cost_func, multitest_collector, dist_collector):
    """
    """
    params = {
        'c1' : 0.8,
        'c_max' : 1.62,
        'domain' : ((min_x, max_x), (min_y, max_y), (scale_from, scale_depth))
    }
    optimizer = pso.PSO(params, particle_collector_type=collector.ParticleDataCollector)
    optimizer.create_swarm(num_pso_particles, cost_func, topology.create_ring_topology, init_coords=None)

    for i in range(0, num_iter):
        sys.stdout.write(".")
        optimizer.iterate()
    print(  )
    p_best = optimizer.get_best_particle()

    best_dist = p_best.get_2d_distance((optim_x, optim_y))
    dist_collector.add_distance(test, best_dist)
    print(p_best)
    print("dist: %01.2f" % best_dist)

    ca = collector.CoordinatesAggregator()
    for p in optimizer.particles:
        ca.pcollectors.append(p.collector)
    multitest_collector.add_test_result(ca, test)


def sa_optimize(domain, cost_func):
    """
    TODO
    """
    sa_optimizer = SimulatedAnnealing()
    sa_optimizer.optimize(domain, cost_func, T=10000, cool=0.95, step=1)


config = ConfigParser.RawConfigParser()
conf_path = sys.argv[1]
if os.path.isfile(conf_path):
    config.read(sys.argv[1])
else:
    print('ERROR: Experiment configuration [%s] not found.' % conf_path)
    sys.exit(1)


multitest_collector = collector.MultiTestCollector()
dist_collector = collector.TotalsDistCollector()
time_results = []
for test in config.sections():
    start_time = time.time()
    total_runs = 0
    if config.has_option(test, 'disabled') and config.getboolean(test, 'disabled') is True:
        print("skipped test [%s]" % test)
        continue
    print("processing test [%s]" % test)

    if config.has_option(test, 'type'):
        test_type = config.get(test, 'type')
    else:
        raise Exception('Test type must be specified')

    num_iter = config.getint(test, 'num_iter')

    # optimal registration coordinates
    optim_x = config.getint(test, 'optim_x') if config.has_option(test, 'optim_x') else 0
    optim_y = config.getint(test, 'optim_y') if config.has_option(test, 'optim_y') else 0

    # search space (not all algorithms need this)
    min_x = config.getint(test, 'min_x') if config.has_option(test, 'min_x') else -100
    max_x = config.getint(test, 'max_x') if config.has_option(test, 'max_x') else 100
    min_y = config.getint(test, 'min_y') if config.has_option(test, 'min_y') else -100
    max_y = config.getint(test, 'max_y') if config.has_option(test, 'max_y') else 100

    # image files
    image1 = config.get(test, 'image1')
    image2 = config.get(test, 'image2')

    # cost function (not all algorithms need this)
    cost_func_name = config.get(test, 'cost_func')
    cf = getattr(globals()['similarity'], cost_func_name)

    if test_type == 'REGISTRATION_WITH_PSO':
        num_pso_particles = config.getint(test, 'num_particles')
        # loading of registered images
        scale_depth = config.getint(test, 'scale_depth')
        scale_from = config.getint(test, 'scale_from') if config.has_option(test, 'scale_from') else 0
        sigma_step = config.getfloat(test, 'sigma_step') if config.has_option(test, 'sigma_step') else 1.0

        tile1 = image.ImageTile(image1, scale_depth=scale_depth, scale_from=scale_from, step_coeff=sigma_step)
        tile2 = image.ImageTile(image2, scale_depth=scale_depth, scale_from=scale_from, step_coeff=sigma_step)

        cost_function = similarity.CostFunction(cf, tile1, tile2)
        print("COST: %s" % cost_function)

        #domain = ((min_x, min_y), (max_x, max_y))

        for test_iter in range(0, config.getint(test, 'repeat')):
            pso_optimize(config, test, cost_function, multitest_collector, dist_collector)
            total_runs += 1

    elif test_type == 'REGISTRATION_WITH_SWARMING_TILES':
        from imreg.marg import marg

        num_incom_segments = config.getint(test, 'st.num_incom_segments')
        segment_radius = config.getint(test, 'st.segment_radius')
        num_agents = config.getint(test, 'st.num_agents')
        entropy_hint = config.getboolean(test, 'st.entropy_hint')
        sample_size = config.getfloat(test, 'st.entropy_hint_sample_size')

        print("TEST [")
        print("\ttype = st,\n\tnum_iter = %d,\n\tnum_agents = %d," % (num_iter, num_agents))
        print("\tnum_incom_segments = %d,\n\tsegment_radius = %d," % (num_incom_segments, segment_radius))
        print("\tentropy_hint = %s,\n\tsample_size = %s\n]" % (entropy_hint, sample_size))

        for test_iter in range(0, config.getint(test, 'repeat')):
            agent_pool = marg.create_agent_pool(image1, image2, num_agents=50,
                    num_incom_segments=num_incom_segments, segment_radius=segment_radius,
                    cost_func=cf, entropy_hint=entropy_hint,sample_size=sample_size)
            for i in range(num_iter):
                agent_pool.iterate()
            best_coords, best_agents = agent_pool.evaluate()
            print(best_coords)
            dist = math.sqrt((best_coords[0] - optim_x) * (best_coords[0] - optim_x)
                    + (best_coords[1] - optim_y) * (best_coords[1] - optim_y    ))

            dist_collector.add_distance(test, dist)
            total_runs += 1


    total_time = time.time() - start_time
    avg_time = total_time / float(total_runs) if total_runs > 0 else 0

    print("-----------------------------------------------------------")
    time_results.append("Total time: %01.2f sec., average time: %01.2f" % (total_time, avg_time))

save_totals(conf_path, dist_collector)
for s in time_results:
    print(s)