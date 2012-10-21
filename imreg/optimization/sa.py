'''
Created on 6.2.2012

Code originates from the book 'Programming Collective Intelligence' by Toby Segaran
'''

import random
import math

class SimulatedAnnealing(object):
    """
    """
    def optimize(self, domain, costf, T=100000.0, cool=0.95, step=1):
        # Initialize the values randomly
        vec = [float(random.randint(domain[i][0], domain[i][1]))
               for i in range(len(domain))]
        while T > 0.1:
            # Choose one of the indices
            i = random.randint(0, len(domain) - 1)
            # Choose a direction to change it
            delta = random.randint(-step, step)
            # Create a new list with one of the values changed
            vecb = vec[:]
            vecb[i] += delta
            if vecb[i] < domain[i][0]: 
                vecb[i] = domain[i][0]
            elif vecb[i] > domain[i][1]: 
                vecb[i] = domain[i][1]
            # Calculate the current cost and the new cost
            ea = costf((vec[0], vec[1], 0))
            eb = costf((vecb[0], vecb[1], 0))
            p = pow(math.e,(-eb-ea) / T)
            # Is it better, or does it make the probability
            # cutoff?
            if eb < ea or random.random() < p:
                vec = vecb
            # Decrease the temperature
            T = T * cool
        print(vec)
        return vec
