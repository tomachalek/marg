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

import csv
import re

class PerIterDataCollector(object):
    """
    """
    
    def __init__(self):
        self.best_values = []
        
    
    def iter_add_best_values(self, values):
        self.best_values.append(values)
        
        
class ParticleDataCollector(object):
    """
    """
    
    def __init__(self):
        self.vals = []
        self.best_vals = []
        self.coords = []
        

class TotalsDistCollector(object):
    """
    """
    
    def __init__(self):
        self._sections = {}
        
    def add_distance(self, section, dist):
        if not section in self._sections:
            self._sections[section] = []
        self._sections[section].append(dist)
 
    def auto_order_sections(self):
        key_dict = {}
        ordered_keys = []
        other_keys = []
        ans = []
        for key in self._sections:
            x = re.search('^.+\s+([0-9]+)$', key)
            if x is not None:
                idx = int(x.group(1))
                ordered_keys.append(idx)
                key_dict[idx] = key
            else:
                other_keys.append(key)
        ordered_keys.sort()
        for i in ordered_keys:
            ans.append(key_dict[i])
        for key in other_keys:
            ans.append(key)
        return ans
            
        
    def export(self, path):
        data_writer = csv.writer(open(path, 'wb'), delimiter=';',
                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        keys = self.auto_order_sections()
        print(keys)
        num_tests = len(self._sections[keys[0]])
        for i in range(0, num_tests):
            row = []
            for key in keys:
                row.append(self._sections[key][i])
            data_writer.writerow(row)
            

class CoordinatesAggregator(object):
    """
    """
    
    def __init__(self):
        self.pcollectors = []
        
        
    def get_particle_coords(self, i, coord_i):
        ans = []
        for item in self.pcollectors[i]:
            ans.append(item[coord_i])
            
            
    def get_average_best_result(self):
        ans = []
        i1 = 0
        i2 = len(self.pcollectors[0].coords)
        for i in range(i1, i2):
            a = 0.0
            for j in range(len(self.pcollectors)):
                a += self.pcollectors[j].best_vals[i]
            ans.append(a / len(self.pcollectors))
        return ans
    
    def get_average_result(self):
        ans = []
        i1 = 0
        i2 = len(self.pcollectors[0].coords)
        for i in range(i1, i2):
            a = 0.0
            for j in range(len(self.pcollectors)):
                a += self.pcollectors[j].vals[i]
            ans.append(a / len(self.pcollectors))
        return ans
    
    
    def get_z_histogram(self):
        ans = []
        for c in self.pcollectors:
            for item in c.coords:
                ans.append(item[2])
        return ans
    
    def get_z_average(self):
        ans = []
        i1 = 0
        i2 = len(self.pcollectors[0].coords)
        for i in range(i1, i2):
            a = 0.0
            for j in range(len(self.pcollectors)):
                a += self.pcollectors[j].coords[i][2]
            ans.append(a / len(self.pcollectors))
        return ans
            

class MultiTestCollector(object):
    """
    """
    
    def __init__(self):
        self._tests = []
        self._labels = {}
        
    def add_test_result(self, test, label = ''):
        self._tests.append(test)
        self._labels[test] = label 
        
    def export_to_csv(self, file_path, func):
        results = []
        for test in self._tests:
            f = getattr(test, func)
            results.append(f())
            
        data_writer = csv.writer(open(file_path, 'wb'), delimiter=';',
                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        labels = []
        print(self._labels)
        for t in self._tests:
            labels.append(self._labels[t])
        data_writer.writerow(labels)
        for i in range(len(results[0])):
            row = []
            for j in range(len(results)):
                row.append(results[j][i])
            data_writer.writerow(row)
