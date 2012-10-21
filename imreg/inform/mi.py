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

'''
Mutual information calculations for grescale images

@author: Tomas Machalek <tomas.machalek@uhk.cz>
@todo: should be optimized using vectorized functions from np
'''

import math
from scipy.stats import histogram
import numpy as np
        
def mutual_information(data1, data2, domain=256, smoothing=0, log_base=2):
    """
    Mutual information for greyscale images
    
    Parameters
    ----------
    
    data1 : ndarray
        first data array
    
    data2 : ndarray
        second data array (size is expected to be the same as in case of data1)
    
    domain : int, optional (default is 256)
        value domain (e.g. all available grey values in case of image)
    
    smoothing : int
        value "k" in additive (aka Laplace) smoothing
    """
    
    m_hist = np.matrix(np.zeros((domain, domain)))
    img1_un = np.ravel(data1)
    img2_un = np.ravel(data2)
    
    # return entropy(histogram(np.ravel(image), numbins=256, defaultlimits=(0, 255))[0])
    
    d1_hist = histogram(img1_un, numbins=256, defaultlimits=(0, 255))[0]
    d2_hist = histogram(img2_un, numbins=256, defaultlimits=(0, 255))[0]
    for i in range(np.size(img1_un)):
        color1 = int(round(img1_un[i]))
        color2 = int(round(img2_un[i]))
        m_hist[color1, color2] += 1

    d1_hist = (d1_hist + smoothing) /  float(len(img1_un) + smoothing * domain)
    d2_hist = (d2_hist + smoothing) /  float(len(img2_un) + smoothing * domain)
    m_hist = (np.ravel(m_hist) + smoothing) /  float(len(img1_un) + smoothing * domain * domain)
    m_hist = np.resize(m_hist, (domain, domain))
    ans = 0
    for i in range(0, domain):
        for j in range(0, domain):
            if m_hist[i][j] == 0 or d1_hist[i] == 0 or d2_hist[j] == 0:
                continue
            else:
                tmp = m_hist[i][j] * math.log(m_hist[i][j] / d1_hist[i] / d2_hist[j], log_base)
                ans += tmp
    return ans
