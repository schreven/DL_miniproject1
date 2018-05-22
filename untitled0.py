# -*- coding: utf-8 -*-
"""
Created on Tue May 15 16:48:14 2018

@author: Bob
"""

import numpy as np
import Visualizations as Vis

err_1_1 = np.load('.\images\err_1_1.npy')
Vis.plot_interstates(err_1_1)