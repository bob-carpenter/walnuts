#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 10:24:07 2025

@author: torekleppe
"""
import numpy as np

#--------------------------------------
# numerical constants
#--------------------------------------
__logZero = -700.0 # (used for communicating that a probability weight is zero)
__wtSumThresh = np.exp(__logZero+1.0)


