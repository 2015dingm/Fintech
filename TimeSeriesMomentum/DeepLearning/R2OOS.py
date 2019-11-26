#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 12:23:16 2018

@author: larryli
"""
import numpy as np

def R2OOS(r_real, r_hat, r_bar):
    denominator_res=(r_real-r_bar)**2
    denominator=np.sum(denominator_res)
    numerator_res=(r_real-r_hat)**2
    numerator=np.sum(numerator_res)
    r2oos=1-numerator/denominator
    return r2oos