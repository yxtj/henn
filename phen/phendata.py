# -*- coding: utf-8 -*-
import numpy as np
from .phennetwork import PhenNetwork

class PhenData():
    def __init__(self, nh, nw, hid, wid):
        self.nh = nh
        self.nw = nw
        self.hid = hid
        self.wid = wid
        
    def compute_off(self, net:PhenNetwork):
        return 0
        pass
    
    
    def divide_data(self, data:np.ndarray, off=2):
        '''
        Requirement: data to be 2D
        '''
        w, h = data.shape # data.shape[-2:]
        hind = np.linspace(0, h, self.nh, False, dtype=int)
        wind = np.linspace(0, w, self.nw, False, dtype=int)
        dpart = np.array([ 
            [None for i in range(self.wnpart)] for j in range(self.hnpart)])
        for i in range(self.hnpart - 1):
            h1 = max(0, hind[i]-off)
            h2 = min(h, hind[i+1]+off)
            for j in range(self.wnpart - 1):
                w1 = max(0, wind[j]-off)
                w2 = min(h, wind[j+1]+off)
                dpart[i,j] = data[h1:h2,w1:w2]
        return dpart