# -*- coding: utf-8 -*-
import numpy as np
import multiprocessing as mp
#from multiprocessing import Process, Queue, Pool
import os, time

from .phendata import PhenData
from .phennetwork import PhenNetwork

class PhenRunnerDup():
    '''
    Parallel Homomorphic Encrypted Deep Neural Network.
    Duplicate the data on cutting edges
    '''
    def __init__(self, n_part, network):
        self.npart = n_part
        self.hnpart, self.wnpart = self.divide_part(n_part)
        assert self.wnpart * self.hnpart == self.npart
        self.net = network
        self.divide_network()
        self.parts = np.array([
            [None for i in range(self.wnpart)] for j in range(self.hnpart)])
    
    def divide_part(self, n):
        s = np.sqrt(n)
        h = int(np.floor(s))
        w = n // h
        return h, w
    
    def divide_network(self):
        pass
    
    def divide_data(self, data:np.ndarray, off=2):
        '''
        Require data to be 2D
        '''
        h, w = data.shape # data.shape[-2:]
        hind = np.linspace(0, h, self.hnpart, False, dtype=int)
        wind = np.linspace(0, w, self.wnpart, False, dtype=int)
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
    
    def forward(self, x:np.ndarray):
        self.mg = mp.Manager()
        self.dict = self.mg.dict()
        in_ch, sx, sy = x.shape
        ox, oy = self.comp_out_size(sx, sy)
        #out = np.empty((self.out_ch, ox, oy), x.dtype)
        
        pool = mp.Pool(self.parallel)
        out = np.array(pool.map(
            lambda ch:self.compute(x, ch, ox, oy), range(self.conv.out_ch)),
            x.dtype)
        pool.join()


# %% 

class PhenRunnerCom():
    '''
    Parallel Homomorphic Encrypted Deep Neural Network.
    Workers communicate to share pixels on the cutting edge.
    '''
    def __init__(self, n_part, network):
        self.npart = n_part
        self.hnpart, self.wnpart = self.divide_part(n_part)
        assert self.wnpart * self.hnpart == self.npart
        self.net = network
        self.divide_network()
        self.parts = np.array([
            [None for i in range(self.wnpart)] for j in range(self.hnpart)])
    
    
# %% main

def main():
    pass

if __name__ == "__mani__":
    main()
    