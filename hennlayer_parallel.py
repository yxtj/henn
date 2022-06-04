# -*- coding: utf-8 -*-
import hennlayer

import multiprocessing as mp
import numpy as np


class parallel_conv():
    def __init__(self, conv:hennlayer.Conv2d, parallel:int):
        self.conv = conv
        self.parallel = parallel
        self.mg = mp.Manager()
        self.dict = self.mg.dict()
        self.dict['conv'] = self.conv
        
    def run(self, x):
        in_ch, sx, sy = x.shape
        ox, oy = self.comp_out_size(sx, sy)
        #out = np.empty((self.out_ch, ox, oy), x.dtype)
        
        pool = mp.Pool(self.parallel)
        out = np.array(pool.map(
            lambda ch:self.compute(x, ch, ox, oy), range(self.conv.out_ch)),
            x.dtype)
        pool.join()
        return out
    
    def compute(self, x, ch, ox, oy):
        o = np.empty((self.out_ch, ox, oy), x.dtype)
        conv = self.dict['conv']
        for i in range(ox):
            #print('  i',i)
            for j in range(oy):
                #print('    j',j)
                o[ch,i,j] = conv.conv(x, ch, i, j)
        return o
    