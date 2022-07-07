# -*- coding: utf-8 -*-
import numpy as np

class PhenDataHolder():
    def __init__(self, nh, nw, hid, wid):
        self.nh = nh
        self.nw = nw
        self.hid = hid
        self.wid = wid        
        self.pid = hid*nw + wid # sequence id
        self.part = None    
    
    def load_part(self, data:np.ndarray, off=0):
        '''
        Requirement: data to be 2D
        '''
        assert data.ndim == 2
        w, h = data.shape # data.shape[-2:]
        hind = np.linspace(0, h, self.nh+1, True, dtype=int)
        wind = np.linspace(0, w, self.nw+1, True, dtype=int)
        h1, h2 = hind[self.hid], min(hind[self.hid+1]+off, h)
        w1, w2 = wind[self.wid], min(wind[self.wid+1]+off, w)
        part = data[h1:h2, w1:w2]
        self.part = part
        

def load_data_part(data:np.ndarray, nh, nw, hid, wid, off=0):
    '''
    Requirement: data to be 2D
    '''
    assert data.ndim == 2
    w, h = data.shape # data.shape[-2:]
    hind = np.linspace(0, h, nh+1, True, dtype=int)
    wind = np.linspace(0, w, nw+1, True, dtype=int)
    h1, h2 = hind[hid], min(hind[hid+1]+off, h)
    w1, w2 = wind[wid], min(wind[wid+1]+off, w)
    part = data[h1:h2, w1:w2]
    return part


def divide_data(data:np.ndarray, nh, nw, off=0):
    '''
    Requirement: data to be 2D
    '''
    assert data.ndim == 2
    w, h = data.shape # data.shape[-2:]
    hind = np.linspace(0, h, nh+1, True, dtype=int)
    wind = np.linspace(0, w, nw+1, True, dtype=int)
    dpart = np.array([ [None for i in range(nw)] for j in range(nh)])
    for i in range(nh):
        #h1 = max(0, hind[i]-off)
        h1 = hind[i]
        h2 = min(h, hind[i+1]+off)
        for j in range(nw):
            #w1 = max(0, wind[j]-off)
            w1 = wind[j]
            w2 = min(h, wind[j+1]+off)
            dpart[i,j] = data[h1:h2,w1:w2]
    return dpart
