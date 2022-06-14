# -*- coding: utf-8 -*-

import numpy as np
import time
import multiprocessing as mp

#import computil
from computil import Conv2dConf, conv2d

#import hennlayer

from .data import PhenData
from .phenetwork import PhenNetwork, PhenConv, PhenLinear, PhenRelu, PhenSquare


class Worker:
    def __init__(self, hid, wid, nh, nw):
        self.hid = hid # height id
        self.wid = wid # width id
        self.nh = nh # number of height separations
        self.nw = nw # number of width separations
        self.sid = hid*nw + wid # sequence id of this worker
        # network related
        self.worker_list = {} # map (hid, wid) to worker id
        # data
        self.data = None
        # neural network
        self.model = None
    
    # interface functions
    
    def init_network(self):
        pass
    
    def load_data(self, data:np.ndarray):
        pass
    
    def load_model(self, net:PhenNetwork):
        pass
    
    def run(self):
        pass
    
    # inner functions - compute
    
    def comp_conv(self, data, conv:PhenConv):
        pass
    
    def comp_linear(self, data):
        pass
    
    def comp_act(self, data):
        pass
    
    # inner functions - data

    def wait_joint_data(self):
        pass
    
    def send_joint_data(self, off_h, off_w, data):
        assert off_h in (0, -1)
        assert off_w in (0, -1)
        assert off_h == -1 or off_w == -1
        tgt_hid = self.hid + off_h
        tgt_wid = self.wid + off_w
        
        pass
    
    def join_data(self):
        pass
        
    # low level functions
    
    def net_send(self, tgt_worker_id, data):
        pass
    
    def net_recv(self):
        pass
    
    
    
    
    