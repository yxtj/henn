# -*- coding: utf-8 -*-

import numpy as np
import time
#import multiprocessing as mp

#import hennlayer

from .data import load_data_part
from .phenetwork import PhenLayer, PhenConv, PhenLinear, PhenFlatten, PhenRelu, PhenSquare
from network import Network

from .shaper import Shaper, make_shaper

class Worker:
    def __init__(self, hid, wid, nh, nw):
        self.hid = hid # height id
        self.wid = wid # width id
        self.nh = nh # number of height separations
        self.nw = nw # number of width separations
        self.sid = hid*nw + wid # sequence id of this worker
        # network
        self.net = None
        self.worker_list = {} # map (hid, wid) to worker id
        # neural network
        self.model = None
        self.datasize = []

    # interface functions

    def init_network(self):
        self.net = Network()
        r = self.net.alltoall((self.hid, self.wid))
        for i, c in enumerate(r):
            self.worker_list[c] = i
        print(f"Worker {self.net.rank} registered in network.")

    def init_model(self, model:list[PhenLayer], inshape:tuple):
        self.model = model
        #self.planner = Planner(self.nh, self.nw, model, inshape)
        self.gshapes = []
        self.shapers = []
        self.ltypes = []
        s = inshape
        self.gshapes.append(s)
        self.shapers.append(make_shaper(self.nh, self.nw, len(s), s))
        for idx, layer in enumerate(self.model):
            s = layer.out_shape(s)
            ss = make_shaper(self.nh, self.nw, layer.dim, s)
            self.gshapes.append(s)
            self.shapers.append(ss)
            self.ltypes.append(layer.ltype)
        for idx in range(len(self.model)):
            layer.bind_in_model(self.shapers[idx], self.shapers[idx+1],
                                idx, self.gshapes[idx], self.ltypes[idx])

    def run(self, data:np.ndarray):
        x = load_data_part(data)
        for layer in self.model:
            if isinstance(layer, PhenConv):
                x = self.comp_conv(x, layer)
            elif isinstance(layer, PhenConv):
                x = self.comp_linear(x, layer)
            elif isinstance(layer, PhenFlatten):
                x = self.comp_flatten(x)
            elif isinstance(layer, PhenRelu) or isinstance(layer, PhenSquare):
                x = self.comp_act(x)
            else:
                print(f"{layer.layer_idx}-th layer of type {type(layer)}"
                      " is not supported")
        return x

    # inner functions - compute

    def comp_conv(self, x, conv:PhenConv):
        x = self.preprocess(conv, x)
        x = conv.forward(x)
        x = self.postprocess(conv, x)
        return x

    def comp_linear(self, x, fc:PhenLinear):
        x = self.preprocess(fc, x)
        x = fc.forward(x)
        x = self.postprocess(fc, x)
        return x

    def comp_flatten(self, x, fn:PhenFlatten):
        x = fn.forward(x)
        return x

    def comp_act(self, x, act:PhenRelu):
        x = act.forward(x)
        return x

    # inner functions - data

    def preprocess(self, layer:PhenLayer, x:np.ndarray):
        # prepare data depdency
        rqtgts = layer.depend_out(x)
        for hid, wid, desc in rqtgts:
            msg = layer.depend_message(x, hid, wid, desc)
            self.send(hid, wid, msg)
        xlist = []
        n = len(layer.depend_in())
        for i in range(n):
            d = self.recv()
            xlist.append(d)
        x = layer.depend_merge(x, xlist)
        # padding
        x = layer.local_prepare(x)
        return x

    def postprocess(self, layer:PhenLayer, x:np.ndarray):
        # load balance
        tgts = layer.join_out()
        for hid, wid, desc in tgts:
            msg = layer.join_message(x, hid, wid, desc)
            self.send(hid, wid, msg)
        n = len(layer.join_in())
        #if n == 0:
        xlist = []
        for i in range(n):
            d = self. recv()
            xlist.append(d)
        x = layer.join_merge(x, xlist)
        return x

    # low level functions

    def send(self, hid, wid, data):
        worker_id = hid*self.nw + wid
        self.net.isend(data, worker_id)

    def recv(self):
        return self.net.recv()





