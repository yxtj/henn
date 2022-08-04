# -*- coding: utf-8 -*-

import numpy as np
import time
import sys
#import multiprocessing as mp

#import hennlayer

from .phenetwork import PhenLayer, PhenConv, PhenLinear, PhenFlatten, PhenRelu, PhenSquare
from network import Network

from .shaper import make_shaper

class Worker:
    def __init__(self, hid, wid, nh, nw):
        self.hid = hid # height id
        self.wid = wid # width id
        self.nh = nh # number of height separations
        self.nw = nw # number of width separations
        self.sid = hid*nw + wid # sequence id of this worker
        # network
        self.net = None
        self.worker_list_m2s = {} # map matrix worker id (hid, wid) to serial id
        self.worker_list_s2m = {} # map serial worker id (hid, wid) to matrix id
        # neural network
        self.model = None
        self.gshapes = None
        self.shapers = None
        self.ltypes = None
        # statistic - network
        self.stat_send_msg = 0
        self.stat_send_byte = 0
        self.stat_recv_msg = 0
        self.stat_recv_byte = 0
        # statistic - computation
        self.stat_time_layer_prepare = []
        self.stat_time_layer_compute = []
        self.stat_time_layer_postprocess = []
        self.stat_time_layer = []
        self.stat_time_sync = 0.0
        self.stat_time_send = 0.0
        self.stat_time_recv = 0.0


    # interface functions

    def init_network(self):
        self.net = Network()
        r = self.net.alltoall((self.hid, self.wid))
        for i, c in enumerate(r):
            self.worker_list_m2s[c] = i
            self.worker_list_s2m[i] = c
        print(f"Worker {self.hid}-{self.wid} registered in network.")

    def init_model(self, model:list[PhenLayer], inshape:tuple):
        self.model = model
        gshapes, shapers, layer_types = self.compute_model_meta(model, inshape)
        self.gshapes = gshapes
        self.shapers = shapers
        self.ltypes = layer_types
        n = len(model)
        for lid in range(n):
            model[lid].bind_in_model(shapers[lid], shapers[lid+1],
                                     lid, gshapes, layer_types)
        self.stat_time_layer_prepare = [0.0 for _ in range(n)]
        self.stat_time_layer_compute = [0.0 for _ in range(n)]
        self.stat_time_layer_postprocess = [0.0 for _ in range(n)]
        self.stat_time_layer = [0.0 for _ in range(n)]
        print(f"Worker {self.hid}-{self.wid} load model.")

    def run(self, data:np.ndarray, complete_input=True):
        if complete_input == True:
            s = self.shapers[0]
            x = s.pick_data(self.hid, self.wid, data)
        else:
            x = data
        for lid, layer in enumerate(self.model):
            print(f"Worker {self.hid}-{self.wid} at Layer-{lid} {self.ltypes[lid]}")
            t = time.time()
            if isinstance(layer, PhenConv):
                x = self.comp_conv(x, lid, layer)
            elif isinstance(layer, PhenLinear):
                x = self.comp_linear(x, lid, layer)
            elif isinstance(layer, PhenFlatten):
                x = self.comp_flatten(x, lid, layer)
            elif isinstance(layer, PhenRelu) or isinstance(layer, PhenSquare):
                x = self.comp_act(x, lid, layer)
            else:
                print(f"{lid}-th layer of type {self.ltypes[lid]}"
                      " is not supported")
            t = time.time() - t
            self.stat_time_layer[lid] += t
        return x

    def show_stat(self):
        s = f"Statistics: worker {self.hid}-{self.wid} (w{self.sid}):\n"\
            f"  send {self.stat_send_msg} messages, {self.stat_send_byte} bytes; "\
            f"recv {self.stat_recv_msg} messages, {self.stat_recv_byte} bytes;\n"\
            f"  layer total time: {self.stat_time_layer};\n"\
            f"    prepare time: {self.stat_time_layer_prepare};\n"\
            f"    compute time: {self.stat_time_layer_compute};\n"\
            f"    postprocess time: {self.stat_time_layer_postprocess};\n"\
            f"  synchronization time: {self.stat_time_sync}, "\
            f"send time: {self.stat_time_send}, recv time: {self.stat_time_recv}"
        print(s)

    # inner functions - model prepare

    def compute_model_meta(self, model:list[PhenLayer], inshape:tuple):
        gshapes = [ inshape ]
        shapers = [ make_shaper(self.nh, self.nw, min(2, len(inshape)), inshape) ]
        layer_types = []
        s = inshape
        for lyr in model:
            #print(s, lyr)
            s = lyr.out_shape(s)
            gshapes.append(s)
            t = lyr.ltype
            layer_types.append(t)
            if t == "conv":
                ss = make_shaper(self.nh, self.nw, 2, s)
            elif t == 'linear':
                ss = make_shaper(self.nh, self.nw, 1, s)
            elif t == 'flatten':
                ss = make_shaper(self.nh, self.nw, 1, s)
            shapers.append(ss)
        return gshapes, shapers, layer_types

    # inner functions - compute

    def comp_conv(self, x, lid:int, conv:PhenConv):
        t0 = time.time()
        x = self.preprocess(conv, x)
        t1 = time.time()
        x = conv.local_forward(x)
        t2 = time.time()
        x = self.postprocess(conv, x)
        t3 = time.time()
        self.stat_time_layer_prepare[lid] = t1-t0
        self.stat_time_layer_compute[lid] = t2-t1
        self.stat_time_layer_postprocess[lid] = t3-t2
        return x

    def comp_linear(self, x, lid:int, fc:PhenLinear):
        t0 = time.time()
        x = self.preprocess(fc, x)
        t1 = time.time()
        x = fc.local_forward(x)
        t2 = time.time()
        x = self.postprocess(fc, x)
        t3 = time.time()
        self.stat_time_layer_prepare[lid] = t1-t0
        self.stat_time_layer_compute[lid] = t2-t1
        self.stat_time_layer_postprocess[lid] = t3-t2
        return x

    def comp_flatten(self, x, lid:int, fn:PhenFlatten):
        t = time.time()
        x = fn.local_forward(x)
        t = time.time() - t
        self.stat_time_layer_compute[lid] = t
        return x

    def comp_act(self, x, lid:int, act:PhenRelu):
        t = time.time()
        x = act.local_forward(x)
        t = time.time() - t
        self.stat_time_layer_compute[lid] = t
        return x

    # inner functions - data

    def preprocess(self, layer:PhenLayer, x:np.ndarray):
        # prepare data depdency
        rqtgts = layer.depend_out(x)
        for hid, wid, desc in rqtgts:
            msg = layer.depend_message(x, hid, wid, desc)
            self.send(hid, wid, msg)
        xlist = []
        dep = layer.depend_in(x)
        for i in range(len(dep)):
            hid, wid, msg = self.recv()
            xlist.append((hid, wid, msg))
        dep_req = [(h, w) for h, w, d in dep]
        dep_rec = [(h, w) for h, w, d in xlist]
        assert sorted(dep_req) == sorted(dep_rec), f"req:{dep_req}, recv:{dep_rec}"
        x = layer.depend_merge(x, xlist)
        return x

    def postprocess(self, layer:PhenLayer, x:np.ndarray):
        # load balance
        tgts = layer.join_out(x)
        for hid, wid, desc in tgts:
            msg = layer.join_message(x, hid, wid, desc)
            self.send(hid, wid, msg)
        req = layer.join_in(x)
        #if n == 0:
        xlist = []
        for i in range(len(req)):
            hid, wid, msg = self.recv()
            xlist.append((hid, wid, msg))
        x = layer.join_merge(x, xlist)
        return x

    # low level functions

    def map_worker_s2m(self, sid):
        return self.worker_list_s2m[sid]

    def map_worker_m2s(self, hid, wid):
        return self.worker_list_m2s[(hid, wid)]

    def send(self, hid, wid, data):
        t = time.time()
        worker_id = self.map_worker_m2s(hid, wid)
        self.stat_send_msg += 1
        self.stat_send_byte += sys.getsizeof(data)
        self.net.isend(data, worker_id)
        t = time.time() - t
        self.stat_time_send += t


    def recv(self):
        t = time.time()
        source, tag, data = self.net.recv()
        hid, wid = self.map_worker_s2m(source)
        self.stat_recv_msg += 1
        self.stat_recv_byte += sys.getsizeof(data)
        t = time.time() - t
        self.stat_time_recv += t
        return hid, wid, data

    def sync(self):
        t = time.time()
        self.net.barrier()
        t = time.time() - t
        self.stat_time_sync += t
