# -*- coding: utf-8 -*-

import numpy as np
import time
from network import Network
import parallel.phenetwork as pn
#from .Phenetwork import PhenLayer, PhenConv, PhenAvgPool, PhenLinear, PhenFlatten, PhenReLU, PhenSquare
from .shaper import make_shaper


__DEBUG__ = False
__INFLATION_FACTOR__ = 1000

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
        self.inshape = None
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
        print(f"Worker {self.hid}-{self.wid} registered in network.", flush=True)

    def init_model(self, model:list[pn.PhenLayer], inshape:tuple):
        self.model = model
        self.inshape = inshape
        n = len(model)
        for lid in range(n):
            model[lid].bind_in_model(inshape, model, lid)
        self.stat_time_layer_prepare = [0.0 for _ in range(n)]
        self.stat_time_layer_compute = [0.0 for _ in range(n)]
        self.stat_time_layer_postprocess = [0.0 for _ in range(n)]
        self.stat_time_layer = [0.0 for _ in range(n)]
        print(f"Worker {self.hid}-{self.wid} load model.", flush=True)

    def run(self, data:np.ndarray, complete_input=True):
        if complete_input == True:
            s = self.model[0].ishaper
            x = s.pick_data(self.hid, self.wid, data)
        else:
            x = data
        for lid, layer in enumerate(self.model):
            print(f"Worker {self.hid}-{self.wid} at Layer-{lid} {self.model[lid].ltype}",
                  flush=True)
            t = time.time()
            if isinstance(layer, pn.PhenConv):
                x = self.comp_conv(x, lid, layer)
            elif isinstance(layer, pn.PhenAvgPool):
                x = self.comp_pool(x, lid, layer)
            elif isinstance(layer, pn.PhenLinear):
                x = self.comp_linear(x, lid, layer)
            elif isinstance(layer, pn.PhenFlatten):
                x = self.comp_flatten(x, lid, layer)
            elif isinstance(layer, pn.PhenReLU) or isinstance(layer, pn.PhenSquare):
                x = self.comp_act(x, lid, layer)
            else:
                print(f"{lid}-th layer of type {self.model[lid].ltype}"
                      " is not supported", flush=True)
            self.sync()
            t = time.time() - t
            self.stat_time_layer[lid] += t
        return x

    def join_result(self, xlocal, root=(0,0)):
        assert len(root) == 2
        assert 0 <= root[0] < self.nh and 0<= root[1] < self.nw
        if root != (self.hid, self.wid):
            self.send(root[0], root[1], xlocal)
            return None
        xmat = np.empty((self.nh, self.nw), dtype=object)
        xmat[self.hid, self.wid] = xlocal
        for i in range(self.nh*self.nw - 1):
            hid, wid, msg = self.recv()
            xmat[hid, wid] = msg
        res = self.model[-1].global_result(xmat)
        return res

    def get_stat(self):
        s = f"Statistics: worker {self.hid}-{self.wid} (w{self.sid}):\n"\
            f"  send {self.stat_send_msg} messages, {self.stat_send_byte} bytes; "\
            f"recv {self.stat_recv_msg} messages, {self.stat_recv_byte} bytes;\n"\
            f"  layer total time: {self.stat_time_layer};\n"\
            f"    prepare time: {self.stat_time_layer_prepare};\n"\
            f"    compute time: {self.stat_time_layer_compute};\n"\
            f"    postprocess time: {self.stat_time_layer_postprocess};\n"\
            f"  synchronization time: {self.stat_time_sync}, "\
            f"send time: {self.stat_time_send}, recv time: {self.stat_time_recv}"
        #print(s, flush=True)
        return s

    # inner functions - compute

    def comp_conv(self, x, lid:int, conv:pn.PhenConv):
        t0 = time.time()
        print(f'  w{self.hid}-{self.wid}: L-{lid} conv-preprocess', flush=True)
        x = self.preprocess(conv, x)
        t1 = time.time()
        print(f'  w{self.hid}-{self.wid}: L-{lid} conv-forward', flush=True)
        x = conv.local_forward(x)
        t2 = time.time()
        print(f'  w{self.hid}-{self.wid}: L-{lid} conv-postprocess', flush=True)
        x = self.postprocess(conv, x)
        t3 = time.time()
        print(f'  w{self.hid}-{self.wid}: L-{lid} conv-done', flush=True)
        self.stat_time_layer_prepare[lid] = t1-t0
        self.stat_time_layer_compute[lid] = t2-t1
        self.stat_time_layer_postprocess[lid] = t3-t2
        return x

    def comp_pool(self, x, lid:int, pool:pn.PhenAvgPool):
        t0 = time.time()
        print(f'  w{self.hid}-{self.wid}: L-{lid} pool-preprocess', flush=True)
        x = self.preprocess(pool, x)
        t1 = time.time()
        print(f'  w{self.hid}-{self.wid}: L-{lid} pool-forward', flush=True)
        x = pool.local_forward(x)
        t2 = time.time()
        print(f'  w{self.hid}-{self.wid}: L-{lid} pool-postprocess', flush=True)
        x = self.postprocess(pool, x)
        t3 = time.time()
        print(f'  w{self.hid}-{self.wid}: L-{lid} pool-done', flush=True)
        self.stat_time_layer_prepare[lid] = t1-t0
        self.stat_time_layer_compute[lid] = t2-t1
        self.stat_time_layer_postprocess[lid] = t3-t2
        return x

    def comp_linear(self, x, lid:int, fc:pn.PhenLinear):
        t0 = time.time()
        print(f'  w{self.hid}-{self.wid}: L-{lid} fc-preprocess', flush=True)
        x = self.preprocess(fc, x)
        t1 = time.time()
        print(f'  w{self.hid}-{self.wid}: L-{lid} fc-forward', flush=True)
        x = fc.local_forward(x)
        t2 = time.time()
        print(f'  w{self.hid}-{self.wid}: L-{lid} fc-postprocess', flush=True)
        x = self.postprocess(fc, x)
        t3 = time.time()
        print(f'  w{self.hid}-{self.wid}: L-{lid} fc-done', flush=True)
        self.stat_time_layer_prepare[lid] = t1-t0
        self.stat_time_layer_compute[lid] = t2-t1
        self.stat_time_layer_postprocess[lid] = t3-t2
        return x

    def comp_flatten(self, x, lid:int, fn:pn.PhenFlatten):
        t = time.time()
        x = fn.local_forward(x)
        t = time.time() - t
        self.stat_time_layer_compute[lid] = t
        return x

    def comp_act(self, x, lid:int, act:pn.PhenReLU):
        t = time.time()
        print(f'  w{self.hid}-{self.wid}: L-{lid} act-forward', flush=True)
        x = act.local_forward(x)
        t = time.time() - t
        print(f'  w{self.hid}-{self.wid}: L-{lid} act-done', flush=True)
        self.stat_time_layer_compute[lid] = t
        return x

    # inner functions - data

    def preprocess(self, layer:pn.PhenLayer, x:np.ndarray):
        # prepare data depdency
        rqtgts = layer.depend_out(x)
        #print(f'w{self.hid}-{self.wid}: send {len(rqtgts)}', flush=True)
        for hid, wid, desc in rqtgts:
            msg = layer.depend_message(x, hid, wid, desc)
            self.send(hid, wid, msg)
        xlist = []
        dep = layer.depend_in(x)
        #print(f'w{self.hid}-{self.wid}: waiting {len(dep)}', flush=True)
        for i in range(len(dep)):
            hid, wid, msg = self.recv()
            xlist.append((hid, wid, msg))
        dep_req = [(h, w) for h, w, d in dep]
        dep_rec = [(h, w) for h, w, d in xlist]
        assert sorted(dep_req) == sorted(dep_rec), \
            f"{self.hid}-{self.wid} req:{dep_req}, recv:{dep_rec}"
        x = layer.depend_merge(x, xlist)
        return x

    def postprocess(self, layer:pn.PhenLayer, x:np.ndarray):
        # load balance
        tgts = layer.join_out(x)
        #print(f'w{self.hid}-{self.wid}: send {len(tgts)}', flush=True)
        for hid, wid, desc in tgts:
            msg = layer.join_message(x, hid, wid, desc)
            self.send(hid, wid, msg)
        req = layer.join_in(x)
        #print(f'w{self.hid}-{self.wid}: waiting {len(req)}', flush=True)
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
        #print(f'w{self.hid}-{self.wid} send {data.shape} ({data.nbytes}) to {hid}-{wid}', flush=True)
        if __DEBUG__:
            data = np.stack([data for _ in range(__INFLATION_FACTOR__)])
        self.net.isend(data, worker_id)
        self.stat_send_msg += 1
        self.stat_send_byte += data.nbytes
        t = time.time() - t
        self.stat_time_send += t


    def recv(self):
        t = time.time()
        source, tag, data = self.net.recv()
        if __DEBUG__:
            data = data[0]
        hid, wid = self.map_worker_s2m(source)
        self.stat_recv_msg += 1
        self.stat_recv_byte += data.nbytes
        #print(f'w{self.hid}-{self.wid} recv {data.shape} ({data.nbytes}) from {hid}-{wid}', flush=True)
        t = time.time() - t
        self.stat_time_recv += t
        return hid, wid, data

    def sync(self):
        t = time.time()
        self.net.barrier()
        t = time.time() - t
        self.stat_time_sync += t
