# -*- coding: utf-8 -*-
import parallel.phenetwork as pn
from parallel.worker import Worker
from network import Network

#import numpy as np
import torch.nn as nn
import hennlayer_torch as hnt


def make_phen_model(nh:int, nw:int, hid:int, wid:int,
                    model_t:nn.Sequential, inshape:tuple):
    res = []
    for i, m in enumerate(model_t):
        if isinstance(m, nn.Conv2d):
            mh = hnt.make_layer(m)
            mp = pn.PhenConv(nh, nw, hid, wid, mh)
        elif isinstance(m, nn.Linear):
            mh = hnt.make_layer(m)
            mp = pn.PhenLinear(nh, nw, hid, wid, mh)
        elif isinstance(m, nn.ReLU):
            mp = pn.PhenReLU(nh, nw, hid, wid)
        elif isinstance(m, nn.Flatten):
            mp = pn.PhenFlatten(nh, nw, hid, wid)
        else:
            print(f'{i}-th layer {m} is not supported.')
        res.append(mp)
    return res


def get_phen_model_info(model_p, inshape:tuple):
    shapes = [inshape]
    ltypes = []
    s = inshape
    for i, m in enumerate(model_p):
        s = m.out_shape(s)
        shapes.append(s)
        ltypes.append(m.ltype)
    #print('\n'.join([f'  Layer-{i} {ltypes[i]}: {shapes[i]} -> {shapes[i+1]}'
    #                 for i in range(len(ltypes))]), flush=True)
    return shapes, ltypes


def setup_worker(nh:int, nw:int, net:Network, model_p, inshape:tuple):
    hid, wid = divmod(net.rank, nw)

    w = Worker(hid, wid, nh, nw)
    w.init_model(model_p, inshape)
    w.init_network()

    return w



