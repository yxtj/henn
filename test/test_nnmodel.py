# -*- coding: utf-8 -*-

import parallel.phenetwork as pn
from parallel.shaper import Shaper, make_shaper

import numpy as np
import hennlayer as hn
import torch
import torch.nn as nn
import hennlayer_torch as hnt
import heutil
import computil


def compute_model_shape(model: list[pn.PhenLayer], inshape: tuple):
    s = inshape
    res = []
    res.append(s)
    for layer in model:
        s = layer.out_shape(s)
        res.append(s)
    return res


def con_conv_conv():
    pass


def con_linear_linear(nh, nw):
    inshape = 100
    model_t = nn.Sequential(nn.Linear(inshape, 24), nn.Linear(24, 5))
    model_h = [hnt.make_layer(layer) for layer in model_t]
    def make_pmodel(hid, wid):
        m = [pn.PhenLinear(nh, nw, hid, wid, model_h[0]),
             pn.PhenLinear(nh, nw, hid, wid, model_h[1])]
        return m
    model_p = [[make_pmodel(hid, wid) for wid in range(nw)]
               for hid in range(nh)]

    shaper1 = make_shaper(nh, nw, 1, (inshape,), interleave=True)
    shaper2 = make_shaper(nh, nw, 1, (24,), interleave=True)
    shaper3 = make_shaper(nh, nw, 1, (5,), interleave=True)
    shapers = [shaper1, shaper2, shaper3]

    diff = []
    for i in range(10):
        x = torch.rand(inshape)
        ot = model_t(x.unsqueeze(0))[0].detach().numpy()
        xp = x.numpy()
        ops = [[shaper1.pick_data(i, j, xp) for j in range(nw)]
               for i in range(nh)]
        for lid in range(len(model_h)):
            o = np.empty((nh, nw), object)
            for hid in range(nh):
                for wid in range(nw):
                    m = model_p[hid][wid][lid]
                    o[hid, wid] = m.local_forward(ops[hid][wid])
            o = heutil.hesum(o.ravel())
            shaper = shapers[lid+1]
            ops = [[shaper.pick_data(i, j, o) for j in range(nw)]
                   for i in range(nh)]
        op = o
        d = np.abs(ot-op).mean()
        diff.append(d)
    print(f"Model {nh}x{nw} of linear-linear: correct:",
          np.all(np.abs(diff) < 1e-4))
    print("  difference:", diff)


def con_conv_linear(nh, nw):
    model_t = nn.Sequential(nn.Conv2d(1, 3, 3), nn.Linear(24, 5))
    model_h = [hnt.make_conv2d(model_t[0]), hnt.make_linear(model_t[1])]
    model_p = [
        [[pn.PhenConv(nh, nw, i, j, model_h[0]) for j in range(nw)]
         for i in range(nh)],
        [[pn.PhenLinear(nh, nw, i, j, model_h[1])
          for j in range(nw)] for i in range(nh)]
    ]

    inshape = (1, 10, 10)

    diff = []
    print(f"Model {nh}x{nw} of conv-linear: correct:",
          np.all(np.abs(diff) < 1e-4))
    print("  difference:", diff)


def main():
    # con_conv_conv()
    con_linear_linear(2, 2)
    #con_conv_linear(2, 2)


if __name__ == "__main__":
    main()
