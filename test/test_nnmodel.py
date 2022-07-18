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


# %% connection

def connect_linear_linear(nh, nw):
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


def connect_conv_linear(nh, nw):
    inshape = (1, 10, 10)

    conv_t = nn.Conv2d(1, 2, 3)
    flt_t = nn.Flatten()
    fc_t = nn.Linear(128, 5)
    model_t = nn.Sequential(conv_t, flt_t, fc_t)
    conv_h = hnt.make_conv2d(conv_t)
    fc_h = hnt.make_linear(fc_t)

    conv_p = [[pn.PhenConv(nh, nw, hid, wid, conv_h) for wid in range(nw)]
               for hid in range(nh)]
    flt_p = [[pn.PhenFlatten(nh, nw, hid, wid) for wid in range(nw)]
               for hid in range(nh)]
    fc_p = [[pn.PhenLinear(nh, nw, hid, wid, fc_h) for wid in range(nw)]
               for hid in range(nh)]
    model_p = [[[conv_p[hid][wid], flt_p[hid][wid], fc_p[hid][wid]]
               for wid in range(nw)]
               for hid in range(nh)]

    shaper1 = make_shaper(nh, nw, 2, (10, 10), interleave=True)
    shaper2 = make_shaper(nh, nw, 2, (8, 8), interleave=True)
    shaper3 = make_shaper(nh, nw, 1, (128,), interleave=True)
    shaper4 = make_shaper(nh, nw, 1, (5,), interleave=True)
    shapers = [shaper1, shaper2, shaper3, shaper4]
    gshapes = [inshape, (2, 8, 8), (128,), (5)]
    layer_types = ["conv", "flatten", "linear"]

    for hid in range(nh):
        for wid in range(nw):
            for lid in range(3):
                model_p[hid][wid][lid].bind_in_model(shapers[lid], shapers[lid+1],
                                                     lid, gshapes, layer_types)

    diff = []
    for i in range(10):
        x = torch.rand(inshape)
        #ot = model_t(x.unsqueeze(0))[0].detach().numpy()
        o = conv_t(x.unsqueeze(0))
        ot1 = o[0].detach().numpy()
        o = flt_t(o)
        ot2 = o[0].detach().numpy()
        o = fc_t(o)
        ot = o[0].detach().numpy()
        # parallel
        xp = x.numpy()
        # layer 1 (conv)
        ops1 = np.empty((nh, nw), dtype=np.ndarray)
        for hid in range(nh):
            for wid in range(nw):
                m = model_p[hid][wid][0]
                box = m._calc_expected_in_box_(hid, wid)
                cut = xp[:, box[0]:box[2], box[1]:box[3]]
                ops1[hid, wid] = m.local_forward(cut)
        #print([o.shape for o in ops1.ravel()])
        op1 = model_p[0][0][0].global_result(ops1)
        d1 = np.abs(ot1-op1).mean()
        # layer 2 (flatten)
        ops2 = np.empty((nh, nw), dtype=np.ndarray)
        for hid in range(nh):
            for wid in range(nw):
                m = model_p[hid][wid][1]
                cut = shaper2.pick_data(hid, wid, op1)
                ops2[hid][wid] = m.local_forward(cut)
        #print([o.shape for o in ops2.ravel()])
        op2 = model_p[0][0][1].global_result(ops2)
        d2 = np.abs(ot2-op2).mean()
        # layer 3 (linear)
        ops3 = np.empty((nh, nw), dtype=np.ndarray)
        for hid in range(nh):
            for wid in range(nw):
                m = model_p[hid][wid][2]
                cut = ops2[hid][wid]
                ops3[hid, wid] = m.local_forward(cut)
        #print([o.shape for o in ops3.ravel()])
        op = heutil.hesum(ops3.ravel())
        d = np.abs(ot-op).mean()
        diff.append((d1,d2,d))
    print(f"Model {nh}x{nw} of conv-linear: correct:",
          np.all(np.abs(diff) < 1e-4))
    print("  difference:", np.array(diff))


def connect_conv_conv(nh, nw):
    inshape = (1, 10, 10)

    conv1_t = nn.Conv2d(1, 3, 5)
    conv2_t  = nn.Conv2d(3, 2, 3)
    model_t = nn.Sequential(conv1_t, conv2_t)
    conv1_h = hnt.make_conv2d(conv1_t)
    conv2_h = hnt.make_conv2d(conv2_t)

    conv1_p = [[pn.PhenConv(nh, nw, hid, wid, conv1_h) for wid in range(nw)]
               for hid in range(nh)]
    conv2_p = [[pn.PhenConv(nh, nw, hid, wid, conv2_h) for wid in range(nw)]
               for hid in range(nh)]
    model_p = [[[conv1_p[hid][wid], conv2_p[hid][wid]]
               for wid in range(nw)]
               for hid in range(nh)]

    #gshapes = [inshape, (3, 6, 6), (2, 4, 4)]
    #layer_types = ["conv", "conv"]
    gshapes = [ inshape ]
    shapers = [ make_shaper(nh, nw, 2, inshape) ]
    layer_types = []
    s = inshape
    for lyr in model_p[0][0]:
        s = lyr.out_shape(s)
        gshapes.append(s)
        ss = make_shaper(nh, nw, 2, s)
        shapers.append(ss)
        t = lyr.ltype
        layer_types.append(t)
    print(gshapes)
    print(shapers)
    print(layer_types)

    for hid in range(nh):
        for wid in range(nw):
            for lid in range(len(model_t)):
                model_p[hid][wid][lid].bind_in_model(shapers[lid], shapers[lid+1],
                                                     lid, gshapes, layer_types)

    diff = []
    for i in range(10):
        x = torch.rand(inshape)
        #ot = model_t(x.unsqueeze(0))[0].detach().numpy()
        o = conv1_t(x.unsqueeze(0))
        ot1 = o[0].detach().numpy()
        o = conv2_t(o)
        ot = o[0].detach().numpy()
        # parallel
        xp = x.numpy()
        # layer 1
        ops1 = np.empty((nh, nw), dtype=np.ndarray)
        for hid in range(nh):
            for wid in range(nw):
                m = model_p[hid][wid][0]
                box = m._calc_expected_in_box_(hid, wid)
                cut = xp[:, box[0]:box[2], box[1]:box[3]]
                ops1[hid, wid] = m.local_forward(cut)
        #print([o.shape for o in ops1.ravel()])
        op1 = model_p[0][0][0].global_result(ops1)
        d1 = np.abs(ot1-op1).mean()
        # layer 2
        ops2 = np.empty((nh, nw), dtype=np.ndarray)
        for hid in range(nh):
            for wid in range(nw):
                m = model_p[hid][wid][1]
                box = m._calc_expected_in_box_(hid, wid)
                cut = op1[:, box[0]:box[2], box[1]:box[3]]
                ops2[hid, wid] = m.local_forward(cut)
        #print([o.shape for o in ops3.ravel()])
        op = model_p[0][0][1].global_result(ops2)
        d = np.abs(ot-op).mean()
        diff.append((d1,d))
    print(f"Model {nh}x{nw} of conv-conv: correct:",
          np.all(np.abs(diff) < 1e-4))
    print("  difference:", np.array(diff))

# %% complete model


# %% main

def main():
    # connect different layers
    #connect_linear_linear(2, 2)
    #connect_conv_linear(2, 2)
    connect_conv_conv(2, 2)


if __name__ == "__main__":
    main()
