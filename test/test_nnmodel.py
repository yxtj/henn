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

from collections import defaultdict


def compute_model_meta(nc:int, nh:int, nw:int, model:list[pn.PhenLayer], inshape:tuple):
    gshapes = [ inshape ]
    if isinstance(inshape, int):
        inshape = (inshape, )
    shapers = [ make_shaper(nc, nh, nw, min(2, len(inshape)), inshape) ]
    layer_types = []
    s = inshape
    for lyr in model:
        s = lyr.out_shape(s)
        gshapes.append(s)
        t = lyr.ltype
        layer_types.append(t)
        if t == "conv":
            ss = make_shaper(nc, nh, nw, 2, s)
        elif t == 'linear':
            ss = make_shaper(nc, nh, nw, 1, s)
        elif t == 'flatten':
            ss = make_shaper(nc, nh, nw, 1, s)
        shapers.append(ss)
    return gshapes, shapers, layer_types


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
    gshapes, shapers, layer_types = compute_model_meta(nh, nw, model_p[0][0], inshape)
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


# %% message-based general model runner

def communication_template(nc, nh, nw, inshape, model_t, model_p, ntrails=10):
    gshapes, shapers, layer_types = compute_model_meta(nc, nh, nw, model_p[0][0], inshape)
    print(gshapes)
    #print(shapers)
    print(layer_types)

    for hid in range(nh):
        for wid in range(nw):
            for lid in range(len(model_t)):
                model_p[hid][wid][lid].bind_in_model(shapers[lid], shapers[lid+1],
                                                     lid, gshapes, layer_types)

    cid = 0
    diff = []
    for i in range(ntrails):
        x = torch.rand(inshape)
        #ot = model_t(x.unsqueeze(0))[0].detach().numpy()
        ots = []
        o = x.unsqueeze(0)
        for i, m in enumerate(model_t):
            o = m(o)
            ot = o[0].detach().numpy()
            ots.append(ot)
        # parallel
        xp = x.numpy()
        ips = np.empty((nh, nw), dtype=np.ndarray)
        for hid in range(nh):
            for wid in range(nw):
                ips[hid, wid] = shapers[0].pick_data(cid, hid, wid, xp)
        ops = np.empty((nh, nw), dtype=np.ndarray)
        d = []
        for lid in range(len(model_p[0][0])):
            #print("layer",lid)
            # prepare
            #print("prepare")
            buffer = defaultdict(list)
            for hid in range(nh):
                for wid in range(nw):
                    # send
                    m = model_p[hid][wid][lid]
                    lx = ips[hid, wid]
                    reqs = m.depend_out(lx)
                    for tc, th, tw, desc in reqs:
                        msg = m.depend_message(lx, tc, th, tw, desc)
                        buffer[(th, tw)].append((cid, hid, wid, msg))
            #print("prepare-merge")
            # depend - recv and merge
            for hid in range(nh):
                for wid in range(nw):
                    #print(hid, wid)
                    m = model_p[hid][wid][lid]
                    dep_s = [(c, h, w) for c, h, w, d in m.depend_in(ips[hid,wid])]
                    dep_r = [(c, h, w) for c, h, w, d in buffer[(hid, wid)]]
                    assert sorted(dep_s) == sorted(dep_r), f"s:{dep_s}, r:{dep_r}"
                    lm = m.depend_merge(ips[(hid, wid)], buffer[(hid, wid)])
                    ips[(hid, wid)] = lm
            # compute
            #print("compute")
            buffer = defaultdict(list)
            for hid in range(nh):
                for wid in range(nw):
                    m = model_p[hid][wid][lid]
                    lo = m.local_forward(ips[hid][wid])
                    ops[hid, wid] = lo
                    # send
                    reqs = m.join_out(lo)
                    #print(hid, wid, lo.shape, reqs)
                    for tc, th, tw, desc in reqs:
                        msg = m.join_message(lo, tc, th, tw, desc)
                        buffer[(th, tw)].append((cid, hid, wid, msg))
            # join - recv and merge
            #print("join")
            for hid in range(nh):
                for wid in range(nw):
                    m = model_p[hid][wid][lid]
                    reqs = m.join_in(ops[hid,wid])
                    #print(hid, wid, reqs)
                    dep_s = [(c, h, w) for c, h, w, d in reqs]
                    dep_r = [(c, h, w) for c, h, w, d in buffer[(hid, wid)]]
                    assert sorted(dep_s) == sorted(dep_r), f"s:{dep_s}, r:{dep_r}"
                    ips[hid, wid] = m.join_merge(ops[hid, wid], buffer[(hid, wid)])
                    #print(m.oshaper.get_shape(hid, wid), ips[hid, wid].shape)
            op = m.global_result(ips)
            d.append(np.abs(ots[lid]-op).mean())
        diff.append(d)
    return diff


# %% message-based layer connection

def communicate_linear_linear(nh, nw):
    inshape = 100
    model_t = nn.Sequential(nn.Linear(inshape, 24), nn.Linear(24, 5))
    model_h = [hnt.make_layer(layer) for layer in model_t]
    def make_pmodel(hid, wid):
        m = [pn.PhenLinear(nh, nw, hid, wid, model_h[0]),
             pn.PhenLinear(nh, nw, hid, wid, model_h[1])]
        return m
    model_p = [[make_pmodel(hid, wid) for wid in range(nw)] for hid in range(nh)]

    # gshapes = [(100,), (24,), (5,)]
    diff = communication_template(nh, nw, inshape, model_t, model_p)
    print(f"Model {nh}x{nw} of linear-linear: correct:",
          np.all(np.abs(diff) < 1e-4))
    print("  difference:", np.array(diff))


def communicate_conv_conv(nh, nw):
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

    diff = communication_template(nh, nw, inshape, model_t, model_p)
    print(f"Model {nh}x{nw} of conv-conv: correct:",
          np.all(np.abs(diff) < 1e-4))
    print("  difference:", np.array(diff))


def communicate_conv_linear(nh, nw):
    inshape = (1, 10, 10)

    conv_t = nn.Conv2d(1, 3, 5)
    flt_t = nn.Flatten()
    fc_t  = nn.Linear(108, 10)
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

    diff = communication_template(nh, nw, inshape, model_t, model_p)
    print(f"Model {nh}x{nw} of conv-linear: correct:",
          np.all(np.abs(diff) < 1e-4))
    print("  difference:", np.array(diff))


def communicate_model_stride(nh, nw, stride):
    inshape = (1, 10, 10)

    conv_t = nn.Conv2d(1, 2, 4, stride)
    flt_t = nn.Flatten()
    k = (10-4) // stride + 1
    n = k*k*2
    fc_t  = nn.Linear(n, 10)
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

    diff = communication_template(nh, nw, inshape, model_t, model_p)
    print(f"Model {nh}x{nw} of conv-linear: correct:",
          np.all(np.abs(diff) < 1e-4))
    print("  difference:", np.array(diff))

# %% message-based model with relu

def model_linear_relu_linear(nh, nw):
    inshape = 100
    fc1_t = nn.Linear(inshape, 24)
    act_t = nn.ReLU()
    fc2_t  = nn.Linear(24, 5)
    model_t = nn.Sequential(fc1_t, act_t, fc2_t)

    fc1_h = hnt.make_layer(fc1_t)
    fc2_h = hnt.make_layer(fc2_t)

    fc1_p = [[pn.PhenLinear(nh, nw, hid, wid, fc1_h) for wid in range(nw)]
               for hid in range(nh)]
    act_p = [[pn.PhenReLU(nh, nw, hid, wid) for wid in range(nw)]
             for hid in range(nh)]
    fc2_p = [[pn.PhenLinear(nh, nw, hid, wid, fc2_h) for wid in range(nw)]
               for hid in range(nh)]
    model_p = [[[fc1_p[hid][wid], act_p[hid][wid], fc2_p[hid][wid]]
               for wid in range(nw)]
               for hid in range(nh)]

    diff = communication_template(nh, nw, inshape, model_t, model_p)
    print(f"Model {nh}x{nw} of linear-relu-linear: correct:",
          np.all(np.abs(diff) < 1e-4))
    print("  difference:", np.array(diff))


def model_conv_relu_conv(nh, nw):
    inshape = (1, 10, 10)

    conv1_t = nn.Conv2d(1, 3, 5)
    act_t = nn.ReLU()
    conv2_t  = nn.Conv2d(3, 2, 3)
    model_t = nn.Sequential(conv1_t, act_t, conv2_t)
    conv1_h = hnt.make_conv2d(conv1_t)
    conv2_h = hnt.make_conv2d(conv2_t)

    conv1_p = [[pn.PhenConv(nh, nw, hid, wid, conv1_h) for wid in range(nw)]
               for hid in range(nh)]
    act_p = [[pn.PhenReLU(nh, nw, hid, wid) for wid in range(nw)]
             for hid in range(nh)]
    conv2_p = [[pn.PhenConv(nh, nw, hid, wid, conv2_h) for wid in range(nw)]
               for hid in range(nh)]
    model_p = [[[conv1_p[hid][wid], act_p[hid][wid], conv2_p[hid][wid]]
               for wid in range(nw)]
               for hid in range(nh)]

    diff = communication_template(nh, nw, inshape, model_t, model_p)
    print(f"Model {nh}x{nw} of conv-relu-conv: correct:",
          np.all(np.abs(diff) < 1e-4))
    print("  difference:", np.array(diff))


def model_conv_relu_linear(nh, nw):
    inshape = (1, 10, 10)

    conv_t = nn.Conv2d(1, 3, 5)
    act_t = nn.ReLU()
    flt_t = nn.Flatten()
    fc_t  = nn.Linear(108, 10)
    model_t = nn.Sequential(conv_t, act_t, flt_t, fc_t)
    conv_h = hnt.make_conv2d(conv_t)
    fc_h = hnt.make_linear(fc_t)

    conv_p = [[pn.PhenConv(nh, nw, hid, wid, conv_h) for wid in range(nw)]
               for hid in range(nh)]
    act_p = [[pn.PhenReLU(nh, nw, hid, wid) for wid in range(nw)]
             for hid in range(nh)]
    flt_p = [[pn.PhenFlatten(nh, nw, hid, wid) for wid in range(nw)]
             for hid in range(nh)]
    fc_p = [[pn.PhenLinear(nh, nw, hid, wid, fc_h) for wid in range(nw)]
               for hid in range(nh)]
    model_p = [[[conv_p[hid][wid], act_p[hid][wid], flt_p[hid][wid], fc_p[hid][wid]]
               for wid in range(nw)]
               for hid in range(nh)]

    diff = communication_template(nh, nw, inshape, model_t, model_p)
    print(f"Model {nh}x{nw} of conv-relu-linear: correct:",
          np.all(np.abs(diff) < 1e-4))
    print("  difference:", np.array(diff))


def model_big_2c2l(nc, nh, nw):
    inshape = (1, 202, 202)

    conv1_t = nn.Conv2d(1, 3, 4, 4)
    conv2_t = nn.Conv2d(3, 5, 4, 4)
    fc1_t  = nn.Linear(720, 100)
    fc2_t  = nn.Linear(100, 10)
    model_t = nn.Sequential(conv1_t, nn.ReLU(), conv2_t, nn.ReLU(),
                            nn.Flatten(), fc1_t, nn.ReLU(), fc2_t)

    conv1_h = hnt.make_conv2d(conv1_t)
    conv2_h = hnt.make_conv2d(conv2_t)
    fc1_h = hnt.make_linear(fc1_t)
    fc2_h = hnt.make_linear(fc2_t)

    def make_phen_model(nc, nh, nw, cid, hid, wid, inshape:tuple, model_t:nn.Sequential):
        res = []
        for i, m in enumerate(model_t):
            if isinstance(m, nn.Conv2d):
                mh = hnt.make_layer(m)
                mp = pn.PhenConv(nc, nh, nw, cid, hid, wid, mh)
            elif isinstance(m, nn.Linear):
                mh = hnt.make_layer(m)
                mp = pn.PhenLinear(nc, nh, nw, cid, hid, wid, mh)
            elif isinstance(m, nn.ReLU):
                mp = pn.PhenReLU(nc, nh, nw, cid, hid, wid)
            elif isinstance(m, nn.Flatten):
                mp = pn.PhenFlatten(nc, nh, nw, cid, hid, wid)
            else:
                print(f'{i}-th layer {m} is not supported.')
            res.append(mp)
        return res

    cid = 0
    model_p = [[[pn.PhenConv(nc, nh, nw, cid, hid, wid, conv1_h),
                 pn.PhenReLU(nc, nh, nw, cid, hid, wid),
                 pn.PhenConv(nc, nh, nw, cid, hid, wid, conv2_h),
                 pn.PhenReLU(nc, nh, nw, cid, hid, wid),
                 pn.PhenFlatten(nc, nh, nw, cid, hid, wid),
                 pn.PhenLinear(nc, nh, nw, cid, hid, wid, fc1_h),
                 pn.PhenReLU(nc, nh, nw, cid, hid, wid),
                 pn.PhenLinear(nc, nh, nw, cid, hid, wid, fc2_h)]
        for wid in range(nw)]
        for hid in range(nh)]

    model_p = [[make_phen_model(nc, nh, nw, cid, hid, wid, inshape, model_t)
                for wid in range(nw)]
               for hid in range(nh)]

    diff = communication_template(nc, nh, nw, inshape, model_t, model_p, 10)
    print(f"Model {nc}x{nh}x{nw} of 2c2l: correct:",
          np.all(np.abs(diff) < 1e-4))
    print("  difference:", np.array(diff))

# %% main

def test_connection():
    connect_linear_linear(2, 2)
    connect_conv_conv(2, 2)
    connect_conv_linear(2, 2)

def test_message():
    communicate_linear_linear(2, 2)
    communicate_conv_conv(2, 2)
    communicate_conv_conv(3, 3)
    communicate_conv_linear(2, 2)

def test_message_relu():
    model_linear_relu_linear(2,2)
    model_conv_relu_conv(2,2)
    model_conv_relu_conv(3,3)
    model_conv_relu_linear(2,2)
    model_conv_relu_linear(3,3)

def main():
    # connect layers
    #test_connection()

    # message-based layer connection
    #test_message()
    #test_message_relu()

    #communicate_model_stride(2,2,2)

    model_big_2c2l(1, 3, 3)


if __name__ == "__main__":
    main()
