# -*- coding: utf-8 -*-

import parallel.phenetwork as pn

import numpy as np
import hennlayer as hn
import torch
import torch.nn as nn
import hennlayer_torch as hnt
import heutil
import computil

# %% correctness

def test_linear():
    fc_t = nn.Linear(5, 10)
    #fc_h = hn.Linear(5, 10)
    #fc_h = hnt.bind_torch(fc_h, fc_t)
    fc_h = hnt.make_linear(fc_t)
    fc_p = pn.PhenLinear(1, 1, 0, 0, fc_h)

    diff = []
    for i in range(10):
        x = torch.rand((5))
        ot = fc_t(x.unsqueeze(0))
        op = fc_p(x.numpy())
        d = np.abs(ot[0].detach().numpy()-op).mean()
        diff.append(d)
    print("Linear correct:", np.all(np.abs(diff)<1e-4))
    print("difference:",diff)


def test_conv():
    conv_t = nn.Conv2d(2, 4, 3)
    conv_h = hnt.make_conv2d(conv_t)
    conv_p = pn.PhenConv(1, 1, 0, 0, conv_h)

    diff = []
    for i in range(10):
        x = torch.rand((2,10,10))
        ot = conv_t(x.unsqueeze(0))
        op = conv_p(x.numpy())
        d = np.abs(ot[0].detach().numpy()-op).mean()
        diff.append(d)
    print("Conv correct:", np.all(np.abs(diff)<1e-4))
    print("  difference:",diff)


def test_relu():
    act_t = nn.ReLU()
    act_p = pn.PhenRelu(1, 1, 0, 0)

    diff = []
    for i in range(10):
        x = torch.rand((1,10))
        ot = act_t(x.unsqueeze(0))
        op = act_p(x.numpy())
        d = np.abs(ot[0].detach().numpy()-op).mean()
        diff.append(d)
    print("ReLU correct:", np.all(np.abs(diff)<1e-4))
    print("  difference:",diff)


# %% parallelization of one layer

def parallel_linear(nh=2, nw=2, nchin=100, nchout=10):
    fc_t = nn.Linear(nchin, nchout)
    fc_h = hnt.make_linear(fc_t)
    fcs = [[pn.PhenLinear(nh, nw, hid, wid, fc_h) for wid in range(nw)]
          for hid in range(nh)]
    #ind = np.linspace(0, nchin, nh*nw+1, dtype=int)

    diff = []
    for _ in range(10):
        x = torch.rand((nchin))
        ot = fc_t(x.unsqueeze(0)).detach().numpy()
        xp = x.numpy()
        ops = []
        for hid in range(nh):
            for wid in range(nw):
                fc = fcs[hid][wid]
                #cut = xp[ind[fc.pid]:ind[fc.pid+1]]
                cut = xp[fc.pid::fc.npart]
                o = fc.local_forward(cut)
                ops.append(o)
        op = heutil.hesum(ops)
        d = np.abs(ot-op).mean()
        diff.append(d)
    print(f"Parallel {nh}x{nw}: linear correct:", np.all(np.abs(diff)<1e-4))
    print("  difference:",diff)


def parallel_conv(nh=2, nw=2, ks=3, stride=1, pad=0, sz=10):
    conv_t = nn.Conv2d(2, 4, ks, stride, pad)
    conv_h = hnt.make_conv2d(conv_t)
    convs = [[pn.PhenConv(nh, nw, hid, wid, conv_h) for wid in range(nw)]
          for hid in range(nh)]

    psx = sz + 2*pad
    psy = sz + 2*pad
    indh = np.linspace(0, psx, nh+1, dtype=int)
    indw = np.linspace(0, psy, nw+1, dtype=int)
    # guarantees that indh[i] is the first position with an output
    if stride != 1:
        for i in range(nh):
            q, r = divmod(indh[i], stride)
            if r != 0:
                indh[i] = (q+1)*stride
        for i in range(nw):
            q, r = divmod(indw[i], stride)
            if r != 0:
                indw[i] = (q+1)*stride

    diff = []
    for _ in range(10):
        x = torch.rand((2, sz, sz))
        ot = conv_t(x.unsqueeze(0))[0].detach().numpy()
        xp = computil.pad_data(x.numpy(), pad)
        ops = np.empty((nh, nw), dtype=object)
        for hid in range(nh):
            for wid in range(nw):
                conv = convs[hid][wid]
                #h1, h2 = indh[hid], min(psx, indh[hid+1]+ks-1)
                #w1, w2 = indw[wid], min(psy, indw[wid+1]+ks-1)
                h1, h2 = indh[hid], min(psx, indh[hid+1]-stride+1+ks-1)
                w1, w2 = indw[wid], min(psy, indw[wid+1]-stride+1+ks-1)
                cut = xp[:, h1:h2, w1:w2]
                #print(hid, wid, 'h:',h1,h2,'w:',w1,w2, 'shape:', cut.shape)
                o = conv.local_forward(cut)
                ops[hid,wid] = o
        op = np.concatenate([np.concatenate(ops[i,:],2) for i in range(nh)], 1)
        d = np.abs(ot-op).mean()
        diff.append(d)
    print(f"Parallel {nh}x{nw}: Conv kernel-{ks}, stride-{stride}, pad-{pad}, correct:", np.all(np.abs(diff)<1e-4))
    print("  difference:",diff)


def parallel_act1(nh=2, nw=2, size=10):
    act_t = nn.ReLU()
    acts = [[pn.PhenReLU(nh, nw, hid, wid) for wid in range(nw)]
          for hid in range(nh)]
    ind = np.linspace(0, size, nh*nw+1, dtype=int)

    diff = []
    for _ in range(10):
        x = torch.rand(size)
        ot = act_t(x.unsqueeze(0)).detach().numpy()
        xp = x.numpy()
        ops = []
        for hid in range(nh):
            for wid in range(nw):
                act = acts[hid][wid]
                cut = xp[ind[act.pid]:ind[act.pid+1]]
                o = act.local_forward(cut)
                ops.append(o)
        op = np.concatenate(ops)
        d = np.abs(ot-op).mean()
        diff.append(d)
    print(f"Parallel {nh}x{nw}: for 1D relu correct:", np.all(np.abs(diff)<1e-4))
    print("  difference:",diff)


def parallel_act2(nh=2, nw=2, szx=10, szy=10):
    act_t = nn.ReLU()
    acts = [[pn.PhenReLU(nh, nw, hid, wid) for wid in range(nw)]
          for hid in range(nh)]
    indh = np.linspace(0, szx, nh+1, dtype=int)
    indw = np.linspace(0, szy, nw+1, dtype=int)

    diff = []
    for _ in range(10):
        x = torch.rand((szx, szy))
        ot = act_t(x.unsqueeze(0)).detach().numpy()[0]
        xp = x.numpy()
        ops = np.empty((nh, nw), dtype=object)
        for hid in range(nh):
            for wid in range(nw):
                act = acts[hid][wid]
                cut = xp[indh[hid]:indh[hid+1], indw[wid]:indw[wid+1]]
                o = act.local_forward(cut)
                ops[hid, wid] = o
        op = np.concatenate([np.concatenate(ops[i,:],1) for i in range(nh)],0)
        d = np.abs(ot-op).mean()
        diff.append(d)
    print(f"Parallel {nh}x{nw}: for 2D relu correct:", np.all(np.abs(diff)<1e-4))
    print("  difference:",diff)

# %% main

def correct_test():
    #test_linear()
    #test_conv()
    test_relu()

def linear_test():
    parallel_linear(2, 2, 100, 10)
    parallel_linear(3, 2, 100, 10)
    parallel_linear(3, 3, 100, 10)

def conv_test():
    parallel_conv(2, 2, 3, 1, 0, 9)
    parallel_conv(2, 2, 3, 1, 0, 10)
    parallel_conv(2, 2, 3, 1, 1, 10)
    parallel_conv(2, 2, 3, 2, 0, 10)
    parallel_conv(2, 2, 3, 2, 1, 10)

def act_test():
    parallel_act1(2, 2, 10)
    parallel_act1(3, 3, 10)
    parallel_act2(2, 2, 10, 10)
    parallel_act2(3, 3, 10, 10)

def main():
    # correctness
    correct_test()

    # parallel linear
    linear_test()

    # parallel conv
    conv_test()

    # parallel activation
    act_test()

if __name__ == "__main__":
    main()
