# -*- coding: utf-8 -*-

import parallel.phenetwork as pn

import numpy as np
import hennlayer as hn
import torch
import torch.nn as nn
import hennlayer_torch as hnt
import hecomp


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
    
    
def parallel_linear(nh=2, nw=2, nchin=100):
    fc_t = nn.Linear(nchin, 10)
    fc_h = hnt.make_linear(fc_t)
    fcs = [[pn.PhenLinear(nh, nw, hid, wid, fc_h) for wid in range(nw)]
          for hid in range(nh)]
    ind = np.linspace(0, nchin, nh*nw+1, dtype=int)
    
    diff = []
    for _ in range(10):
        x = torch.rand((nchin))
        ot = fc_t(x.unsqueeze(0)).detach().numpy()
        xp = [x[ind[i]:ind[i+1]] for i in range(nh*nw)]
        ops = []
        for hid in range(nh):
            for wid in range(nw):
                fc = fcs[hid][wid]
                o = fc(xp[fc.pid])
                ops.append(o)
        op = hecomp.hesum(ops)
        d = np.abs(ot-op).mean()
        diff.append(d)
    print(f"Conv parallel {nh}x{nw} correct:", np.all(np.abs(diff)<1e-4))
    print("  difference:",diff)
    

def main():
    # correctness
    #test_linear()
    #test_conv()
    
    # parallel
    parallel_linear(2, 2, 100)
    parallel_linear(3, 2, 100)
    parallel_linear(3, 3, 100)
    

if __name__ == "__main__":
    main()
    