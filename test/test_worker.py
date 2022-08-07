# -*- coding: utf-8 -*-
import parallel.phenetwork as pn
import parallel.worker as worker
from network import Network

import time
import numpy as np
import torch
import torch.nn as nn
import hennlayer_torch as hnt

import sys

# %% simple test

def simple_test_1():
    net = Network()
    assert net.size == 1

    inshape = (1, 10, 10)
    model_t = nn.Sequential(nn.Conv2d(1, 3, 4),
                            nn.Flatten(),
                            nn.Linear(147, 10))
    model_h = [hnt.make_layer(model_t[0]), hnt.make_layer(model_t[2])]
    model_p = [pn.PhenConv(1, 1, 0, 0, model_h[0]),
               pn.PhenFlatten(1, 1, 0, 0),
               pn.PhenLinear(1, 1, 0, 0, model_h[1]),]

    w = worker.Worker(0, 0, 1, 1)
    w.init_model(model_p, inshape)
    w.init_network()


    data = np.random.random(inshape)
    r = w.run(data)
    w.show_stat()

    g = w.join_result(r)
    ot = model_t(torch.from_numpy(data).float().unsqueeze(0))
    diff = np.abs(ot.detach().numpy() - g).mean()
    print("difference:", diff)

# two workers
def simple_test_2():
    net = Network()
    assert net.size == 2
    nh, nw = 1, net.size
    hid, wid = 0, net.rank

    inshape = (1, 10, 10)
    model_t = nn.Sequential(nn.Conv2d(1, 3, 4),
                            nn.Flatten(),
                            nn.Linear(147, 10))
    model_h = [hnt.make_layer(model_t[0]), hnt.make_layer(model_t[2])]
    model_p = [pn.PhenConv(nh, nw, hid, wid, model_h[0]),
               pn.PhenFlatten(nh, nw, hid, wid),
               pn.PhenLinear(nh, nw, hid, wid, model_h[1]),]

    w = worker.Worker(hid, wid, nh, nw)
    w.init_model(model_p, inshape)
    w.init_network()

    data = np.random.random(inshape)
    t = time.time()
    r = w.run(data)
    t = time.time() - t
    w.show_stat()

    g = w.join_result(r)
    if net.rank == 0:
        print("Total Time:", t)
        ot = model_t(torch.from_numpy(data).float().unsqueeze(0))
        diff = np.abs(ot.detach().numpy() - g).mean()
        print("difference:", diff)

def big_test(nh, nw):
    net = Network()
    assert net.size == nh * nw
    hid, wid = divmod(net.rank, nw)
    #hid, wid = 0, 0

    inshape = (1, 202, 202)
    model_t = nn.Sequential(nn.Conv2d(1, 3, 4, 4),
                            nn.Conv2d(3, 5, 4, 4),
                            nn.Flatten(),
                            nn.Linear(720, 100),
                            nn.Linear(100, 10))
    model_h = [hnt.make_layer(model_t[0]), hnt.make_layer(model_t[1]),
               hnt.make_layer(model_t[3]), hnt.make_layer(model_t[4])]
    model_p = [pn.PhenConv(nh, nw, hid, wid, model_h[0]),
               pn.PhenConv(nh, nw, hid, wid, model_h[1]),
               pn.PhenFlatten(nh, nw, hid, wid),
               pn.PhenLinear(nh, nw, hid, wid, model_h[2]),
               pn.PhenLinear(nh, nw, hid, wid, model_h[3]),]

    w = worker.Worker(hid, wid, nh, nw)
    w.init_model(model_p, inshape)
    w.init_network()

    data = np.random.random(inshape)
    t = time.time()
    r = w.run(data)
    t = time.time() - t
    w.show_stat()

    g = w.join_result(r)
    if net.rank == 0:
        print("Total Time:", t)
        ot = model_t(torch.from_numpy(data).float().unsqueeze(0))
        diff = np.abs(ot.detach().numpy() - g).mean()
        print("Difference:", diff)


# %% general test

def make_phen_model(nh, nw, hid, wid, inshape:tuple, model_t:nn.Sequential):
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


def general_worker_run(nh, nw, inshape, model_t):
    net = Network()
    assert net.size == nh * nw
    hid, wid = divmod(net.rank, nw)

    model_p = make_phen_model(nh, nw, hid, wid, inshape, model_t)

    w = worker.Worker(hid, wid, nh, nw)
    w.init_model(model_p, inshape)
    w.init_network()

    data = np.random.random(inshape)
    t = time.time()
    r = w.run(data)
    t = time.time() - t
    w.show_stat()

    g = w.join_result(r)
    if net.rank == 0:
        print("Total Time:", t)
        ot = model_t(torch.from_numpy(data).float().unsqueeze(0))
        diff = np.abs(ot.detach().numpy() - g).mean()
        print("Difference:", diff)


# %% main

def basic_test():
    simple_test_1()
    simple_test_2()
    big_test(2, 2)

def test_case1(nh=2, nw=2):
    inshape = (1, 202, 202)
    model_t = nn.Sequential(nn.Conv2d(1, 3, 4, 4), nn.ReLU(),
                            nn.Flatten(),
                            nn.Linear(7500, 100), nn.ReLU(),
                            nn.Linear(100, 10))
    general_worker_run(nh, nw, inshape, model_t)

def test_case2(nh=2, nw=2):
    inshape = (1, 202, 202)
    model_t = nn.Sequential(nn.Conv2d(1, 3, 4, 4), nn.ReLU(),
                            nn.Conv2d(3, 5, 4, 4), nn.ReLU(),
                            nn.Flatten(),
                            nn.Linear(720, 100), nn.ReLU(),
                            nn.Linear(100, 10))
    general_worker_run(nh, nw, inshape, model_t)


def main():
    torch.manual_seed(123)
    np.random.seed(1)

    #basic_test()

    #test_case1(2, 2)
    #test_case1(3, 3)

    #test_case2(2, 2)
    #test_case2(2, 3)

    net = Network()
    if len(sys.argv) == 4:
        c = int(sys.argv[1])
        nh = int(sys.argv[2])
        nw = int(sys.argv[3])
        assert nh*nw == net.size
        if c == 1:
            test_case1(nh, nw)
        elif c == 2:
            test_case2(nh, nw)
        else:
            print(f"test case {c} not supported")
    else:
        if net.rank == 0:
            print("Usage: <case-id> <nh> <nw>")


if __name__ == "__main__":
    main()
