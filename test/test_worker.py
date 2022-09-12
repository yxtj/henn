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

    if net.rank == 0:
        #info = [(i, m.ishaper.gshape, m.ltype) for i, m in enumerate(model_p)]
        shapes = [inshape]
        ltypes = []
        s = inshape
        for i, m in enumerate(model_p):
            s = m.out_shape(s)
            shapes.append(s)
            ltypes.append(m.ltype)
        print('\n'.join([f'  Layer-{i} {ltypes[i]}: {shapes[i]} -> {shapes[i+1]}'
                         for i in range(len(ltypes))]), flush=True)

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

def test_alex(nh=2, nw=2):
    inshape = (3, 227, 227)
    # 3*227*227 -conv(11, 4)-> 96*55*55 -max(3,2)-> 96*27*27
    # -conv(5,1,2)-> 256*27*27 -max(3,2)-> 256*13*13
    # -conv(3,1,1)-> 384*13*13
    # -conv(3,1,1)-> 384*13*13
    # -conv(3,1,1)-> 256*13*13 -max(3,2)-> 256*6*6
    # -fc-> 4096 -fc-> 4096 -fc-> 1000
    model_t = nn.Sequential(
        nn.Conv2d(3, 96, 11, 4), nn.ReLU(), nn.MaxPool2d(3, 2),
        nn.Conv2d(256, 96, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(3, 2),
        nn.Conv2d(256, 384, 3, 1, 1), nn.ReLU(),
        nn.Conv2d(384, 384, 3, 1, 1), nn.ReLU(),
        nn.Conv2d(384, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(3, 2),
        nn.Flatten(),
        nn.Linear(4096, 4096), nn.ReLU(),
        nn.Linear(4096, 4096), nn.ReLU(),
        nn.Linear(4096, 1000))
    general_worker_run(nh, nw, inshape, model_t)

def test_vgg(nh=2, nw=2, layers=16):
    inshape = (3, 224, 224)
    def make_block(inch, outch, n):
        l = [nn.Conv2d(inch, outch, 3, 1, 1)]
        l += [nn.Conv2d(outch, outch, 3, 1, 1) for i in range(n-1)]
        l += [nn.MaxPool2d(2, 2)]
        return l
    # 3*224*224 -conv*-> 64*224*224 -max-> 128*112*112
    # -conv-> 256*112*112 -conv-> 256*112*112 -max-> 256*56*56
    # -conv-> 512*56*56 -conv*-> 512*56*56 -max-> 512*28*28
    # -conv-> 512*28*28 -conv*-> 512*28*28 -max-> 512*14*14
    # -conv-> 512*14*14 -conv*-> 512*14*14 -max-> 512*7*7
    # -fc-> 4096 -fc-> 4096 -fc-> 1000
    if layers == 11:
        l = [*make_block(3, 64, 1), *make_block(64, 128, 1),
             *make_block(128, 256, 2), *make_block(256, 512, 2),
             *make_block(256, 512, 2)]
    elif layers == 13:
        l = [*make_block(3, 64, 2), *make_block(64, 128, 2),
             *make_block(128, 256, 2), *make_block(256, 512, 2),
             *make_block(256, 512, 2)]
    elif layers == 16:
        l = [*make_block(3, 64, 2), *make_block(64, 128, 2),
             *make_block(128, 256, 3), *make_block(256, 512, 3),
             *make_block(256, 512, 3)]
    elif layers == 19:
        l = [*make_block(3, 64, 2), *make_block(64, 128, 2),
             *make_block(128, 256, 4), *make_block(256, 512, 4),
             *make_block(256, 512, 4)]
    model_t = nn.Sequential(
        *l, nn.Flatten(),
        nn.Linear(7*7*512, 4096), nn.ReLU(),
        nn.Linear(4096, 4096), nn.ReLU(),
        nn.Linear(4096, 1000))
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
        c = sys.argv[1]
        nh = int(sys.argv[2])
        nw = int(sys.argv[3])
        assert nh*nw == net.size
        if c == '1':
            test_case1(nh, nw)
        elif c == '2':
            test_case2(nh, nw)
        elif c == 'alex':
            test_alex(nh, nw)
        elif c.startswith('vgg-'):
            v = int(c[4:])
            assert v in [11, 13, 16, 19]
            test_vgg(nh, nw, v)
        else:
            print(f"test case {c} not supported")
    else:
        if net.rank == 0:
            print("Usage: <case-id> <nh> <nw>")


if __name__ == "__main__":
    main()
