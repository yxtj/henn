# -*- coding: utf-8 -*-
import parallel.phenetwork as pn
import parallel.worker as worker
from network import Network

import numpy as np
import torch
import torch.nn as nn
import hennlayer_torch as hnt

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
    r = w.run(data)
    w.show_stat()

    g = w.join_result(r)
    if net.rank == 0:
        ot = model_t(torch.from_numpy(data).float().unsqueeze(0))
        diff = np.abs(ot.detach().numpy() - g).mean()
        print("difference:", diff)


# %% general test



# %% main

def main():
    torch.manual_seed(123)
    np.random.seed(1)

    #simple_test_1()
    simple_test_2()


if __name__ == "__main__":
    main()
