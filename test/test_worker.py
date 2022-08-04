# -*- coding: utf-8 -*-
import parallel.phenetwork as pn
import parallel.worker as worker
from parallel.shaper import Shaper, make_shaper

import numpy as np
import hennlayer as hn
import torch
import torch.nn as nn
import hennlayer_torch as hnt

# %% test functions

def simple_test():
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


    data = np.random.random((1,10,10))
    w.run(data)

    w.show_stat()

# %% main

def main():
    simple_test()


if __name__ == "__main__":
    main()
