# -*- coding: utf-8 -*-
import api
import sys
import time
import numpy as np
import torch.nn as nn

# %% define model

def get_model_alex(conv=True, fc=True):
    assert (conv or fc), 'the output should be non-empty'
    inshape = (3, 227, 227)
    # 3*227*227 -conv(11, 4)-> 96*55*55 -max(3,2)-> 96*27*27
    # -conv(5,1,2)-> 256*27*27 -max(3,2)-> 256*13*13
    # -conv(3,1,1)-> 384*13*13
    # -conv(3,1,1)-> 384*13*13
    # -conv(3,1,1)-> 256*13*13 -max(3,2)-> 256*6*6
    # -fc-> 4096 -fc-> 4096 -fc-> 1000
    if conv:
        l1 = [
            nn.Conv2d(3, 96, 11, 4), nn.ReLU(), nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 96, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(3, 2)
            ]
    if fc:
        l2 = [
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Linear(4096, 1000)
            ]
    if conv and fc:
        model_t = nn.Sequential(*l1, nn.Flatten(), *l2)
    elif conv:
        model_t = nn.Sequential(*l1)
    elif fc:
        model_t = nn.Sequential(*l2)
    # model_t = nn.Sequential(
    #     nn.Conv2d(3, 96, 11, 4), nn.ReLU(), nn.MaxPool2d(3, 2),
    #     nn.Conv2d(256, 96, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(3, 2),
    #     nn.Conv2d(256, 384, 3, 1, 1), nn.ReLU(),
    #     nn.Conv2d(384, 384, 3, 1, 1), nn.ReLU(),
    #     nn.Conv2d(384, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(3, 2),
    #     nn.Flatten(),
    #     nn.Linear(4096, 4096), nn.ReLU(),
    #     nn.Linear(4096, 4096), nn.ReLU(),
    #     nn.Linear(4096, 1000))
    return inshape, model_t


def get_model_vgg(layers=16, conv=True, fc=True):
    assert (conv or fc), 'the output should be non-empty'
    assert layers in [11, 13, 16, 19]
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
    if conv:
        if layers == 11:
            l1 = [*make_block(3, 64, 1), *make_block(64, 128, 1),
                 *make_block(128, 256, 2), *make_block(256, 512, 2),
                 *make_block(512, 512, 2)]
        elif layers == 13:
            l1 = [*make_block(3, 64, 2), *make_block(64, 128, 2),
                 *make_block(128, 256, 2), *make_block(256, 512, 2),
                 *make_block(512, 512, 2)]
        elif layers == 16:
            l1 = [*make_block(3, 64, 2), *make_block(64, 128, 2),
                 *make_block(128, 256, 3), *make_block(256, 512, 3),
                 *make_block(512, 512, 3)]
        elif layers == 19:
            l1 = [*make_block(3, 64, 2), *make_block(64, 128, 2),
                 *make_block(128, 256, 4), *make_block(256, 512, 4),
                 *make_block(512, 512, 4)]
    if fc:
        l2 = [
            nn.Linear(7*7*512, 4096), nn.ReLU(),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Linear(4096, 1000)
            ]
    if conv and fc:
        model_t = nn.Sequential(*l1, nn.Flatten(), *l2)
    elif conv:
        model_t = nn.Sequential(*l1)
    elif fc:
        model_t = nn.Sequential(*l2)
    return inshape, model_t

# %% main

def run_model(inshape, model_t, nh, nw, cmp=False):
    net = api.Network()
    hid, wid = divmod(net.rank, nw)

    model_p = api.make_phen_model(nh, nw, hid, wid, model_t, inshape)
    shapes, ltypes = api.get_phen_model_info(model_p, inshape)
    if net.rank == 0:
        #info = [(i, m.ishaper.gshape, m.ltype) for i, m in enumerate(model_p)]
        print('\n'.join([f'  Layer-{i} {ltypes[i]}: {shapes[i]} -> {shapes[i+1]}'
                         for i in range(len(ltypes))]), flush=True)

    w = api.setup_worker(nh, nw, net, model_p, inshape)

    data = np.random.random(inshape)
    t = time.time()
    r = w.run(data)
    t = time.time() - t

    s = w.get_stat()
    print(s, flush=True)

    tmax = max(net.gather(t, 0))
    if net.rank == 0:
        print("Total Time:", tmax, flush=True)

    if cmp and net.rank == 0:
        g = w.join_result(r) if cmp else None
        import torch
        ot = model_t(torch.from_numpy(data).float().unsqueeze(0))
        diff = np.abs(ot.detach().numpy() - g).mean()
        print("Difference:", diff, flush=True)


if __name__ == '__main__':
    argv = sys.argv
    if len(argv) == 4:
        c = sys.argv[1]
        nh = int(sys.argv[2])
        nw = int(sys.argv[3])
        if c == 'alex':
            inshape, model_t = get_model_alex()
        elif c.startswith('vgg-'):
            v = int(c[4:])
            assert v in [11, 13, 16, 19]
            inshape, model_t = get_model_vgg(v)
        else:
            print(f"model {c} is not supported")
        run_model(inshape, model_t, nh, nw)
    else:
        print("Usage: <model> <nh> <nw>")


