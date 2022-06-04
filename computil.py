# -*- coding: utf-8 -*-

import numpy as np
import heutil

# %% convolution related

class Conv2dConf():
    def __init__(self, in_ch, out_ch, kernel_size,
                 stride=1, padding=0, groups=1):
        '''
        in_ch/out_ch: integer, number of input/output channel
        kernel_size: integer or pair of integer, the kernel size in 2D plain
        stride: integer or integer pair, step size of the moving window
        padding: integer or integer pair, step size of the moving window
        groups: integer, number of group (x-channel operation are within a group)
        -------
        in_ch/groups must be an integer. 
        weight must be of size (out_ch, in_ch/groups, kernel_size[0], kernel_size[1]).
        bias must be of size (in_ch/groups)
        '''
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.nelem = kernel_size[0]*kernel_size[1]
        self.factor = 1.0 / self.nelem
        if isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride
        if isinstance(padding, int):
            padding = (padding, padding)
        self.padding = padding
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.groups = groups
        assert in_ch % groups == 0
        self.in_ch_pg = in_ch // groups # in channles per group
        self.out_ch_pg = out_ch // groups
        
    def check(self, weight, bias):
        assert weight.shape == (self.out_ch, self.in_ch_pg, *self.kernel_size)
        if bias:
            assert bias.shape == (self.out_ch)


def conv2d_one(x, ch, i, j, conf:Conv2dConf, weight, bias):
    _, nx, ny = x.shape
    g = ch // conf.out_ch_pg # tell group by out-channel
    ch_f = g * conf.in_ch_pg
    ch_l = ch_f + conf.in_ch_pg
    #print(g, ch_f, ch_l)
    i_f = i * conf.stride[0] - conf.padding[0]
    i_l = i_f + conf.kernel_size[0]
    wi_f = -i_f if i_f < 0 else 0
    wi_l = nx - i_f if i_l > nx else conf.kernel_size[0]
    j_f = j * conf.stride[1] - conf.padding[1]
    j_l = j_f + conf.kernel_size[1]
    wj_f = -j_f if j_f < 0 else 0
    wj_l = ny - j_f if j_l > ny else conf.kernel_size[1]
    #print(g, i_f, i_l, wi_f, wi_l, '-', j_f, j_l, wj_f, wj_l)
    #i_f, j_f = max(0, i_f), max(0, j_f)
    #i_l, j_l = min(nx, i_l), min(ny, j_l)
    cut = x[ch_f:ch_l, max(0, i_f):i_l, max(0, j_f):j_l]
    cut = cut*weight[ch, :, wi_f:wi_l, wj_f:wj_l]
    r = heutil.sum_list(cut.ravel())
    r = cut.sum()
    if bias:
        r += bias[ch]
    return r
    

def conv2d(x, conf:Conv2dConf, weight, bias):
    assert x.ndim == 3
    in_ch, sx, sy = x.shape
    ox, oy = conf.comp_out_size(sx, sy)
    out = np.empty((conf.out_ch, ox, oy), x.dtype)
    for ch in range(conf.out_ch):
        #print('ch',ch)
        for i in range(ox):
            #print('  i',i)
            for j in range(oy):
                #print('    j',j)
                out[ch,i,j] = conv2d_one(x, ch, i, j)
    return out

