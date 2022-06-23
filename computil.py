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
        self.nelem = in_ch*kernel_size[0]*kernel_size[1]
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
    
    def comp_out_size(self, sx:int, sy:int, padded:bool=False):
        # <padded> means whether <sx> and <sy> already considered the padding pixels
        px = 0 if padded else 2*self.padding[0]
        py = 0 if padded else 2*self.padding[1]
        ox = (sx+px-self.kernel_size[0]+1) // self.stride[0]
        oy = (sy+py-self.kernel_size[1]+1) // self.stride[1]
        #if self.ceil_mode:
        #    ox = int(np.ceil(ox))
        #    oy = int(np.ceil(oy))
        #else:
        #    ox = int(np.floor(ox))
        #    oy = int(np.floor(oy))
        return ox, oy


def conv2d(x, conf:Conv2dConf, weight, bias):
    #return conv2d_v1(x, conf, weight, bias)
    return conv2d_v2(x, conf, weight, bias)

# %% convolution version 1

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
    r = heutil.hesum(cut.ravel())
    r = cut.sum()
    if bias is not None:
        r += bias[ch]
    return r


def conv2d_v1(x, conf:Conv2dConf, weight, bias):
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
                out[ch,i,j] = conv2d_one(x, ch, i, j, conf, weight, bias)
    return out


# %% convolution version 2

def pad_data(x, padding:(int,tuple), left=True, up=True, right=True, down=True):
    if padding is None or (isinstance(padding, int) and padding==0) or \
        (isinstance(padding, (tuple, list)) and padding[0]==0 and padding[1]==0):
        return x
    if isinstance(padding, int):
        px, py = padding, padding
    else:
        px, py = padding
    nc, nx, ny = x.shape
    newx, newy = nx, ny
    offx, offy = 0, 0
    if left:
        newx += px
        offx = px
    if up:
        newy += py
        offy = py
    if right:
        newx += px
    if down:
        newy += py
    if np.issubdtype(x.dtype, np.number):
        data = np.zero((nc, newx, newy), x.dtype)
    else:
        data = np.empty((nc, newx, newy), x.dtype)
    data[:, offx:offx+nx, offy:offy+ny] = x
    # pad 0 for HE case
    if not np.issubdtype(x.dtype, np.number):
        if left:
            data[:,:px] = 0
        if right:
            data[:,-px:] = 0
        if up:
            data[:,:,:py] = 0
        if down:
            data[:,:,-py:] = 0
    return data
        

def conv2d_channel(x, ch, conf:Conv2dConf, weight, bias, out=None):
    # No padding is considered inside this function.
    # <x> is considered already padded.
    _, nx, ny = x.shape
    if conf.groups != 1:
        g = ch // conf.out_ch_pg # tell group by out-channel
        ch_f = g * conf.in_ch_pg
        ch_l = ch_f + conf.in_ch_pg
        data = x[ch_f:ch_l]
    else:
        data = x
    #print(g, ch_f, ch_l)
    ksx, ksy = conf.kernel_size
    if out is None:
        ox, oy = conf.comp_out_size(nx, ny, True)
        out = np.empty((1, ox, oy), x.dtype)
    
    for i in range(0, nx - ksx + 1, conf.stride[0]):
        for j in range(0, ny - ksy + 1, conf.stride[1]):
            cut = data[:, i:i+ksx, j:j+ksy]
            o = heutil.hewsum(cut.ravel(), weight[ch].ravel(), bias[ch])
            out[i, j] = o
    return out


def conv2d_v2(x, conf:Conv2dConf, weight, bias):
    assert x.ndim == 3
    in_ch, sx, sy = x.shape
    x = pad_data(x, conf.padding)
    ox, oy = conf.comp_out_size(sx, sy, True)
    out = np.empty((conf.out_ch, ox, oy), x.dtype)
    for ch in range(conf.out_ch):
        conv2d_channel(x, ch, conf, weight, bias, out[ch])
    #out = [conv2d_channel(x, ch, conf, weight, bias) for ch in range(conf.out_ch)]
    #out = np.concatenate(out)
    return out
