# -*- coding: utf-8 -*-

import numpy as np
import hennlayer
import computil
import heutil

class PhenNetwork():
    def __init__(self, nh, nw, hid, wid):
        self.nh = nh
        self.nw = nw
        self.hid = hid
        self.wid = wid
        
    def forward(self, data:np.ndarray):
        pass
    
    
# %% layers

class PhenLayer():
    def __init__(self, nh, nw, hid, wid):
        self.nh = nh
        self.nw = nw
        self.hid = hid
        self.wid = wid
        # derivated properties
        self.npart = nh*nw # number of parts in total
        self.pid = hid*nw + wid # part id (sequence id)
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        raise NotImplementedError("This function is not implemented")
        
    def in_shape(self):
        """
        Get the expected shape of input data, as a tuple.
        For unfixed input: return None.
        For dimensions with unfixed size: return None on that dimension.
        """
        raise NotImplementedError("This function is not implemented")
        
    def out_shape(self, inshape:tuple):
        """
        Get the expected shape of output data, as a tuple, given data as <inshape>.
        """
        raise NotImplementedError("This function is not implemented")
    

class PhenConv(PhenLayer):
    def __init__(self, nh, nw, hid, wid, conv:hennlayer.Conv2d):
        super().__init__(nh, nw, hid, wid)
        self.conf = computil.Conv2dConf(conv.in_ch, conv.out_ch, conv.kernel_size,
                                        conv.stride, conv.padding, conv.groups)
        self.weight = conv.weight
        self.bias = conv.bias
    
    def forward(self, x:np.ndarray):
        return computil.conv2d(x, self.conf, self.weight, self.bias, False)
    
    def in_shape(self):
        return (self.in_ch, None, None)
    
    def out_shape(self, inshape:tuple):
        assert len(inshape) == 3
        assert inshape[0] == self.conf.out_ch
        return self.conf.comp_out_size(inshape[1], inshape[2])


class PhenLinear(PhenLayer):
    def __init__(self, nh, nw, hid, wid, linear:hennlayer.Linear):
        super().__init__(nh, nw, hid, wid)
        self.in_ch = linear.in_ch
        self.out_ch = linear.out_ch
        self.weight = linear.weight # shape: out_ch * in_ch
        self.bias = linear.bias # shape: out_ch
        # local computation related
        m = self.in_ch / self.npart
        off_f = int(m*self.pid)
        off_l = int(m*(self.pid+1)) if self.pid+1 != self.npart else self.in_ch
        #print(m, off_f, off_l)
        self.local_weight = self.weight[:, off_f:off_l]
        self.local_bias = self.bias if self.pid == 0 else None
        
    def forward(self, x:np.ndarray):
        #print(x, self.local_weight, self.local_bias)
        #print(x.shape, self.local_weight.shape)
        r = heutil.dot_product_21(self.local_weight, x)
        #print(r.shape, None if self.local_bias is None else self.local_bias.shape)
        if self.local_bias is not None:
            r += self.local_bias
        return r

    def in_shape(self):
        return (self.in_ch, )

    def out_shape(self, inshape:tuple):
        assert len(inshape) == 1
        return (self.out_ch, )
        

class PhenFlatten(PhenLayer):
    def __init__(self, nh, nw, hid, wid, ishape):
        super().__init__(nh, nw, hid, wid)
        self.ishape = ishape
        assert ishape.ndim == 3
        dch, dw, dh = ishape
            
    def forward(self, x:np.ndarray):
        return x.reshape((-1))

    def in_shape(self):
        return None

    def out_shape(self, inshape:tuple):
        assert len(inshape) == 1
        return (None, )


class PhenRelu(PhenLayer):
    def __init__(self, nh, nw, hid, wid):
        super().__init__(nh, nw, hid, wid)
    
    def forward(self, x:np.ndarray):
        if x.dtype is not object:
            out = np.maximum(x, 0)
        else:
            shape = x.shape
            out = np.array([heutil.relu(i) for i in x.ravel()]).reshape(shape)
        return out

    def in_shape(self):
        return None

    def out_shape(self, inshape:tuple):
        return inshape


class PhenSquare(PhenLayer):
    def __init__(self, nh, nw, hid, wid):
        super().__init__(nh, nw, hid, wid)
    
    def forward(self, x:np.ndarray):
        if x.dtype is not object:
            out = x*x
        else:
            shape = x.shape
            out = np.array([heutil.square(i) for i in x.ravel()]).reshape(shape)
        return out

    def in_shape(self):
        return None

    def out_shape(self, inshape:tuple):
        return inshape
    

# %% test


def __test__():
    import hennlayer_torch as ht
    import torch
    import mnist_analyze
    
    model = mnist_analyze.Net2()
    model.load_state_dict(torch.load("pretrained/net2.pth"))
    
        
    
    
