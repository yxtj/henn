# -*- coding: utf-8 -*-

import numpy as np
import hennlayer
import computil
import heutil

from .shaper import Shaper, make_shaper #Shaper1D, Shaper2D,


# %% layers

class PhenLayer():
    def __init__(self, nh, nw, hid, wid, name=None):
        self.nh = nh
        self.nw = nw
        self.hid = hid
        self.wid = wid
        # derivated properties
        self.npart = nh*nw # number of parts in total
        self.pid = hid*nw + wid # part id (sequence id)
        self.name = name

    def name(self):
        return self.name

    def bind_in_model(self, ishaper:Shaper, oshaper:Shaper,
                      idx:int, gshapes:list[tuple], layer_names:list[str]):
        self.ishaper = ishaper
        self.oshaper = oshaper
        self.layer_idx = idx
        self.gshapes = gshapes
        self.layers = layer_names

    def __call__(self, x:np.ndarray):
        return self.local_forward(x)

    # workflow: depend -> local_prepare -> local_forward -> balance

    # data dependency

    def depend_out(self, x:np.ndarray):
        """
        Get list of (hid, wid, desc) for parts that DEPEND ON the data of this
          part. i.e. Return the parts to which this part SENDs message.
        The type and content of <desc> is determined by each layer.
        """
        return []

    def depend_in(self, x:np.ndarray):
        """
        Get list of (hid, wid, desc) for parts which this part depends on.
          i.e. Return the parts from which this part RECEIVEs messages.
        The type and content of <desc> is determined by each layer.
        """
        return []

    def depend_message(self, x:np.ndarray, tgt_hid:int, tgt_wid:int, desc):
        """
        Return the data shard on which another part depends.
        This function respond to the depend_out() function.
        """
        return (self.hid, self.wid, x)

    def depend_merge(self, xlocal:np.ndarray, xlist:list):
        """
        Merge the local data and dependent data (get by cross_message) for processing.
        The xlist is a list storing the results of cross_message() from other parts.
        """
        return xlocal

    # local computation

    def local_prepare(self, x:np.ndarray):
        """
        Preprocessing of the input data.
        The output of this function is feed to the local_forward() function.
        """
        return x

    def local_forward(self, x:np.ndarray):
        """
        The main function of processing local part.
        Return the local processing reuslt.
        """
        raise NotImplementedError("The local_forward function is not implemented")

    # load balance

    def balance_out(self):
        """
        Get list of (hid, wid) for parts whose data is partly hold by this part.
        i.e. Return the parts to which this part SENDs message.
        """
        return []

    def balance_in(self):
        """
        Get a list of (hid, wid) for parts which hold data of this part.
        i.e. Return the parts from which this part RECEIVE data shards.
        """
        return []

    def balance_message(self, x:np.ndarray, tgt_hid, tgt_wid):
        """
        Return the data cut that should be hold by (tgt_hid, tgt_wid).
        This is used for load-balancing across layers.
        The return type is None or (tgt_hid, tgt_wid, x, desc), where <desc> is
         the additional description for x. If <desc> is not necessary, leave None.
        """
        desc = None
        return (tgt_hid, tgt_wid, x, desc)

    def balance_merge(self, xlocal:np.ndarray, xlist:list):
        """
        Merge and reshape the local data shard and received shards.
        Elements of <xlist> are the return values of balance_merge().
        """
        return xlocal

    # get the global result

    def global_join(self, xmat):
        """
        Merge local results of all parts and return the global result.
        Return the final result of this layer as if there is no parallelization.
        """
        raise NotImplementedError("The global_join function is not implemented")

    # shape related

    def in_shape(self):
        """
        Get the expected shape of global input data, as a tuple.
        For unfixed input: return None.
        For dimensions with unfixed size: return None on that dimension.
        """
        raise NotImplementedError("This function is not implemented")

    def out_shape(self, inshape:tuple):
        """
        Get the expected shape of global output data, as a tuple,
          given data as <inshape>.
        """
        raise NotImplementedError("This function is not implemented")

    def in_shape_local(self):
        self.ishaper.get_shape(self.hid, self.wid)

    def out_shape_local(self):
        self.oshaper.get_shape(self.hid, self.wid)

    # helper function

    def _merge_(self, xlocal, xlist, coord_local=(0,0), mat_shape=(2,2)):
        pass

# %% convolution layer

# TODO: balance data after one conv
# e.g. conv with kernel size 3 for 0-5, with 2 workers:
# input cut: [0, 1, 2], [3, 4, 5]
# fill dependent: [0, 1, 2, 3, 4], [3, 4, 5]
# output: [0, 1, 2], [3]
# inbalance we need to move 2 to worker-2
# expected input for next conv; [0, 1], [2, 3]

class PhenConv(PhenLayer):
    def __init__(self, nh, nw, hid, wid, conv:hennlayer.Conv2d):
        super().__init__(nh, nw, hid, wid, "conv")
        self.conf = computil.Conv2dConf(conv.in_ch, conv.out_ch, conv.kernel_size,
                                        conv.stride, conv.padding, conv.groups)
        self.weight = conv.weight
        self.bias = conv.bias

    def bind_in_model(self, ishaper:Shaper, oshaper:Shaper,
                      idx:int, gshapes:list[tuple], layer_names:list[str]):
        ishaper = make_shaper(self.nh, self.nw, 2, gshapes[idx])
        oshaper = make_shaper(self.nh, self.nw, 2, gshapes[idx+1])
        super().bind_in_model(ishaper, oshaper, idx, gshapes, layer_names)

    def depend_out(self, x:np.ndarray):
        res = []
        _, h, w = x.shape
        hneed = self.conf.kernel_size[0] - self.conf.stride[0]
        wneed = self.conf.kernel_size[1] - self.conf.stride[1]
        if self.wid != 0:
            # right
                res.append((self.hid, self.wid - 1, (h, wneed)))
        if self.hid != 0:
            # lower
            if wneed != 0:
                res.append((self.hid - 1, self.wid, (hneed, w)))
        if self.hid != 0 and self.wid != 0:
            # lower right
            if hneed != 0 and wneed != 0:
                res.append((self.hid - 1, self.wid - 1, (hneed, wneed)))
        return res

    def depend_in(self, x:np.ndarray):
        res = []
        _, h, w = x.shape
        hneed = self.conf.kernel_size[0] - self.conf.stride[0]
        wneed = self.conf.kernel_size[1] - self.conf.stride[1]
        if self.wid != self.nw-1:
            # right
            if hneed != 0:
                res.append((self.hid, self.wid + 1, (h, wneed)))
        if self.hid != self.nh-1:
            # lower
            if wneed != 0:
                res.append((self.hid + 1, self.wid, (hneed, w)))
        if self.hid != self.nh-1 and self.wid != self.nw-1:
            # lower right
            if hneed != 0 and wneed != 0:
                res.append((self.hid + 1, self.wid + 1, (hneed, wneed)))
        return res

    def depend_message(self, x:np.ndarray, tgt_hid:int, tgt_wid:int, desc:tuple):
        assert 0 <= tgt_hid < self.nh and tgt_hid != self.hid
        assert 0 <= tgt_wid < self.nw and tgt_wid != self.wid
        if tgt_hid == self.hid-1 and tgt_wid == self.wid-1:
            # upper left
            return (tgt_hid, tgt_wid, x[:, :desc[0], :desc[1]])
        elif tgt_hid == self.hid and tgt_wid == self.wid-1:
            # left
            assert desc[0] == x.shape[1]
            return (tgt_hid, tgt_wid, x[:, :, :desc[1]])
        elif tgt_hid == self.hid-1 and tgt_wid == self.wid:
            # upper
            assert desc[1] == x.shape[2]
            return (tgt_hid, tgt_wid, x[:, :desc[0], :])
        return None

    def depend_merge(self, xlocal:np.ndarray, xlist:list):
        h = 2 if self.hid != self.nh-1 else 1
        w = 2 if self.wid != self.nw-1 else 1
        xmat = np.empty((h, w), dtype=np.ndarray)
        xmat[0][0] = xlocal
        for hid, wid, xr in xlist:
            offh = hid - self.hid
            offw = wid - self.wid
            xmat[offh, offw] = xr
        xmat = np.ndarray(xmat) # turn into 3D
        res = np.concatenate(
            [ np.concatenate(xmat[i,:],2) for i in range(self.nw) ], 1)
        return res

    def local_prepare(self, x:np.ndarray):
        if self.conf.padding == 0:
            return x
        res = computil.pad_data(x, self.padding, self.wid==0, self.hid==0,
                                self.wid==self.nw-1, self.hid==self.nh-1)
        return res

    def local_forward(self, x:np.ndarray):
        return computil.conv2d(x, self.conf, self.weight, self.bias, False)

    def balance_out(self, x:np.ndarray):
        return []

    def balance_in(self, x:np.ndarray):
        return []

    def balance_message(self, x:np.ndarray, tgt_hid, tgt_wid):
        desc = None
        return (tgt_hid, tgt_wid, x, desc)

    def balance_merge(self, xlocal:np.ndarray, xlist:list):
        raise NotImplementedError("The balance_merge function is not implemented")

    def global_join(self, xmat):
        xmat = np.ndarray(xmat)
        assert xmat.shape == (self.nh, self.nw, self.conf.out_ch)
        res = np.concatenate(
            [ np.concatenate(xmat[i,:],2) for i in range(self.nw) ], 1)
        return res

    def in_shape(self):
        return (self.in_ch, None, None)

    def out_shape(self, inshape:tuple):
        assert len(inshape) == 3
        assert inshape[0] == self.conf.in_ch
        ox, oy = self.conf.comp_out_size(inshape[1], inshape[2])
        return (self.conf.out_ch, ox, oy)

# %% fully connected layer

class PhenLinear(PhenLayer):
    def __init__(self, nh, nw, hid, wid, linear:hennlayer.Linear):
        super().__init__(nh, nw, hid, wid, "linear")
        self.in_ch = linear.in_ch
        self.out_ch = linear.out_ch
        self.weight = linear.weight # shape: out_ch * in_ch
        self.bias = linear.bias # shape: out_ch
        # local computation related
        ishaper = make_shaper(self.nh, self.nw, 1, (self.in_ch,), interleave=True)
        self.ishaper = ishaper
        # set local weight:
        #   assume the previous layer is also a Linear layer
        #self.local_weight = self.weight[:, self.pid::self.npart]
        self.local_weight = self.ishaper.pick_data(hid, wid, self.weight)
        # set local bias:
        #   the pid-th part handles the pid-th channel's bias
        self.local_bias = np.zeros((self.out_ch))
        self.local_bias[self.pid::self.npart] = self.bias[self.pid::self.npart]

    def bind_in_model(self, ishaper:Shaper, oshaper:Shaper,
                      idx:int, gshapes:list[tuple], layer_names:list[str]):
        super().bind_in_model(self.ishaper, oshaper, idx, gshapes, layer_names)
        # find last non-trivial layer
        last_idx = idx - 1
        while last_idx >= 0 and layer_names[last_idx] in ["flatten", "identity"]:
            last_idx -= 1
        # update the local weights for Conv Layer
        if idx != 0 and last_idx >= 0 and layer_names[last_idx] == "conv":
            #assert len(gshapes[last_idx]) == 3
            pshaper = make_shaper(self.nh, self.nw, 2, gshapes[last_idx+1])
            w = self.weight.reshape((self.out_ch, -1))
            self.local_weight = pshaper.pick_data(self.hid, self.wid, w)

    def depend_to(self, x:np.ndarray):
        return []

    def depend_from(self, x:np.ndarray):
        return []

    def depend_message(self, x:np.ndarray, tgt_hid:int, tgt_wid:int, desc):
        return (self.hid, self.wid, x)

    def depend_merge(self, xlocal:np.ndarray, xlist:list):
        return xlocal

    def local_prepare(self, x:np.ndarray):
        return x

    def local_forward(self, x:np.ndarray):
        #print(x, self.local_weight, self.local_bias)
        #print(x.shape, self.local_weight.shape)
        r = heutil.dot_product_21(self.local_weight, x)
        #print(r.shape, None if self.local_bias is None else self.local_bias.shape)
        if self.local_bias is not None:
            r += self.local_bias
        return r

    def global_join(self, xmat):
        xlist = xmat.ravel()
        #assert len(xlist) == self.npart
        out = np.empty(self.out_ch, xlist[0].dtype)
        for i in range(len(xlist)):
            out[i::self.npart] = xlist[i]
        return out

    def in_shape(self):
        return (self.in_ch, )

    def out_shape(self, inshape:tuple):
        assert len(inshape) == 1
        return (self.out_ch, )


# %% flatten layer

class PhenFlatten(PhenLayer):
    def __init__(self, nh, nw, hid, wid, ishape):
        super().__init__(nh, nw, hid, wid, "flatten")
        self.ishape = ishape
        assert ishape.ndim == 3
        dch, dw, dh = ishape

    def local_forward(self, x:np.ndarray):
        return x.reshape((-1))

    def in_shape(self):
        return None

    def out_shape(self, inshape:tuple):
        assert len(inshape) == 1
        return (None, )


# %% activation layers

class PhenRelu(PhenLayer):
    def __init__(self, nh, nw, hid, wid):
        super().__init__(nh, nw, hid, wid, "relu")

    def local_forward(self, x:np.ndarray):
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
        super().__init__(nh, nw, hid, wid, "square")

    def local_forward(self, x:np.ndarray):
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



