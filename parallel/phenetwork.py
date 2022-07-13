# -*- coding: utf-8 -*-

import numpy as np
import hennlayer
import computil
import heutil

from .shaper import Shaper, make_shaper #Shaper1D, Shaper2D,


# %% layers

class PhenLayer():
    def __init__(self, nh, nw, hid, wid, ltype=None, name=None, dim=None):
        self.nh = nh
        self.nw = nw
        self.hid = hid
        self.wid = wid
        # derivated properties
        self.npart = nh*nw # number of parts in total
        self.pid = hid*nw + wid # part id (sequence id)
        # layer property
        self.ltype = ltype.lower()
        if dim is not None:
            self.dim = dim
        else:
            if self.ltype == "linear":
                self.dim = 1
            elif self.ltype == "conv":
                self.dim = 2
            elif self.ltype == "flatten":
                self.dim = 2
            elif self.ltype == "identity":
                self.dim = 0
            elif self.ltype == "relu" or self.ltype == "square":
                self.dim = 0
            else:
                self.dim = 0
        self.name = name

    def bind_in_model(self, ishaper:Shaper, oshaper:Shaper,
                      idx:int, gshapes:list[tuple], layer_types:list[str]):
        self.ishaper = ishaper
        self.oshaper = oshaper
        self.layer_idx = idx
        self.gshapes = gshapes
        if self.name is None:
            self.name = str(idx)
        #self.layers = layer_types

    def __call__(self, x:np.ndarray):
        return self.local_forward(x)

    # workflow: depend -> local_forward -> join

    # data dependency (x is local input)

    def depend_out(self, x:np.ndarray):
        """
        Get list of (hid, wid, desc) for parts that DEPEND ON the data of this
          part. i.e. Return the parts to which this part SENDs message.
        The <desc> attribute is feed to depend_message(), to avoid redundant
          computation. Its type and content may vary among different Layers.
        """
        return []

    def depend_in(self, x:np.ndarray):
        """
        Get list of (hid, wid) for parts which this part depends on.
          i.e. Return the parts from which this part RECEIVEs messages.
        """
        return []

    def depend_message(self, x:np.ndarray, tgt_hid:int, tgt_wid:int, desc=None):
        """
        Return the data shard on which another part depends.
        This function respond to the depend_out() function.
        """
        return x

    def depend_merge(self, xlocal:np.ndarray, xlist:list):
        """
        Merge the local data and dependent data (get by cross_message) for processing.
        The xlist is a list storing the results of cross_message() from other parts.
        """
        return xlocal

    # local computation

    def local_forward(self, x:np.ndarray):
        """
        The main function of processing local part.
        Return the local processing reuslt.
        """
        raise NotImplementedError("The local_forward function is not implemented")

    # join the local computation (x is local output)
    # involve: merge partial results, repartition for load-balance

    def join_out(self, x:np.ndarray):
        """
        Get list of (hid, wid) for parts whose data is partly hold by this part.
        i.e. Return the parts to which this part SENDs message.
        The <desc> attribute is feed to join_message(), to avoid redundant
          computation. Its type and content may vary among different Layers.
        """
        return []

    def join_in(self, x:np.ndarray):
        """
        Get a list of (hid, wid) for parts which hold data of this part.
        i.e. Return the parts from which this part RECEIVE data shards.
        """
        return []

    def join_message(self, x:np.ndarray, tgt_hid:int, tgt_wid:int, desc=None):
        """
        Return the data cut that should be hold by (tgt_hid, tgt_wid).
        This is used for load-balancing across layers.
        """
        return x

    def join_merge(self, xlocal:np.ndarray, xlist:list):
        """
        Merge and reshape the local data shard and received shards.
        Elements of <xlist> are the return values of join_merge().
        """
        return xlocal

    # get the global result

    def global_result(self, xmat):
        """
        Merge local results of all parts and return the global result.
        Return the final result of this layer as if there is no parallelization.
        """
        raise NotImplementedError("The global_result function is not implemented")

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
        xmat = np.empty(mat_shape, dtype=np.ndarray)
        xmat[coord_local] = xlocal
        for hid, wid, xr in xlist:
            offh = hid - self.hid
            offw = wid - self.wid
            xmat[coord_local[0] + offh, coord_local[1] + offw] = xr
        xmat = np.ndarray(xmat) # turn into 3D
        res = np.concatenate(
            [ np.concatenate(xmat[i,:],2) for i in range(self.nw) ], 1)
        return res

# %% convolution layer

# Need to re-partition data after one conv
# e.g. conv with kernel size 3 for 0-5, with 2 workers:
# input cut: [0, 1, 2], [3, 4, 5]
# fill dependent: [0, 1, 2, 3, 4], [3, 4, 5]
# output: [0, 1, 2], [3]
# inbalance we need to move 2 to worker-2
# expected input for next conv; [0, 1], [2, 3]

class PhenConv(PhenLayer):
    def __init__(self, nh, nw, hid, wid, conv:hennlayer.Conv2d, name=None):
        super().__init__(nh, nw, hid, wid, "conv", name)
        self.conf = computil.Conv2dConf(conv.in_ch, conv.out_ch, conv.kernel_size,
                                        conv.stride, conv.padding, conv.groups)
        self.weight = conv.weight
        self.bias = conv.bias

    def bind_in_model(self, ishaper:Shaper, oshaper:Shaper,
                      idx:int, gshapes:list[tuple], layer_types:list[str]):
        assert ishaper.dim() == 2
        assert oshaper.dim() == 2
        super().bind_in_model(ishaper, oshaper, idx, gshapes, layer_types)
        # inputs:
        # global coordinate of the upper left pixel (inclusive)
        self.gi_ul = self.ishaper.get_offset(self.hid, self.wid)
        # global coordinate of the lower right pixel (exclusive)
        self.gi_lr = self.gi_ul + self.ishaper.get_shape(self.hid, self.wid)
        # outputs:
        #self.go_ul = self.conf.comp_out_coord(self.gi_ul[0], self.gi_ul[1], True)
        #self.go_lr = self.conf.comp_out_coord(self.gi_lr[0], self.gi_lr[1], True)

    # helpers

    def _calc_depend_hw_(self, hid, wid):
        if hid == self.hid and wid == wid:
            lr = self.gi_lr
        else:
            ul = self.ishaper.get_offset(hid, wid)
            lr = ul + self.ishaper.get_shape(hid, wid)
        last_h = (lr[0] // self.conf.stride[0])*self.conf.stride[0]
        last_w = (lr[1] // self.conf.stride[1])*self.conf.stride[1]
        hneed = max(0, last_h + self.conf.kernel_size[0] - 1 - lr[0])
        wneed = max(0, last_w + self.conf.kernel_size[1] - 1 - lr[1])
        return hneed, wneed

    def _calc_expected_out_box_(self, hid, wid):
        oul = self.oshaper.get_offset(hid, wid)
        olr = oul + self.oshaper.get_shape(hid, wid)
        return (*oul, *olr)

    def _calc_computed_out_box_(self, hid, wid):
        iul = self.ishaper.get_offset(hid, wid)
        ilr = iul + self.ishaper.get_shape(hid, wid)
        oul = self.conf.comp_out_coord(iul[0], iul[1], True)
        olr = self.conf.comp_out_coord(ilr[0], ilr[1], True)
        return (*oul, *olr)

    # depend for Conv: copy dependent data

    def depend_out(self, x:np.ndarray):
        res = []
        _, h, w = x.shape
        # upper
        if self.hid > 0:
            hneed, wneed = self._calc_depend_hw_(self.hid-1, self.wid)
            if hneed > 0:
                res.append(self.hid - 1, self.wid, (hneed, w))
        # left
        if self.wid > 0:
            hneed, wneed = self._calc_depend_hw_(self.hid, self.wid-1)
            if hneed > 0:
                res.append(self.hid, self.wid - 1, (h, wneed))
        # upper left
        if self.hid > 0 and self.wid > 0:
            hneed, wneed = self._calc_depend_hw_(self.hid-1, self.wid-1)
            if hneed > 0 and wneed > 0:
                res.append((self.hid - 1, self.wid - 1, (hneed, wneed)))
        return res

    def depend_in(self, x:np.ndarray):
        res = []
        _, h, w = x.shape
        hneed, wneed = self._calc_depend_hw_(self.hid, self.wid)
        # right
        if self.wid + 1 < self.nw and wneed > 0:
            res.append((self.hid, self.wid + 1, (h, wneed)))
        # lower
        if self.hid + 1 < self.nh-1 and hneed > 0:
            res.append((self.hid + 1, self.wid, (hneed, w)))
        # lower right
        if self.hid +1 < self.nh and self.wid + 1 < self.nw:
            if hneed > 0 and wneed > 0:
                res.append((self.hid + 1, self.wid + 1, (hneed, wneed)))
        return res

    def depend_message(self, x:np.ndarray, tgt_hid:int, tgt_wid:int, desc):
        assert 0 <= tgt_hid < self.nh and tgt_hid != self.hid
        assert 0 <= tgt_wid < self.nw and tgt_wid != self.wid
        hneed, wneed = desc
        return x[:, :hneed, :wneed]
        if tgt_hid == self.hid-1 and tgt_wid == self.wid-1:
            # upper left
            return x[:, :hneed, :wneed]
        elif tgt_hid == self.hid and tgt_wid == self.wid-1:
            # left
            #assert desc[0] == x.shape[1]
            return x[:, :, :wneed]
        elif tgt_hid == self.hid-1 and tgt_wid == self.wid:
            # upper
            #assert desc[1] == x.shape[2]
            return x[:, :hneed, :]
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

    def local_forward(self, x:np.ndarray):
        # padding
        if self.conf.padding != 0:
            x = computil.pad_data(x, self.conf.padding, self.wid==0, self.hid==0,
                                  self.wid==self.nw-1, self.hid==self.nh-1)
        # convolute
        return computil.conv2d(x, self.conf, self.weight, self.bias, False)

    # join of Conv: balance the output

    def join_out(self, x:np.ndarray):
        expected = self._calc_expected_out_box_(self.hid, self.wid)
        computed = self._calc_computed_out_box_(self.hid, self.wid)
        res = []
        eup, elf, edw, ert = expected
        cup, clf, cdw, crt = computed
        # up - 3
        if cup < eup:
            up_n = eup - cup
            # up
            res.append((self.hid-1, self.wid, (up_n, min(crt, ert) - max(clf, elf))))
            # up left
            if clf < elf:
                res.append((self.hid-1, self.wid-1, (up_n, elf - clf)))
            # up right
            if crt > ert:
                res.append((self.hid-1, self.wid+1, (up_n, crt - ert)))
        # left
        if clf < elf:
            res.append((self.hid, self.wid-1, (cdw - max(cup, eup), elf - clf)))
        # right
        if crt > ert:
            res.append((self.hid, self.wid+1, (cdw - max(cup, eup), crt - ert)))
        # down - 3
        if cdw > edw:
            dw_n = cdw - edw
            # down
            res.append((self.hid+1, self.wid, (dw_n, min(crt, ert) - max(clf, elf))))
            # down left
            if clf < elf:
                res.append((self.hid+1, self.wid-1, (dw_n, elf - clf)))
            # down right
            if crt > ert:
                res.append((self.hid+1, self.wid+1, (dw_n, crt - ert)))
        return res

    def join_in(self, x:np.ndarray):
        expected = self._calc_expected_out_box_(self.hid, self.wid)
        computed = self._calc_computed_out_box_(self.hid, self.wid)
        res = []
        eup, elf, edw, ert = expected
        cup, clf, cdw, crt = computed
        _, h, w = x.shape
        # up - 3
        if cup > eup:
            up_n = cup - eup
            res.append((self.hid-1, self.wid, (up_n, w)))
            if clf > elf:
                res.append((self.hid-1, self.wid-1, (up_n, clf - elf)))
            if crt < ert:
                res.append((self.hid-1, self.wid+1, (up_n, ert - crt)))
        # left
        if clf > elf:
            res.append((self.hid, self.wid-1, (h, clf - elf)))
        # right
        if crt < ert:
            res.append((self.hid, self.wid+1, (h, ert - crt)))
        # down - 3
        if cdw < edw:
            dw_n = edw - cdw
            res.append((self.hid+1, self.wid, (dw_n, w)))
            if clf > elf:
                res.append((self.hid+1, self.wid-1, (dw_n, clf - elf)))
            if crt < ert:
                res.append((self.hid+1, self.wid+1, (dw_n, ert - crt)))
        return res

    def join_message(self, x:np.ndarray, tgt_hid, tgt_wid, desc):
        h, w = desc
        return x[:, :h, :w]

    def join_merge(self, xlocal:np.ndarray, xlist:list):
        return xlocal

    def global_result(self, xmat):
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
    def __init__(self, nh, nw, hid, wid, linear:hennlayer.Linear, name=None):
        super().__init__(nh, nw, hid, wid, "linear", name)
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
        #self.jshaper = make_shaper(self.nh, self.nw, 1, (self.out_ch,), interleave=True)
        if self.bias is None:
            self.local_bias = None
        else:
            self.local_bias = np.zeros((self.out_ch))
            self.local_bias[self.pid::self.npart] = self.bias[self.pid::self.npart]

    def bind_in_model(self, ishaper:Shaper, oshaper:Shaper,
                      idx:int, gshapes:list[tuple], layer_types:list[str]):
        assert ishaper == self.ishaper
        super().bind_in_model(ishaper, oshaper, idx, gshapes, layer_types)
        # find last non-trivial layer
        last_idx = idx - 1
        while last_idx >= 0 and layer_types[last_idx] in ["flatten", "identity"]:
            last_idx -= 1
        # update the local weights for Conv Layer
        if idx != 0 and last_idx >= 0 and layer_types[last_idx] == "conv":
            #assert len(gshapes[last_idx]) == 3
            poshape= gshapes[last_idx+1]
            pshaper = make_shaper(self.nh, self.nw, 2, poshape)
            w = self.weight.reshape(poshape)
            self.local_weight = pshaper.pick_data(self.hid, self.wid, w).ravel()

    def local_forward(self, x:np.ndarray):
        #print(x, self.local_weight, self.local_bias)
        #print(x.shape, self.local_weight.shape)
        r = heutil.dot_product_21(self.local_weight, x)
        #print(r.shape, None if self.local_bias is None else self.local_bias.shape)
        if self.local_bias is not None:
            r += self.local_bias
        return r

    def join_out(self, x:np.ndarray):
        return [(*divmod(i, self.nw), None) for i in range(self.npart)]

    def join_in(self, x:np.ndarray):
        return [divmod(i, self.nw) for i in range(self.npart)]

    def join_message(self, x:np.ndarray, tgt_hid, tgt_wid, desc):
        m = self.oshaper.pick_data(tgt_hid, tgt_wid, x)
        return m

    def join_merge(self, xlocal:np.ndarray, xlist:list):
        r = heutil.hesum(xlist)
        return r

    def global_result(self, xmat):
        # interleaving based
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
    def __init__(self, nh, nw, hid, wid, name=None):
        super().__init__(nh, nw, hid, wid, "flatten", name)

    def local_forward(self, x:np.ndarray):
        return x.reshape((-1))

    def in_shape(self):
        return None

    def out_shape(self, inshape:tuple):
        assert len(inshape) == 1
        o = np.prod(inshape)
        return (o, )


# %% identity layer

class PhenIdentity(PhenLayer):
    def __init__(self, nh, nw, hid, wid, name=None):
        super().__init__(nh, nw, hid, wid, "identity", name)

    def local_forward(self, x:np.ndarray):
        return x

    def in_shape(self):
        return None

    def out_shape(self, inshape:tuple):
        return inshape


# %% activation layers

class PhenRelu(PhenLayer):
    def __init__(self, nh, nw, hid, wid, name=None):
        super().__init__(nh, nw, hid, wid, "relu", name)

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
    def __init__(self, nh, nw, hid, wid, name=None):
        super().__init__(nh, nw, hid, wid, "square", name)

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



