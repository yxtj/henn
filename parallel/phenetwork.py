# -*- coding: utf-8 -*-

import numpy as np
import hennlayer
import computil
import heutil

from .shaper import Shaper, make_shaper #Shaper1D, Shaper2D,

# for debug:
__DEBUG__ = False
import time
TIME_MA = 2.6e-6
TIME_BS = 66e-3

# %% layers

class PhenLayer():
    def __init__(self, nc, nh, nw, cid, hid, wid, ltype=None, name=None, dim=None):
        self.nc = nc # channel
        self.nh = nh # height
        self.nw = nw # width
        self.cid = cid
        self.hid = hid
        self.wid = wid
        # derivated properties
        self.nhw = nh*nw
        self.npart = nc*nh*nw # number of parts in total
        self.sid = cid*self.nhw + hid*self.nw + wid # sequence id
        # layer property
        self.ltype = ltype.lower()
        if dim is not None:
            self.dim = dim
        else:
            if self.ltype == "linear":
                self.dim = 1
            elif self.ltype == "conv":
                self.dim = 2
            elif self.ltype.endswith("-pool"):
                self.dim = 2
            #elif self.ltype.endswith("-pool1d"):
            #    self.dim = 1
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

    def is_edge(self):
        return self.hid == 0 or self.wid == 0 \
            or self.hid == self.nh-1 or self.wid == self.nw-1

    def is_all_channel(self):
        return self.nc == 1

    # workflow: depend -> local_forward -> join

    # data dependency (x is local input)

    def depend_out(self, x:np.ndarray):
        """
        Get list of (cid, hid, wid, desc) for parts that DEPEND ON the data of this
          part. i.e. Return the parts to which this part SENDs message.
        The "desc" attribute is feed to depend_message(), to avoid redundant
          computation. Its type and content may vary among different Layers.
        """
        return []

    def depend_in(self, x:np.ndarray):
        """
        Get list of (cid, hid, wid) for parts which this part depends on.
          i.e. Return the parts from which this part RECEIVEs messages.
        """
        return []

    def depend_message(self, x:np.ndarray, tgt_cid:int, tgt_hid:int, tgt_wid:int, desc=None):
        """
        Return the data shard on which another part depends.
        This function respond to the depend_out() function.
        """
        return x

    def depend_merge(self, xlocal:np.ndarray, xlist:list):
        """
        Merge the local data and dependent data (get by cross_message) for processing.
        The "xlist" is a list of (cid, hid, wid, data), where <data> is the result of
          depend_message() on part (cid, hid, wid).
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
        Get list of (cid, hid, wid) for parts whose data is partly hold by this part.
        i.e. Return the parts to which this part SENDs message.
        The "desc" attribute is feed to join_message(), to avoid redundant
          computation. Its type and content may vary among different Layers.
        """
        return []

    def join_in(self, x:np.ndarray):
        """
        Get a list of (cid, hid, wid) for parts which hold data of this part.
        i.e. Return the parts from which this part RECEIVE data shards.
        """
        return []

    def join_message(self, x:np.ndarray, tgt_cid:int, tgt_hid:int, tgt_wid:int, desc=None):
        """
        Return the data cut that should be hold by (tgt_cid, tgt_hid, tgt_wid).
        This is used for load-balancing across layers.
        """
        return x

    def join_merge(self, xlocal:np.ndarray, xlist:list):
        """
        Merge and reshape the local data shard and received shards.
        The "xlist" is a list of (cid, hid, wid, data), where <data> is the result of
          join_message() on part (cid, hid, wid).
        """
        return xlocal

    # get the global result

    def global_result(self, xmat:np.ndarray):
        """
        Merge local results of all parts and return the global result.
        Return the final result of this layer as if there is no parallelization.
        "xmat" is np.ndarray of size (nh, nw) and its dtype is np.ndarray.
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
          given data as "inshape".
        """
        raise NotImplementedError("This function is not implemented")

    def in_shape_local(self):
        self.ishaper.get_shape(self.cid, self.hid, self.wid)

    def out_shape_local(self):
        self.oshaper.get_shape(self.cid, self.hid, self.wid)


# %% 2d bases

class Phen2DBase(PhenLayer):

    def _calc_depend_hw_(self, cid, hid, wid):
        if hid == self.hid and self.wid == wid:
            lr = self.gi_lr
        else:
            ul = self.ishaper.get_offset(cid, hid, wid)
            s = self.ishaper.get_shape(cid, hid, wid)
            lr = (ul[0] + s[0], ul[1] + s[1])
        last_h = lr[0] - (lr[0] % self.conf.stride[0])
        last_w = lr[1] - (lr[1] % self.conf.stride[1])
        hneed = max(0, last_h + self.conf.kernel_size[0] - 1 - lr[0])
        wneed = max(0, last_w + self.conf.kernel_size[1] - 1 - lr[1])
        return hneed, wneed

    def _calc_expected_in_box_(self, cid, hid, wid):
        if hid == self.hid and wid == self.wid:
            ul = self.gi_ul
            lr = self.gi_lr
        else:
            ul = self.ishaper.get_offset(cid, hid, wid)
            s = self.ishaper.get_shape(cid, hid, wid)
            lr = (ul[0] + s[0], ul[1] + s[1])
        last_h = ((lr[0]-1) // self.conf.stride[0])*self.conf.stride[0]
        last_w = ((lr[1]-1) // self.conf.stride[1])*self.conf.stride[1]
        lower = max(lr[0] - 1, last_h + self.conf.kernel_size[0] - 1) + 1
        right = max(lr[1] - 1, last_w + self.conf.kernel_size[1] - 1) + 1
        lower = min(self.ishaper.gshape[0], lower)
        right = min(self.ishaper.gshape[1], right)
        return (*ul, lower, right)

    def _calc_expected_out_box_(self, cid, hid, wid):
        oul = self.oshaper.get_offset(cid, hid, wid)
        s = self.oshaper.get_shape(cid, hid, wid)
        olr = (oul[0] + s[0], oul[1] + s[1])
        return (*oul, *olr)

    def _calc_computed_out_box_(self, cid, hid, wid):
        itop, ileft, idown, iright = self._calc_expected_in_box_(cid, hid, wid)
        #print(f'w{hid}-{wid}-in', itop, ileft, idown, iright)
        idown -= self.conf.kernel_size[0] - 1
        iright -= self.conf.kernel_size[0] - 1
        oul = self.conf.comp_out_coord(itop, ileft, True, False, (1,1))
        olr = self.conf.comp_out_coord(idown, iright, True, False, (1,1))
        #print(f'w{hid}-{wid}-out',oul, olr)
        assert oul != (None, None) and olr != (None, None), \
            f"p{cid}-{hid}-{wid}: ul {oul}, lr {olr}"
        return (*oul, *olr)

    # depend: copy dependent data

    def depend_out(self, x:np.ndarray):
        box = self.ishaper.get_range(self.cid, self.hid, self.wid)
        offh, offw = box[0], box[1]
        dp_u = max(0, box[0] - self.conf.kernel_size[0] + 1)
        dp_l = max(0, box[1] - self.conf.kernel_size[1] + 1)
        # dp_down and dp_right are inclusive
        #dp_d = max(1, box[2] - self.conf.kernel_size[0])
        #dp_r = max(1, box[3] - self.conf.kernel_size[1])
        c1, h1, w1 = self.ishaper.comp_part((dp_u, dp_l))
        #h2, w2 = self.ishaper.comp_part((dp_d, dp_r))
        h2, w2 = self.hid, self.wid
        res = []
        c = 0
        for h in range(h1, h2+1):
            for w in range(w1, w2+1):
                if h != self.hid or w != self.wid:
                    b = self._calc_expected_in_box_(c, h, w)
                    o = computil.box_overlap(box, b)
                    if o is not None:
                        desc = (o[0]-offh, o[1]-offw, o[2]-offh, o[3]-offw)
                        res.append((c, h, w, desc))
        return res

    def depend_in(self, x:np.ndarray):
        box = self._calc_expected_in_box_(self.cid, self.hid, self.wid)
        overlaps = self.ishaper.comp_covered_parts(box)
        res = []
        c = 0
        for c, h, w, d in overlaps:
            if h != self.hid or w != self.wid:
                b = self.ishaper.get_range(c, h, w)
                desc = computil.box_overlap(box, b)
                res.append((c, h, w, desc))
        return res

    def depend_message(self, x:np.ndarray, tgt_cid:int, tgt_hid:int, tgt_wid:int, desc):
        assert 0 <= tgt_cid < self.nc
        assert 0 <= tgt_hid < self.nh
        assert 0 <= tgt_wid < self.nw
        assert tgt_hid != self.hid or tgt_wid != self.wid
        h1, w1, h2, w2 = desc
        return x[:, h1:h2, w1:w2]

    def depend_merge(self, xlocal:np.ndarray, xlist:list):
        if len(xlist) == 0:
            return xlocal
        # make xmat
        hw = np.array([(h,w) for c, h, w, _ in xlist])
        hmin, wmin = self.hid, self.wid
        hmax, wmax = hw.max(0)
        nh = hmax - hmin + 1
        nw = wmax - wmin + 1
        assert 1 + len(xlist) == nh*nw, \
            "received data does not form a matrix: local:"+str(xlocal.shape)+\
            " remote:"+str([(h,w,d.shape) for h,w,d in xlist])
        xmat = np.empty((nh, nw), dtype=np.ndarray)
        # put local data
        xmat[0,0] = xlocal
        # put remote data
        for c, h, w, data in xlist:
            xmat[h - hmin, w - wmin] = data
        # merge data
        res = np.concatenate(
            [ np.concatenate(xmat[i,:],2) for i in range(nh) ], 1)
        return res

    # join: balance the output

    def join_out(self, x:np.ndarray):
        box = self._calc_computed_out_box_(self.cid, self.hid, self.wid)
        overlaps = self.oshaper.comp_covered_parts(box)
        res = []
        for c, h, w, desc in overlaps:
            if h == self.hid and w == self.wid:
                pass
            else:
                res.append((c, h, w, desc))
        return res

    def join_in(self, x:np.ndarray):
        box = self._calc_expected_out_box_(self.cid, self.hid, self.wid)
        # compute which part holds the box
        up, left = self.conf.comp_in_coord(box[0], box[1])
        down, right = self.conf.comp_in_coord(box[2]-1, box[3]-1)
        c1, h1, w1 = self.ishaper.comp_part((up, left))
        c2, h2, w2 = self.ishaper.comp_part((down, right))
        res = []
        c = 0
        for h in range(h1, h2+1):
            for w in range(w1, w2+1):
                if h != self.hid or w != self.wid:
                    b = self._calc_computed_out_box_(c, h, w)
                    desc = computil.box_overlap(box, b)
                    res.append((c, h, w, desc))
        return res

    def join_message(self, x:np.ndarray, tgt_cid, tgt_hid, tgt_wid, desc):
        itop, ileft, idown, iright = self._calc_expected_in_box_(
            self.cid, self.hid, self.wid)
        offset = self.conf.comp_out_coord(itop, ileft, True, False, (1,1))
        #offset = self.oshaper.get_offset(self.hid, self.wid)
        h1 = desc[0] - offset[0]
        w1 = desc[1] - offset[1]
        h2 = desc[2] - offset[0]
        w2 = desc[3] - offset[1]
        return x[:, h1:h2, w1:w2]

    def join_merge(self, xlocal:np.ndarray, xlist:list):
        box = self._calc_computed_out_box_(self.cid, self.hid, self.wid)
        overlaps = self.oshaper.comp_covered_parts(box)
        olocal = list(filter(lambda chwd: chwd[1]==self.hid and chwd[2]==self.wid,
                             overlaps))
        if len(olocal) == 1:
            h1, w1, h2, w2 = olocal[0][3]
        if len(xlist) == 0:
            offh, offw = box[0], box[1]
            return xlocal[:, h1-offh:h2-offh, w1-offw:w2-offw]
        # make xmat
        hw = np.array([(h,w) for c, h, w, _ in xlist])
        hmin, wmin = hw.min(0)
        hmax, wmax = hw.max(0)
        if len(olocal) != 0:
            nh = max(hmax, self.hid) - min(hmin, self.hid) + 1
            nw = max(wmax, self.wid) - min(wmin, self.wid) + 1
        else:
            nh = hmax - hmin + 1
            nw = wmax - wmin + 1
        assert len(olocal) + len(xlist) == nh*nw, \
            "received data does not form a matrix: local:"+str(olocal)+" remote:"+\
            str([(h,w,d.shape) for h,w,d in xlist])
        xmat = np.empty((nh, nw), dtype=np.ndarray)
        # put remote data
        for c, h, w, data in xlist:
            xmat[h - hmin, w - wmin] = data
        # put local data
        if len(olocal) == 1:
            offh, offw = box[0], box[1]
            d = xlocal[:, h1-offh:h2-offh, w1-offw:w2-offw]
            xmat[self.hid-hmin, self.wid-wmin] = d
        # merge data
        res = np.concatenate(
            [ np.concatenate(xmat[i,:],2) for i in range(nh) ], 1)
        return res

    # scale-adapted repartion



# %% convolution layer

# Need to re-partition data after one conv
# e.g. conv with kernel size 3 for 0-5, with 2 workers:
# input cut: [0, 1, 2], [3, 4, 5]
# fill dependent: [0, 1, 2, 3, 4], [3, 4, 5]
# output: [0, 1, 2], [3]
# inbalance we need to move 2 to worker-2
# expected input for next conv; [0, 1], [2, 3]

class PhenConv(Phen2DBase):
    def __init__(self, nc, nh, nw, cid, hid, wid,
                 conv:hennlayer.Conv2d, name=None):
        super().__init__(nc, nh, nw, cid, hid, wid, "conv", name)
        self.conf = computil.Conv2dConf(conv.in_ch, conv.out_ch, conv.kernel_size,
                                        conv.stride, conv.padding, conv.groups)
        self.weight = conv.weight
        self.bias = conv.bias
        self.crng = np.linspace(0, self.conf.out_ch_pg, nc, dtype=int)

    def bind_in_model(self, ishaper:Shaper, oshaper:Shaper,
                      idx:int, gshapes:list[tuple], layer_types:list[str]):
        assert ishaper.dim() == 2
        assert oshaper.dim() == 2
        super().bind_in_model(ishaper, oshaper, idx, gshapes, layer_types)
        # inputs:
        # global coordinate of the upper left pixel (inclusive)
        inbox = self.ishaper.get_range(self.cid, self.hid, self.wid)
        self.gi_ul = (inbox[0], inbox[1])
        # global coordinate of the lower right pixel (exclusive)
        self.gi_lr = (inbox[2], inbox[3])

    def local_forward(self, x:np.ndarray):
        # padding
        if self.conf.padding != (0, 0):
            x = computil.pad_data(x, self.conf.padding, self.wid==0, self.hid==0,
                                  self.wid==self.nw-1, self.hid==self.nh-1)
        # convolute
        itop, ileft = self.ishaper.get_offset(self.cid, self.hid, self.wid)
        if itop % self.conf.stride[0] == 0:
            h = 0
        else:
            h = self.conf.stride[0] - itop % self.conf.stride[0]
        if ileft % self.conf.stride[0] == 0:
            w = 0
        else:
            w = self.conf.stride[1] - ileft % self.conf.stride[1]
        x = x[:, h:, w:]
        res = computil.conv2d(x, self.conf, self.weight, self.bias, False)
        if __DEBUG__:
            time.sleep(res.size*np.prod(self.conf.kernel_size)*TIME_MA)
        return res

    def global_result(self, xmat:np.ndarray):
        assert xmat.ndim == 2
        assert xmat.shape == (self.nh, self.nw)
        assert xmat[0,0].shape[0] == self.conf.out_ch
        res = np.concatenate(
            [ np.concatenate(xmat[i,:],2) for i in range(self.nh) ], 1)
        return res

    def in_shape(self):
        return (self.in_ch, None, None)

    def out_shape(self, inshape:tuple):
        assert len(inshape) == 3
        assert inshape[0] == self.conf.in_ch
        ox, oy = self.conf.comp_out_size(inshape[1], inshape[2])
        return (self.conf.out_ch, ox, oy)

# %% pooling layer

class PhenAvgPool(PhenLayer):
    def __init__(self, nc, nh, nw, cid, hid, wid, layer:hennlayer.AvgPool2d, name=None):
        super().__init__(nc, nh, nw, cid, hid, wid, 'avg-pool', name)
        self.conf = computil.Pool2dConf(layer.kernel_size, layer.stride,
                                        layer.padding)
        self.factor = 1.0/np.prod(self.conf.kernel_size)

    def local_forward(self, x:np.ndarray):
        # padding
        if self.conf.padding != (0, 0) and self.is_edge():
            x = computil.pad_data(x, self.conf.padding, self.wid==0, self.hid==0,
                                  self.wid==self.nw-1, self.hid==self.nh-1)
        # pooling
        isc, isx, isy = x.shape
        osx, osy = self.conf.comp_out_size(isx, isy, True)
        end_i = isx - self.conf.kernel_size[0] + 1
        end_j = isy - self.conf.kernel_size[1] + 1
        out = np.empty((isc, osx, osy), dtype=x.dtype)
        for c in range(isc):
            oi = 0
            for i in range(0, end_i, self.conf.stride[0]):
                oj = 0
                i2 = i+self.conf.kernel_size[0]
                for j in range(0, end_j, self.conf.stride[1]):
                    d = x[c, i:i2, j:j+self.conf.kernel_size[1]]
                    out[c, oi, oj] = heutil.hesum(d.ravel())*self.factor
                    oj += 1
                oi += 1
        return out

    def global_result(self, xmat:np.ndarray):
        assert xmat.ndim == 2
        assert xmat.shape == (self.nh, self.nw)
        assert xmat[0,0].shape[0] == self.conf.out_ch
        res = np.concatenate(
            [ np.concatenate(xmat[i,:],2) for i in range(self.nh) ], 1)
        return res

    def in_shape(self):
        return (self.in_ch, None, None)

    def out_shape(self, inshape:tuple):
        assert len(inshape) == 3
        ox, oy = self.conf.comp_out_size(inshape[1], inshape[2])
        return (inshape[0], ox, oy)

# %% fully connected layer

class PhenLinear(PhenLayer):
    def __init__(self, nc, nh, nw, cid, hid, wid, linear:hennlayer.Linear, name=None):
        super().__init__(nc, nh, nw, cid, hid, wid, "linear", name)
        self.in_ch = linear.in_ch
        self.out_ch = linear.out_ch
        self.weight = linear.weight # shape: out_ch * in_ch
        self.bias = linear.bias # shape: out_ch
        # local computation related
        ishaper = make_shaper(self.nc, self.nh, self.nw, 1,
                              (self.in_ch,), interleave=True)
        self.ishaper = ishaper
        # set local weight:
        #   assume the previous layer is also a Linear layer
        #self.local_weight = self.weight[:, self.sid::self.npart]
        self.local_weight = self.ishaper.pick_data(cid, hid, wid, self.weight)
        # set local bias:
        #   the pid-th part handles the pid-th channel's bias
        if self.bias is None:
            self.local_bias = None
        else:
            self.local_bias = np.zeros((self.out_ch))
            self.local_bias[self.sid::self.npart] = self.bias[self.sid::self.npart]

    def bind_in_model(self, ishaper:Shaper, oshaper:Shaper,
                      idx:int, gshapes:list[tuple], layer_types:list[str]):
        assert ishaper == self.ishaper, f'expect:{self.ishaper}, get:{ishaper}'
        super().bind_in_model(ishaper, oshaper, idx, gshapes, layer_types)
        # find last non-trivial layer
        last_idx = idx - 1
        while last_idx >= 0 and layer_types[last_idx] in [
                "flatten", "identity", "relu", "square"]:
            last_idx -= 1
        # update the local weights for Conv Layer
        if idx != 0 and last_idx >= 0 and layer_types[last_idx] == "conv":
            #assert len(gshapes[last_idx]) == 3
            poshape= gshapes[last_idx+1]
            pshaper = make_shaper(self.nc, self.nh, self.nw, 2, poshape)
            w = self.weight.reshape((self.out_ch, *poshape))
            lw = pshaper.pick_data(self.cid, self.hid, self.wid, w)
            self.local_weight = lw.reshape(self.out_ch, -1)

    def local_forward(self, x:np.ndarray):
        #print(self.hid, self.wid, x, self.local_weight, self.local_bias)
        #print(self.hid, self.wid, x.shape, self.local_weight.shape)
        r = heutil.dot_product_21(self.local_weight, x)
        #print(r.shape, None if self.local_bias is None else self.local_bias.shape)
        if self.local_bias is not None:
            r += self.local_bias
        if __DEBUG__:
            time.sleep(self.local_weight.size*TIME_MA)
        return r

    def join_out(self, x:np.ndarray):
        c = 0
        l = [(c, *divmod(i, self.nw), None) for i in range(self.npart) if i != self.sid]
        return l

    def join_in(self, x:np.ndarray):
        c = 0
        l = [(c, *divmod(i, self.nw), None) for i in range(self.npart) if i != self.sid]
        return l

    def join_message(self, x:np.ndarray, tgt_cid, tgt_hid, tgt_wid, desc):
        m = self.oshaper.pick_data(tgt_cid, tgt_hid, tgt_wid, x)
        return m

    def join_merge(self, xlocal:np.ndarray, xlist:list):
        dl = self.oshaper.pick_data(self.cid, self.hid, self.wid, xlocal)
        data = [dl] + [d for cid, hid, wid, d in xlist]
        r = heutil.hesum(data)
        return r

    def global_result(self, xmat:np.ndarray):
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
    def __init__(self, nc, nh, nw, cid, hid, wid, name=None):
        super().__init__(nc, nh, nw, cid, hid, wid, "flatten", name)

    def bind_in_model(self, ishaper, oshaper, idx, gshapes, layer_types):
        assert ishaper.dim() >= 2
        assert oshaper.dim() == 1
        super().bind_in_model(ishaper, oshaper, idx, gshapes, layer_types)

    def local_forward(self, x:np.ndarray):
        return x.reshape((-1))

    def global_result(self, xmat:np.ndarray):
        assert self.ishaper.dim() == 2
        out = np.empty(self.oshaper.n, xmat[0, 0].dtype)
        gh, gw = self.ishaper.gshape
        cid = 0
        for hid in range(self.nh):
            for wid in range(self.nw):
                # assume input is 3d (channel, height, width)
                shape = self.ishaper.get_shape(cid, hid, wid)
                h, w = shape
                off = self.ishaper.get_offset(cid, hid, wid)
                x = xmat[hid, wid].reshape(-1, h, w)
                out.reshape(-1, gh, gw)[:, off[0]:off[0]+h, off[1]:off[1]+w] = x
        return out

    def in_shape(self):
        return None

    def out_shape(self, inshape:tuple):
        assert len(inshape) != 1
        o = np.prod(inshape)
        return (o, )

# %% identity layer

class PhenIdentity(PhenLayer):
    def __init__(self, nc, nh, nw, cid, hid, wid, ks, stride, pad, name=None):
        super().__init__(nc, nh, nw, cid, hid, wid, "identity", name)

    def local_forward(self, x:np.ndarray):
        return x

    def global_result(self, xmat:np.ndarray):
        #assert self.ishaper == self.oshaper
        dim = xmat[0, 0].ndim
        assert dim == 1 or dim == 3
        if dim == 1:
            out = np.empty(self.oshaper.gshape, xmat[0, 0].dtype)
        else:
            nch = xmat[0, 0].shape[0]
            out = np.empty((nch, *self.oshaper.gshape), xmat[0, 0].dtype)
        cid = 0
        for hid in range(self.nh):
            for wid in range(self.nw):
                rng = self.oshaper.get_range(cid, hid, wid)
                if dim == 1:
                    out[rng[0]:rng[1]] = xmat[hid, wid]
                else:
                    out[:, rng[0]:rng[2], rng[1]:rng[3]] = xmat[hid, wid]
        return out

    def in_shape(self):
        return None

    def out_shape(self, inshape:tuple):
        return inshape


# %% activation layers

class PhenReLU(PhenLayer):
    def __init__(self, nc, nh, nw, cid, hid, wid, name=None):
        super().__init__(nc, nh, nw, cid, hid, wid, "relu", name)

    def local_forward(self, x:np.ndarray):
        if x.dtype is not object:
            out = np.maximum(x, 0)
        else:
            shape = x.shape
            out = np.array([heutil.relu(i) for i in x.ravel()]).reshape(shape)
        if __DEBUG__:
            time.sleep(out.size*TIME_BS)
        return out

    def global_result(self, xmat:np.ndarray):
        #assert self.ishaper == self.oshaper
        dim = xmat[0, 0].ndim
        assert dim == 1 or dim == 3
        if dim == 1:
            out = np.empty(self.oshaper.gshape, xmat[0, 0].dtype)
        else:
            nch = xmat[0, 0].shape[0]
            out = np.empty((nch, *self.oshaper.gshape), xmat[0, 0].dtype)
        cid = 0
        for hid in range(self.nh):
            for wid in range(self.nw):
                #TODO: use consecutive for 1D-shaper and oshaper.get_range here
                #rng = self.oshaper.get_range(hid, wid)
                if dim == 1:
                    #out[rng[0]:rng[1]] = xmat[hid, wid]
                    m = self.oshaper.get_meta(cid, hid, wid)
                    out[m[0]::m[1]] = xmat[hid, wid]
                else:
                    rng = self.oshaper.get_range(cid, hid, wid)
                    out[:, rng[0]:rng[2], rng[1]:rng[3]] = xmat[hid, wid]
        return out

    def in_shape(self):
        return None

    def out_shape(self, inshape:tuple):
        return inshape


class PhenSquare(PhenLayer):
    def __init__(self, nc, nh, nw, cid, hid, wid, name=None):
        super().__init__(nc, nh, nw, cid, hid, wid, "square", name)

    def local_forward(self, x:np.ndarray):
        if x.dtype is not object:
            out = x*x
        else:
            shape = x.shape
            out = np.array([heutil.square(i) for i in x.ravel()]).reshape(shape)
        if __DEBUG__:
            time.sleep(out.size*TIME_BS)
        return out

    def global_result(self, xmat:np.ndarray):
        #assert self.ishaper == self.oshaper
        dim = xmat[0, 0].ndim
        assert dim == 1 or dim == 3
        if dim == 1:
            out = np.empty(self.oshaper.gshape, xmat[0, 0].dtype)
        else:
            nch = xmat[0, 0].shape[0]
            out = np.empty((nch, *self.oshaper.gshape), xmat[0, 0].dtype)
        cid = 0
        for hid in range(self.nh):
            for wid in range(self.nw):
                rng = self.oshaper.get_range(cid, hid, wid)
                if dim == 1:
                    out[rng[0]:rng[1]] = xmat[hid, wid]
                else:
                    out[:, rng[0]:rng[2], rng[1]:rng[3]] = xmat[hid, wid]
        return out

    def in_shape(self):
        return None

    def out_shape(self, inshape:tuple):
        return inshape

