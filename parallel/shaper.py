# -*- coding: utf-8 -*-

import numpy as np


class Shaper:
    def __init__(self, nh:int, nw:int, data_shape:tuple):
        self.nh = nh
        self.nw = nw
        self.npart = nh * nw
        self.gshape = data_shape

    def dim(self):
        return len(self.gshape)

    def __eq__(self, other):
        return type(self) == type(other) and \
            (self.nh,self.nw,self.gshape) == (other.nh,other.nw,other.gshape)

    def get_shape(self, hid, wid):
        raise NotImplementedError("method get_shape is not implemented")

    def get_meta(self, hid, wid):
        raise NotImplementedError("method get_meta is not implemented")

    def pick_data(self, hid, wid, data:np.ndarray):
        raise NotImplementedError("method pick_data is not implemented")

    def get_offset(self, hid, wid):
        raise NotImplementedError("method get_offset is not implemented")

    def get_indexes(self, hid, wid):
        raise NotImplementedError("method get_indexes is not implemented")

    def comp_part(self, coord:tuple):
        raise NotImplementedError("method comp_part is not implemented")


def make_shaper(nh:int, nw:int, dim:int, data_shape:tuple, **kwargs):
    assert isinstance(data_shape, tuple)
    if dim == 1:
        if 'interleave' in kwargs and kwargs['interleave'] is True:
            return Shaper1D_interleave(nh, nw, data_shape[-1:])
        elif 'consecutive' in kwargs and kwargs['consecutive'] is True:
            return Shaper1D_consecutive(nh, nw, data_shape[-1:])
        else:
            return Shaper1D_interleave(nh, nw, data_shape[-1:])
    elif dim == 2:
        return Shaper2D(nh, nw, data_shape[-2:])
    raise ValueError(f"shaper of {dim} dimension is not supported")

# %% 1d shaper

class Shaper1D_consecutive(Shaper):
    def __init__(self, nh:int, nw:int, data_shape:tuple):
        assert len(data_shape) == 1
        super().__init__(nh, nw, data_shape)
        n = data_shape[0]
        self.n = n
        self.ind = np.linspace(0, n, self.npart+1, dtype=int)

    def get_shape(self, hid, wid):
        sid = hid*self.nw + wid
        i1, i2 = self.ind[sid], self.ind[sid+1]
        return i2-i1

    def get_meta(self, hid, wid):
        sid = hid*self.nw + wid
        i1, i2 = self.ind[sid], self.ind[sid+1]
        return i1, i2

    def pick_data(self, hid, wid, data:np.ndarray):
        assert data.ndim >= 1
        sid = hid*self.nw + wid
        i1, i2 = self.ind[sid], self.ind[sid+1]
        return data[..., i1:i2]

    def get_offset(self, hid, wid):
        sid = hid*self.nw + wid
        return self.ind[sid]

    def get_indexes(self, hid, wid):
        sid = hid*self.nw + wid
        i1, i2 = self.ind[sid], self.ind[sid+1]
        return np.arange(i1, i2)

    def comp_part(self, coord):
        if isinstance(coord, int):
            sid = coord
        elif isinstance(coord, tuple) and len(coord) == 1:
            sid = coord[0]
        else:
            raise NotImplementedError("type of coord is not supported")
        assert sid < self.n
        p = sid // self.npart
        if self.ind[p] <= sid:
            return p
        else:
            return p-1



class Shaper1D_interleave(Shaper):
    def __init__(self, nh:int, nw:int, data_shape:tuple):
        assert len(data_shape) == 1
        super().__init__(nh, nw, data_shape)
        n = data_shape[0]
        self.n = n

    def get_shape(self, hid, wid):
        sid = hid*self.nw + wid
        return (self.n - sid + self.npart - 1) // self.npart

    def get_meta(self, hid, wid):
        sid = hid*self.nw + wid
        return sid, self.npart

    def pick_data(self, hid, wid, data:np.ndarray):
        assert data.ndim >= 1
        sid = hid*self.nw + wid
        return data[..., sid::self.npart]

    def get_offset(self, hid, wid):
        sid = hid*self.nw + wid
        return self.ind[sid]

    def get_indexes(self, hid, wid):
        sid = hid*self.nw + wid
        return np.arange(sid, self.n, self.npart)

    def comp_part(self, coord):
        if isinstance(coord, int):
            sid = coord
        elif isinstance(coord, tuple) and len(coord) == 1:
            sid = coord[0]
        else:
            raise NotImplementedError("type of coord is not supported")
        assert sid < self.n
        p = sid % self.npart
        return p

Shaper1D = Shaper1D_consecutive

# %% 2d shaper

class Shaper2D(Shaper):
    def __init__(self, nh:int, nw:int, data_shape:tuple):
        assert len(data_shape) == 2
        super().__init__(nh, nw, data_shape)
        datah, dataw = data_shape
        self.indh = np.linspace(0, datah, self.nh+1, dtype=int)
        self.indw = np.linspace(0, dataw, self.nw+1, dtype=int)

    def get_shape(self, hid, wid):
        h1, h2 = self.indh[hid], self.indh[hid+1]
        w1, w2 = self.indw[wid], self.indw[wid+1]
        return h2-h1, w2-w1

    def get_meta(self, hid, wid):
        h1, h2 = self.indh[hid], self.indh[hid+1]
        w1, w2 = self.indw[wid], self.indw[wid+1]
        return h1, h2, w1, w2

    def pick_data(self, hid, wid, data:np.ndarray):
        assert data.ndim >= 2
        h1, h2, w1, w2 = self.get_meta(hid, wid)
        return data[..., h1:h2, w1:w2]

    def get_offset(self, hid, wid):
        return (self.indh[hid], self.indw[wid])

    def get_indexes(self, hid, wid):
        h1, h2, w1, w2 = self.get_meta(hid, wid)
        h = h2 - h1
        w = w2 - w1
        off = np.zeros((h, w), dtype=int)
        for i in range(h):
            off[i,:] = (h1+i)*self.nw + np.arange(w1, w2)
        return off

    def comp_part(self, coord:tuple):
        if isinstance(coord, tuple) and len(coord) == 2:
            h, w = coord
        else:
            raise NotImplementedError("type of coord is not supported")
        hid = h // self.nh
        wid = w // self.nw
        if self.indh[hid] > h:
            hid -= 1
        if self.indw[wid] > w:
            wid -= 1
        return hid, wid


