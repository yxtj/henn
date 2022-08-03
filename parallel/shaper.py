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

    def __repr__(self):
        return f"Shaper of {self.nh}x{self.nw} on shape {self.gshape}"

    def get_shape(self, hid, wid):
        raise NotImplementedError("method get_shape is not implemented")

    def get_meta(self, hid, wid):
        raise NotImplementedError("method get_meta is not implemented")

    def get_offset(self, hid, wid):
        raise NotImplementedError("method get_offset is not implemented")

    def get_range(self, hid, wid):
        raise NotImplementedError("method get_range is not implemented")

    def get_indexes(self, hid, wid):
        raise NotImplementedError("method get_indexes is not implemented")

    def pick_data(self, hid, wid, data:np.ndarray):
        raise NotImplementedError("method pick_data is not implemented")

    def comp_part(self, coord:tuple):
        raise NotImplementedError("method comp_part is not implemented")

    def comp_covered_parts(self, rng:tuple):
        """
        Compute the parts covered by the input range <rng>.
        "rng" is the range of all dimensions. Ranges are in L-close-R-open form.
            [d1_f, d2_f, ..., dk_f, d1_l, d2_l, ..., dk_l]
        Returns list of parts covered by <rng> and the covered ranges:
            (hid, wid, (d1_f, ..., dk_f, d1_l, ..., dk_l))
        """
        raise NotImplementedError("method comp_covered_parts is not implemented")

    def comp_partly_convered_parts(self, rng:tuple):
        """
        Compute the parts that are partly converd by the input range.
        The input and return type are the same as comp_convered_parts
        """
        raise NotImplementedError("method comp_partly_convered_parts is not implemented")

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

    def __repr__(self):
        return super().__repr__()+" 1D-consecutive"

    def get_shape(self, hid, wid):
        sid = hid*self.nw + wid
        i1, i2 = self.ind[sid], self.ind[sid+1]
        return i2-i1

    def get_meta(self, hid, wid):
        sid = hid*self.nw + wid
        i1, i2 = self.ind[sid], self.ind[sid+1]
        return i1, i2

    def get_offset(self, hid, wid):
        sid = hid*self.nw + wid
        return self.ind[sid]

    def get_range(self, hid, wid):
        sid = hid*self.nw + wid
        i1, i2 = self.ind[sid], self.ind[sid+1]
        return (i1, i2)

    def get_indexes(self, hid, wid):
        sid = hid*self.nw + wid
        i1, i2 = self.ind[sid], self.ind[sid+1]
        return np.arange(i1, i2)

    def pick_data(self, hid, wid, data:np.ndarray):
        assert data.ndim >= 1
        sid = hid*self.nw + wid
        i1, i2 = self.ind[sid], self.ind[sid+1]
        return data[..., i1:i2]

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

    def comp_covered_parts(self, rng:tuple):
        # right: a[i-1] <= v < a[i]
        s1 = np.searchsorted(self.ind, rng[0], 'right') - 1
        # left:  a[i-1] < v <= a[i]
        s2 = np.searchsorted(self.ind, rng[1], 'left')
        res = []
        for s in range(s1, s2):
            first = max(rng[0], self.ind[s])
            last = min(rng[1], self.ind[s+1])
            h, w = divmod(s, self.nw)
            res.append((h, w, (first, last)))
        return res

    def comp_partly_covered_parts(self, rng:tuple):
        s1 = np.searchsorted(self.ind, rng[0], 'right') - 1
        s2 = np.searchsorted(self.ind, rng[1], 'left')
        res = []
        if self.ind[s1] != rng[0]:
            h, w = divmod(s1, self.nw)
            res.append((h, w, (rng[0])))
        if self.ind[s2] != rng[1]:
            h, w = divmod(s1, self.nw)
            res.append((h, w, (rng[1])))
        return res


class Shaper1D_interleave(Shaper):
    def __init__(self, nh:int, nw:int, data_shape:tuple):
        assert len(data_shape) == 1
        super().__init__(nh, nw, data_shape)
        n = data_shape[0]
        self.n = n

    def __repr__(self):
        return super().__repr__()+" 1D-interleave"

    def get_shape(self, hid, wid):
        sid = hid*self.nw + wid
        return (self.n - sid + self.npart - 1) // self.npart

    def get_meta(self, hid, wid):
        sid = hid*self.nw + wid
        return sid, self.npart

    def get_offset(self, hid, wid):
        sid = hid*self.nw + wid
        return self.ind[sid]

    def get_indexes(self, hid, wid):
        sid = hid*self.nw + wid
        return np.arange(sid, self.n, self.npart)

    def pick_data(self, hid, wid, data:np.ndarray):
        assert data.ndim >= 1
        sid = hid*self.nw + wid
        return data[..., sid::self.npart]

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

    def get_offset(self, hid, wid):
        return (self.indh[hid], self.indw[wid])

    def get_range(self, hid, wid):
        h1, h2 = self.indh[hid], self.indh[hid+1]
        w1, w2 = self.indw[wid], self.indw[wid+1]
        return (h1, w1, h2, w2)

    def get_indexes(self, hid, wid):
        h1, h2, w1, w2 = self.get_meta(hid, wid)
        h = h2 - h1
        w = w2 - w1
        off = np.zeros((h, w), dtype=int)
        for i in range(h):
            off[i,:] = (h1+i)*self.nw + np.arange(w1, w2)
        return off

    def pick_data(self, hid, wid, data:np.ndarray):
        assert data.ndim >= 2
        h1, h2, w1, w2 = self.get_meta(hid, wid)
        return data[..., h1:h2, w1:w2]

    def comp_part(self, coord:tuple):
        assert isinstance(coord, tuple) and len(coord) == 2, "<coord> should be a pair"
        h, w = coord
        hid = np.searchsorted(self.indh, h, 'right') - 1
        wid = np.searchsorted(self.indw, w, 'right') - 1
        return hid, wid
        #ph = self.indh[-1]/self.nh
        #pw = self.indw[-1]/self.nw
        #hid = h / ph
        #wid = w / pw
        #if self.indh[hid+1] == h:
        #    hid += 1
        #if self.indw[wid+1] == w:
        #    wid += 1
        #return hid, wid

    def comp_covered_parts(self, box:tuple):
        h1, h2 = np.searchsorted(self.indh, (box[0], box[2]-1), 'right') - 1
        w1, w2 = np.searchsorted(self.indw, (box[1], box[3]-1), 'right') - 1
        res = []
        for h in range(h1, h2+1):
            th1 = self.indh[h]
            th2 = self.indh[h+1]
            top = max(box[0], th1)
            bottom = min(box[2], th2)
            for w in range(w1, w2+1):
                tw1 = self.indw[w]
                tw2 = self.indw[w+1]
                left = max(box[1], tw1)
                right = min(box[3], tw2)
                if top<bottom and left<right:
                    res.append((h, w, (top, left, bottom, right)))
        return res

    def comp_partly_covered_parts(self, box:tuple):
        h1 = np.searchsorted(self.indh, box[0], 'right') - 1
        w1 = np.searchsorted(self.indw, box[1], 'right') - 1
        h2 = np.searchsorted(self.indh, box[2], 'left')
        w2 = np.searchsorted(self.indw, box[3], 'left')
        f_top = self.indh[h1] != box[0]
        f_left = self.indw[w1] != box[1]
        f_bottom = self.indh[h2] != box[2]
        f_right = self.indw[w2] != box[3]
        res = []
        # top line
        if f_top:
            th1 = box[0]
            th2 = min(box[2], self.indh[h1+1])
            for w in range(w1, w2):
                tw1 = max(box[1], self.indw[w])
                tw2 = min(box[3], self.indw[w+1])
                res.append((h1, w, (th1, tw1, th2, tw2)))
        # middle lines
        if f_left or f_right:
            if f_left:
                tw1_l = max(box[1], self.indw[w1])
                tw2_l = min(box[3], self.indw[w1+1])
            if f_right:
                tw1_r = max(box[1], self.indw[w2-1])
                tw2_r = min(box[3], self.indw[w2])
            for h in range(h1+(1 if f_top else 0), h2-(1 if f_bottom else 0)):
                th1 = max(box[0], self.indh[h])
                th2 = min(box[2], self.indh[h+1])
                if f_left:
                    res.append((h, w1, (th1, tw1_l, th2, tw2_l)))
                if f_right:
                    res.append((h, w2-1, (th1, tw1_r, th2, tw2_r)))
        # bottom line
        if f_bottom:
            th1 = max(box[0], self.indh[h2-1])
            th2 = box[2]
            for w in range(w1, w2):
                tw1 = max(box[1], self.indw[w])
                tw2 = min(box[3], self.indw[w+1])
                res.append((h2-1, w, (th1, tw1, th2, tw2)))
        return res
