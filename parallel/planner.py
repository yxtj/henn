# -*- coding: utf-8 -*-

import numpy as np
import computil
from .phenetwork import PhenLayer, PhenConv, PhenLinear, PhenFlatten, PhenRelu, PhenSquare

class Planner:
    def __init__(self, nh:int, nw:int, model:list[PhenLayer], inshape:tuple):
        self.nh = nh
        self.nw = nw
        self.npart = nh*nw
        self.model = model
        self.gshapes = self.comp_global_size(model, inshape)
        self.lshape_imp = self.comp_local_size_meta(model, inshape)
        self.lshapes = self.comp_local_size(model, inshape)
        
    def comp_global_size(self, model:list[PhenLayer], inshape:tuple):
        shapes = []
        s = inshape
        shapes.append(s)
        for layer in model:
            s = layer.out_shape(s)
            shapes.append(s)
        return shapes

    def comp_local_size_meta(self, model:list[PhenLayer], gshapes:list):
        res = []
        for i, layer in enumerate(model):
            s = self.gshapes[i]
            if isinstance(layer, PhenConv):
                r = self.comp_conv_local_shape_meta(layer.conf, s)
            elif isinstance(layer, PhenLinear):
                r = self.comp_linear_local_shape_meta(s)
            elif isinstance(layer, PhenFlatten):
                pass
            elif isinstance(layer, PhenRelu) or isinstance(layer, PhenSquare):
                # keep the shape of last layer
                pass
            else:
                print(f"{i}-th layer is not supported:", type(layer))
                pass
            res.append(r)
        return res
    
    def get_local_shape_meta(self, idx):
        return self.lshape_imp[idx]
    
    def comp_conv_local_shape_meta(self, conf:computil.Conv2dConf, inshape:tuple):
        nch, szh, szw = inshape
        psx = szh + 2*conf.padding[0]
        psy = szw + 2*conf.padding[1]
        indh = np.linspace(0, psx, self.nh+1, dtype=int)
        indw = np.linspace(0, psy, self.nw+1, dtype=int)
        # guarantees that indh[i] is the first position with an output
        if conf.stride != 1:
            for i in range(self.nh):
                q, r = divmod(indh[i], conf.stride)
                if r != 0:
                    indh[i] = (q+1)*conf.stride
            for i in range(self.nw):
                q, r = divmod(indw[i], conf.stride)
                if r != 0:
                    indw[i] = (q+1)*conf.stride
        return 'conv', indh, indw
    
    def comp_conv_local_shape(self, meta):
        t, indh, indw = meta
        assert t == 'conv'
        res = np.zeros((self.nh, self.nw, 4), dtype=int)
        for i in range(self.nh):
            for j in range(self.nw):
                res[i,j] = (indh[i+1] - indh[i], indw[j+1] - indw[j])
        return res
    
    def comp_linear_local_shape_meta(self, inshape:tuple):
        if len(inshape) == 3:
            # the previous layer is Conv
            nch, szh, szw = inshape
            
        elif len(inshape) == 1:
            # the previous layer is Linear
            nch = inshape[0]
            res = np.linspace(0, nch, self.npart+1, dtype=int)
        else:
            raise ArithmeticError("The in-shape is not supported")
        return 'linear', res
    
    def comp_linear_local_shape(self, meta):
        t, ind = meta
        assert t == 'linear'
        res = np.zeros((self.nh*self.nw, 2))
        for i in range(self.npart):
            res[i] = ind[i+1] - ind[i]
        return res.reshape((self.nh, self.nw, 2))
    
    
    # getters for local shape and offset
    
    def get_local_data_shape(self, layer_idx:int, hid:int, wid:int):
        pass
        
    def get_local_data_shape_conv(self, layer_idx:int, hid:int, wid:int):
        indh, indw = self.lshape_imp[layer_idx]
        shape = (indh[hid+1] - indh[hid], indw[wid+1] - indw[wid])
        return shape
        
    def get_local_data_offset_conv(self, layer_idx:int, hid:int, wid:int):
        indh, indw = self.lshape_imp[layer_idx]
        coord = (indh[hid], indw[wid], indh[hid+1], indw[wid+1])
        h, w = coord[2]-coord[0], coord[3]-coord[1]
        off = np.zeros((h, w), dtype=int)
        for i in range(h):
            off[i,:] = (indh[hid]+i)*self.nw + indw[wid] + np.arange(w)
        return off
    
    def get_local_data_shape_linear(self, layer_idx:int, hid:int, wid:int):
        sid = hid*self.nw + wid
        igshape = self.gshapes[layer_idx][0]
        #ogshape = self.gshapes[layer_idx+1][0]
        ls = (igshape - sid + self.npart - 1) // self.npart
        return ls
        
    def get_local_data_offset_linear(self, layer_idx:int, hid:int, wid:int):
        sid = hid*self.nw + wid
        igshape = self.gshapes[layer_idx][0]
        #ogshape = self.gshapes[layer_idx+1][0]
        return np.arange(sid, igshape, self.npart)
   
    