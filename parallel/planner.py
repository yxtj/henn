# -*- coding: utf-8 -*-

import numpy as np
from .phenetwork import PhenLayer

class Planner:
    def __init__(self, nh:int, nw:int, model:list[PhenLayer], inshape:tuple):
        self.nh = nh
        self.nw = nw
        self.model = model
        self.shapes = self.comp_size(model, inshape)
        
    def comp_size(self, model:list[PhenLayer], inshape:tuple):
        shapes = []
        s = inshape
        shapes.append(s)
        for layer in model:
            s = layer.out_shape(s)
            shapes.append(s)
        return shapes
