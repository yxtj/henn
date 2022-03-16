# -*- coding: utf-8 -*-

import numpy as np
import warnings
import heutil

# %% helper for display

def _addindent(s_, numSpaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s

# %% HE layers

class ENNLayer():
    def forward(self, x):
        return x
    
    def extra_repr(self):
        return ""
    
    def get_modules(self):
        modules = {}
        for k, v in self.__dict__.items():
            if isinstance(v, ENNLayer):
                modules[k] = v
        return modules
    
    def __repr__(self):
        extra_repr = self.extra_repr()
        extra_lines = [] if not extra_repr else extra_repr.split('\n')
        child_lines = []
        for key, module in self.get_modules().items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
        lines = extra_lines + child_lines
        
        main_str = self.__class__.__name__ + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'
        main_str += ')'
        return main_str
    
    def __call__(self, x):
        return self.forward(x)
    
    
class Identity(ENNLayer):
    def forward(self, x):
        return x
    
    def __repr__(self):
        return "Identity()"


class ENNLayer2dBase(ENNLayer):
    def __init__(self, kernel_size, stride=1, padding=0, ceil_mode=False):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.size = kernel_size[0]*kernel_size[1]
        self.factor = 1.0 / self.size
        if isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride
        if isinstance(padding, int):
            padding = (padding, padding)
        self.padding = padding
        self.ceil_mode = ceil_mode

    def comp_out_size(self, sx, sy):
        ox = (sx+2*self.padding[0]-self.kernel_size[0]) / self.stride[0] + 1
        oy = (sy+2*self.padding[1]-self.kernel_size[1]) / self.stride[1] + 1
        if self.ceil_mode:
            ox = int(np.ceil(ox))
            oy = int(np.ceil(oy))
        else:
            ox = int(np.floor(ox))
            oy = int(np.floor(oy))
        return ox, oy

# %% linear layers

class Linear(ENNLayer):
    def __init__(self, in_ch:int, out_ch:int, bias=True):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.w_bias = bias
        self.weight = None
        self.bias = None
        #self.weight = (np.random.random((in_ch, out_ch)) - 0.5)
        #self.bias = (np.random.random(out_ch) - 0.5)
    
    def extra_repr(self):
        return f"in={self.in_ch}, out={self.out_ch}, bias={self.w_bias}"

    def bind(self, weight:np.ndarray, bias:np.ndarray):
        assert weight.ndim == 2
        assert (self.w_bias == False and bias is None) or \
            (self.w_bias == True and bias.ndim == 1)
        assert self.w_bias == False or weight.shape[0] == bias.shape[0]
        if weight.shape != (self.out_ch, self.in_ch):
            warnings.warn('shape not match: (%d,%d) vs ref (%d,%d)' %
                          (self.out_ch, self.in_ch, *weight.shape))
        self.out_ch, self.in_ch = weight.shape
        self.weight = weight
        self.bias = weight
        
    def forward(self, x):
        #return np.dot_product(self.weight, x) + self.bias
        out = heutil.dot_product_21(self.weight, x)
        if self.bias:
            out += self.bias
        return out


# %% activation layers

class ActivationSquare(ENNLayer):
    def forward(self, x):
        return x * x
    

class ActivationTanh3(ENNLayer):
    def forward(self, x):
        # tanh'(x) = 1 - tanh(x)^2
        # taylor series on odd x: 1, -1/3, 2/15, -17/315, 62/2835, -1382/155925
        x2 = x * x
        x4 = x2 * x2
        return x * ( 1 - 1/3*x2 + 2/15*x4)


class ActivationReLU(ENNLayer):
    def forward(self, x):
        return (x + heutil.sign(x)*x) / 2


# %% pooling layers

class Pooling2dBase(ENNLayer2dBase):
    def __init__(self, kernel_size, stride=1, padding=0, ceil_mode=False):
        super().__init__(kernel_size, stride, padding, ceil_mode)
    
    def pick_data(self, x, ch, i, j):
        data = x[ch]
        sx, sy = data.shape
        a1 = -self.padding[0] + i*self.stride[0]
        a2 = a1 + self.kernel_size[0]
        a1 = max(0, a1)
        a2 = min(a2, sx)
        b1 = -self.padding[1] + j*self.stride[1]
        b2 = b1 + self.kernel_size[1]
        b1 = max(0, b1)
        b2 = min(b2, sy)
        return data[a1:a2,b1:b2].ravel()


class AvgPool2d(Pooling2dBase):
    def __init__(self, kernel_size, stride=None, padding=0,
                 ceil_mode=False, count_include_pad=True):
        if stride is None:
            stride = kernel_size
        super().__init__(kernel_size, stride, padding, ceil_mode)
        self.count_include_pad = count_include_pad
    
    def extra_repr(self):
        return f"kernel={self.kernel_size}, stride={self.stride}, "\
            f"padding={self.padding}, ceil_mode={self.ceil_mode}"
        
    def forward(self, x):
        nch, sx, sy = x.shape
        ox, oy = self.comp_out_size(sx, sy)
        out = np.empty((nch, ox, oy), x.dtype)
        for ch in range(nch):
            for i in range(ox):
                for j in range(oy):
                    o = self.pick_data(x, ch, i, j)
                    #s = o.sum()
                    s = heutil.sum_list(o)
                    if self.count_include_pad or o.size == self.size:
                        out[ch, i,j] = s*self.factor
                    else:
                        f = 1.0 / o.size
                        out[ch, i,j] = s*f
        return out
 
 
class MaxPool2d(Pooling2dBase):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False):
        if stride is None:
            stride = kernel_size
        super().__init__(kernel_size, stride, padding, ceil_mode)
    
    def extra_repr(self):
        return f"kernel={self.kernel_size}, stride={self.stride}, "\
            f"padding={self.padding}, ceil_mode={self.ceil_mode}"
    
    def forward(self, x):
        # max = ( (a+b) + sign(a-b)*(a-b) )/2
        nch, sx, sy = x.shape
        ox, oy = self.comp_out_size(sx, sy)
        out = np.empty((nch, ox, oy), x.dtype)
        for ch in range(nch):
            for i in range(ox):
                for j in range(oy):
                    o = self.pick_data(x, ch, i, j)
                    out[ch, i,j] = heutil.max_list(o, quick=True)
        return out

# adaptive pooling

def adaptive_pool_idx(n_input, n_output):
    start_points = np.floor(np.arange(
        n_output, dtype=np.float32) * n_input / n_output).astype(int)
    end_points = np.ceil((np.arange(
        n_output, dtype=np.float32)+1) * n_input / n_output).astype(int)
    return [*zip(start_points, end_points)]


class AdaptiveAvgPool2d(ENNLayer):
    def __init__(self, output_size):
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        assert len(output_size) == 2
        self.output_size = output_size
        
    def extra_repr(self):
        return f"output_size={self.output_size}"
        
    def forward(self, x):
        nc, sx, sy = x.shape
        ox = sx if self.output_size[0] is None else self.output_size[0]
        rng_x = adaptive_pool_idx(sx, ox)
        oy = sy if self.output_size[1] is None else self.output_size[1]
        if sx == sy and ox == oy:
            rng_y = rng_x
        else:
            rng_y = adaptive_pool_idx(sy, oy)
        out = np.empty([nc, *self.output_size], x.dtype)
        for ch in range(nc):
            for i, (xf, xl) in enumerate(rng_x):
                for j, (yf, yl) in enumerate(rng_y):
                    t = heutil.avg_list(x[ch, xf:xl, yf:yl])
                    out[ch,i,j] = t
        return out


# %% convolution layers
class Conv2d(ENNLayer2dBase):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0,
                 groups=1, bias=True):
        super().__init__(kernel_size, stride, padding, False)
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.groups = groups
        assert in_ch % groups == 0
        self.in_ch_pg = in_ch // groups # in channles per group
        self.out_ch_pg = out_ch // groups
        self.w_bias = bias
        self.weight = None
        self.bias = None
    
    def extra_repr(self):
        return f"{self.in_ch}, {self.out_ch}, "\
            f"kernel_size={self.kernel_size}), stride={self.stride}, "\
            f"padding={self.padding}, bias={self.w_bias}"
    
    def bind(self, weight:np.ndarray, bias:np.ndarray, groups:int=1,
             stride:int=None, padding:int=None):
        assert weight.ndim == 4
        assert weight.shape == (self.out_ch, self.in_ch_pg, *self.kernel_size)
        assert (self.w_bias == False and bias is None) or \
            (self.w_bias == True and bias.ndim == 1)
        assert self.w_bias == False or bias.shape[0] == self.out_ch
        if self.groups != groups:
            warnings.warn("number of groups does not match: (%d) vs ref (%f)" %
                          (self.groups, groups))
            self.groups = groups
        out_ch_pg = weight.shape[0]
        out_ch = out_ch_pg*groups
        if out_ch_pg != self.out_ch_pg:
            warnings.warn('out channels do not match: (%d,%d) vs ref (%d,%d)' %
                          (self.out_ch, self.out_ch_pg, out_ch, out_ch_pg))
            self.out_ch = out_ch
            self.out_ch_pg = out_ch_pg
        in_ch_pg = weight.shape[1]
        in_ch = in_ch_pg*groups
        if in_ch_pg != self.in_ch_pg:
            warnings.warn('in channels do not match: (%d,%d) vs ref (%d,%d)' %
                          (self.in_ch, self.in_ch_pg, in_ch, in_ch_pg))
            self.in_ch = in_ch
            self.in_ch_pg = in_ch_pg
        kernel_size = weight.shape[2:]
        if self.kernel_size != kernel_size:
            warnings.warn('kernel size does not match: (%d,%d) vs ref (%d,%d)' %
                          (*self.kernel_size, *kernel_size))
            self.kernel_size = kernel_size
            self.size = kernel_size[0] * kernel_size[1]
            self.factor = 1.0/self.size
        # stride and padding is not mandatory
        if stride is not None and stride != self.stride:
            warnings.warn('stride does not match: %d vs ref %d' %
                          (self.stride, stride))
            self.stride = stride
        if padding is not None and padding != self.padding:
            warnings.warn('padding does not match: %d vs ref %d' %
                          (self.padding, padding))
            self.padding = padding
        self.weight = weight
        self.bias = bias
    
    def conv(self, x, ch, i, j):
        _, nx, ny = x.shape
        g = ch // self.out_ch_pg # tell group by out-channel
        ch_f = g * self.in_ch_pg
        ch_l = ch_f + self.in_ch_pg
        #print(g, ch_f, ch_l)
        i_f = i * self.stride[0] - self.padding[0]
        i_l = i_f + self.kernel_size[0]
        wi_f = -i_f if i_f < 0 else 0
        wi_l = nx - i_f if i_l > nx else self.kernel_size[0]
        j_f = j * self.stride[1] - self.padding[1]
        j_l = j_f + self.kernel_size[1]
        wj_f = -j_f if j_f < 0 else 0
        wj_l = ny - j_f if j_l > ny else self.kernel_size[1]
        #print(g, i_f, i_l, wi_f, wi_l, '-', j_f, j_l, wj_f, wj_l)
        #i_f, j_f = max(0, i_f), max(0, j_f)
        #i_l, j_l = min(nx, i_l), min(ny, j_l)
        cut = x[ch_f:ch_l, max(0, i_f):i_l, max(0, j_f):j_l]
        cut = cut*self.weight[ch, :, wi_f:wi_l, wj_f:wj_l]
        #r = cut.sum() + float(self.bias[ch])
        r = heutil.sum_list(cut) + float(self.bias[ch])
        return r
    
    def forward(self, x):
        assert x.ndim == 3
        in_ch, sx, sy = x.shape
        ox, oy = self.comp_out_size(sx, sy)
        out = np.empty((self.out_ch, ox, oy), x.dtype)
        for ch in range(self.out_ch):
            for i in range(ox):
                for j in range(oy):
                    out[ch,i,j] = self.conv(x, ch, i, j)
        return out

# %% normalization

class BatchNorm2d(ENNLayer):
    def __init__(self, num_feat, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_feat = num_feat
        self.eps = eps
        self.momentum = momentum
        self.r = 1.0
        self.b = 0.0
        
    def extra_repr(self):
        return f"{self.num_feat}, eps={self.eps}, momentum={self.momentum}"
    
    def reset(self):
        self.r = 1.0
        self.b = 0.0
        
    def forward(self, x):
        shape = x.shape
        assert x.ndim == 3 and shape[0] == self.num_feat
        e = heutil.avg_list(x)
        v = heutil.var_list(x, e)
        up = x - e
        down = heutil.sqrt(v + self.eps)
        y = up / down * self.r + self.b
        return y

    
class LocalResponseNorm2d(ENNLayer):
    def __init__(self, size, alpha=1e-4, beta=0.75, k=1.0):
        super().__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k
    
    def extra_repr(self):
        return "{self.size}, alpha={self.alpha}, beta={self.beta}, k={self.k}"
    
    def forward(self, x):
        nch, nx, ny = x.shape
        d = x*x
        res = np.empty(x.shape, x.dtype)
        for ch in range(nch):
            c_f = max(0, ch-self.size//2)
            c_l = min(nch, ch+self.size//2)
            factor = self.alpha/(c_l-c_f)
            for i in range(nx):
                for j in range(ny):
                    r = heutil.sum_list(d[c_f:c_l,i,j].ravel())*factor + self.k
                    r = r**(-self.beta) # TODO
                    res[ch,i,j] = x[ch,i,j]*r
        return res
                
# %% sequential

class Sequential(ENNLayer):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
    
    def extra_repr(self):
        body = "\n".join([f"({i}): {l}" for i,l in enumerate(self.layers)])
        return body
    
    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x


# %% test

def test_conv():
    conv = Conv2d(3, 64, 11, 4)
    conv.weight = np.random.random((64,3,11,11))
    conv.bias = np.random.random(64)
    a = np.random.random((3, 244, 244))
    o = conv(a)
    

if __name__ == '__main__':
    test_conv()
    
    
