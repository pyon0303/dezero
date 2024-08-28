import numpy as np
from dezero.utils import get_conv_outsize, pair
from dezero import Function, Parameter, as_variable
from dezero.functions import linear

def im2col(x, kernel_size, stride, pad, to_matrix=True):
    y = Im2col(kernel_size, stride, pad, to_matrix)(x)
    return y

class Im2col(Function):
    def __init__(self, kernel_size, stride, pad, to_matrix):
        super().__init__()
        self.input_shape = None
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.to_matrix = to_matrix
    
    def forward(self, x):
        self.input_shape = x.shape
        y = im2col_array(x, self.kernel_size, self.stride, self.pad, self.to_matrix)
        
        return y
    
    def backward(self, gy):
        gx = gy
        return gx


def im2col_array(img, kernel_size, stride, pad, to_matrix=True):
    N, C, H, W = img.shape # N == number of data (batch size)
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)
    
    img = np.pad(img, ((0, 0), (0, 0), (PH, PH + SH - 1), (PW, PW + SW - 1)), mode='constant', constant_values=(0,))
    col = np.ndarray((N, C, KH, KW, OH, OW), dtype=img.dtype)
    
    for j in range(KH):
        j_lim = j + SH * OH
        for i in range(KW):
            i_lim = i + SW * OW
            col[:, :, j, i, :, :] = img[:, :, j:j_lim:SH, i:i_lim:SW]
    
    if to_matrix:
        tmp = col.transpose((0, 4, 5, 1, 2, 3))
        col = tmp.reshape((N * OH * OW, -1))
    
    return col

def conv2d_simple(x, W, b=None, stride=1, pad=0):
    x, W = as_variable(x), as_variable(W)
    
    Weight = W
    N, C, H, W = x.shape
    OC, C, KH, KW = Weight.shape
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)
    
    col = im2col(x, (KH, KW), stride, pad, to_matrix=True)
    Weight = Weight.reshape(OC, -1).transpose()
    t = linear(col, Weight, b)
    tmp = t.reshape(N, OH, OW, OC)
    y = tmp.transpose(0, 3, 1, 2)
    return y

def pooling_simple(x, kernel_size, stride=2, pad=0):
    x = as_variable(x)
    
    N, C, H ,W = x.shape
    KH, KW = pair(kernel_size)
    PH, PW = pair(pad)
    SH, SW = pair(stride)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)
    
    col = im2col(x, kernel_size, stride, pad, to_matrix=True)
    col = col.reshape(-1, KH * KW)
    y = col.max(axis=1)
    y = y.reshape(N, OH, OW, C).transpose(0, 3, 1, 2)
    return y
    
    